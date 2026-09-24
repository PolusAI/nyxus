#define NOMINMAX
#include <iostream>
#include "nyxus_dicom_loader.h"
#include "image_loader.h"
#include "grayscale_tiff.h"
#include "raw_tiff.h"
#include "omezarr.h"
#include "dirs_and_files.h"
#include "helpers/fsystem.h"
#include "raw_nifti.h"
#include "ome/format_detect.h"		// container-family classification for loader dispatch
#include "mask_plane.h"
#include "volume_walk.h"

ImageLoader::ImageLoader() {}

bool ImageLoader::open (SlideProps & p, const FpImageOptions & fpopts)
{
	int n_threads = 1;

	std::string & int_fpath = p.fname_int,
		& seg_fpath = p.fname_seg;

	// A newly opened pair is read at its first plane until stream_volume_planes() selects another
	cur_channel = 0;
	cur_timeframe = 0;

	// intensity image

	try 
	{
		// Classify by container family so loader dispatch is identical across all loaders.
		Nyxus::ContainerKind fmt = Nyxus::detect_container_family (int_fpath);

		// The map the scan recorded, in the terms every tile loader takes. The quantized branch
		// spans [inten_offset, inten_offset + inten_scale*DR], which is the [fpmin, fpmax] the scan
		// recorded (the fp overrides included). The offset branch ignores fpmax and shifts by
		// inten_offset alone. Derived once so no backend can drift from another.
		bool quantize = p.inten_map == IntenMap::quantized;
		double dr = fpopts.target_dyn_range(),
			fpmin = p.inten_offset,
			fpmax = quantize ? p.inten_offset + p.inten_scale * dr : p.max_preroi_inten;

		// How the offset branch narrows. preserve_hu exists to carry absolute intensities, so it
		// rounds to nearest: inten_scale is 1 there and the inverse cannot recover a fraction a
		// truncating cast drops, which is what a fractional DICOM RescaleSlope or NIfTI scl_slope
		// produces. Without the flag a real-valued slide truncates, as it always has.
		// SlideProps::to_grey_level() reads p.preserve_hu directly and narrows the same way.
		bool round_offset = p.preserve_hu;

		// The header-rescaled formats take one of two maps. On the stored map the loader leaves the
		// rescale off and shifts the stored integers by inten_stored_shift, so nothing is narrowed;
		// on the offset map it rescales to physical units and shifts by inten_offset.
		bool stored = p.inten_map == IntenMap::stored;
		double medical_shift = stored ? p.inten_stored_shift : p.inten_offset;
		bool medical_rescale = ! stored;

		if (fmt == Nyxus::ContainerKind::OmeZarr)
		{
			#ifdef OMEZARR_SUPPORT
				// Zarr takes the same map as TIFF, so a signed dataset keeps its negatives and a
				// real-valued one its fraction rather than wrapping or truncating into the
				// unsigned destination type.
				intFL = new NyxusOmeZarrLoader<uint32_t>(n_threads, int_fpath, fpmin, fpmax, dr, quantize, round_offset);
			#else
				std::string erm = "This version of Nyxus was not build with OmeZarr support";
				#ifdef WITH_PYTHON_H
					throw std::runtime_error (erm);
				#endif	
				std::cerr << erm << "\n";
			#endif
		}
		else
			if (fmt == Nyxus::ContainerKind::Dicom)
			{
				#ifdef DICOM_SUPPORT
					// A DICOM slide carries physical units through its rescale tags, so it takes the
					// stored map, or the offset map under preserve_hu.
					intFL = new NyxusGrayscaleDicomLoader<uint32_t>(n_threads, int_fpath, medical_shift, medical_rescale, round_offset);
				#else
					std::string erm = "This version of Nyxus was not build with DICOM support";
					#ifdef WITH_PYTHON_H
						throw std::runtime_error(erm);
					#endif	
						std::cerr << erm << "\n";
					#endif
			}
			else
				if (fmt == Nyxus::ContainerKind::Nifti)
				{
					// Same as DICOM for an integer volume. A real-valued one never takes the stored
					// map, so it rescales and shifts by the offset the scan recorded.
					intFL = new NiftiLoader<uint32_t> (int_fpath, medical_shift, medical_rescale, round_offset);
				}
				else 
				{
					// flavors of TIFF (TIFF, OME.TIFF)

					if (Nyxus::check_tile_status(int_fpath))
					{
						intFL = new NyxusGrayscaleTiffTileLoader<uint32_t> (
							n_threads,
							int_fpath,
							true,
							fpmin,
							fpmax,
							dr,
							quantize,
							round_offset);
					} 
					else 
					{
						intFL = new NyxusGrayscaleTiffStripLoader<uint32_t>(n_threads, int_fpath, fpmin, fpmax, dr, quantize, round_offset);
					}
				}
	}
	catch (std::exception const& e)	
	{
		std::cout << "Error while initializing the image loader for intensity image file " << int_fpath << ": " << e.what() << "\n";
		return false;
	}

	if (intFL == nullptr)
		return false;

	// Intensity slide
	th = intFL->tileHeight (lvl);
	tw = intFL->tileWidth (lvl);
	td = intFL->tileDepth (lvl);
	tt = intFL->tileTimestamps (lvl);

	tileSize = th * tw * td * tt;

	fh = intFL->fullHeight (lvl);
	fw = intFL->fullWidth (lvl);
	fd = intFL->fullDepth (lvl);
	ft = intFL->fullTimestamps (lvl);

	ntw = intFL->numberTileWidth(lvl);
	nth = intFL->numberTileHeight(lvl);
	ntd = intFL->numberTileDepth(lvl);

	ptrI = std::make_shared<std::vector<uint32_t>>(tileSize);

	// wholeslide
	if (seg_fpath.empty())
		return true;

	// segmented slide

	try 
	{
		// The mask is classified by the same container family as the intensity, so an
		// .ome.zarr mask routes to the Zarr loader (not the TIFF fallback).
		Nyxus::ContainerKind fmt = Nyxus::detect_container_family (seg_fpath);

		if (fmt == Nyxus::ContainerKind::OmeZarr)
		{
			#ifdef OMEZARR_SUPPORT
				segFL = new NyxusOmeZarrLoader<uint32_t>(n_threads, seg_fpath);		// a mask carries labels, not physical units: offset 0, no quantization
			#else
				std::cout << "This version of Nyxus was not build with OmeZarr support." <<std::endl;
			#endif
		}
		else
			if (fmt == Nyxus::ContainerKind::Dicom)
			{
				#ifdef DICOM_SUPPORT
					segFL = new NyxusGrayscaleDicomLoader<uint32_t>(n_threads, seg_fpath, 0.0, false);		// a mask carries labels, not physical units
				#else
					std::cout << "This version of Nyxus was not build with DICOM support." <<std::endl;
				#endif
			}
			else
				if (fmt == Nyxus::ContainerKind::Nifti)
				{
					segFL = new NiftiLoader <uint32_t> (seg_fpath, 0.0, false);		// a mask carries labels, not physical units
				}
				else
				{
					// flavors of TIFF

					if (Nyxus::check_tile_status(seg_fpath))
					{
						segFL = new NyxusGrayscaleTiffTileLoader<uint32_t>(
							n_threads, 
							seg_fpath, 
							false,
							0.0, // dummy min
							1.0, // dummy max
							fpopts.target_dyn_range());
					} 
					else 
					{
						segFL = new NyxusGrayscaleTiffStripLoader<uint32_t>(n_threads, seg_fpath);
					}
				}
	}
	catch (std::exception const& e)	
	{
		std::cout << "Error while initializing the image loader for mask image file " <<  seg_fpath << ": " << e.what() << "\n";
		return false;
	}

	if (segFL == nullptr)
		return false;

	// File #2 (labels)

	// -- check whole file consistency
	auto fh_seg = segFL->fullHeight(lvl),
		fw_seg = segFL->fullWidth(lvl), 
		fd_seg = segFL->fullDepth(lvl);
	if (fh != fh_seg || fw != fw_seg || fd != fd_seg)
	{
		std::cout << "\terror: INT: " << int_fpath << " SEG: " << seg_fpath << " :  mismatch in full height, width, or depth FH " << fh << ":" << fh_seg << " FW " << fw << ":" << fw_seg << " FD " << fd << ":" << fd_seg << "\n";
		return false;
	}

	// -- check tile consistency
	auto th_seg = segFL->tileHeight(lvl),
		tw_seg = segFL->tileWidth(lvl),
		td_seg = segFL->tileDepth(lvl);
	if (th != segFL->tileHeight(lvl) || tw != segFL->tileWidth(lvl) || td != segFL->tileDepth(lvl))
	{
		std::cout << "\terror: INT: " << int_fpath << " SEG: " << seg_fpath << " :  mismatch in tile height, width, or depth TH " << th << ":" << th_seg << " TW " << tw << ":" << tw_seg << " TD " << td << ":" << td_seg << "\n";
		return false;
	}

	ptrL = std::make_shared<std::vector<uint32_t>>(tileSize);

	return true;
}

void ImageLoader::close()
{
	if (segFL)
	{
		delete segFL;
		segFL = nullptr;
	}

	if (intFL)
	{
		delete intFL;
		intFL = nullptr;
	}
}

bool ImageLoader::load_tile(size_t tile_idx)
{
	if (tile_idx >= ntw * nth * ntd)
		return false;

	auto tRow = tile_idx / ntw;
	auto tCol = tile_idx % ntw;
	
	intFL->loadTileFromFile (ptrI, tRow, tCol, lyr, cur_channel, cur_timeframe, lvl);

	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (ptrL, tRow, tCol, lyr, cur_channel, cur_timeframe, lvl);

	return true;
}

bool ImageLoader::load_tile (size_t tile_row, size_t tile_col)
{
	if (tile_row >= nth || tile_col >= ntw)
		return false;

	intFL->loadTileFromFile (ptrI, tile_row, tile_col, lyr, cur_channel, cur_timeframe, lvl);

	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (ptrL, tile_row, tile_col, lyr, cur_channel, cur_timeframe, lvl);

	return true;
}

size_t ImageLoader::assemble_tile_layer (AbstractTileLoader<uint32_t>* fl,
	std::shared_ptr<std::vector<uint32_t>>& ptr,
	std::vector<std::vector<uint32_t>>& planes, size_t lz, size_t channel, size_t timeframe)
{
	const Nyxus::VolumeGrid grid = Nyxus::volume_grid_of (*fl, lvl);
	const size_t frameBase = grid.frame_base (timeframe),
		z0 = lz * grid.tile_d,
		depth = (std::min) (grid.tile_d, fd - z0);

	if (planes.size() < depth)
		planes.resize (depth);
	for (size_t k = 0; k < depth; k++)
		planes[k].assign ((size_t) fw * fh, 0u);

	Nyxus::walk_tile_layer (grid, lz,
		[&](size_t tr, size_t tc, size_t lz_)
		{
			fl->loadTileFromFile (ptr, tr, tc, lz_, channel, timeframe, lvl);
		},
		[&](size_t src, size_t x0, size_t y, size_t z, size_t n)
		{
			auto from = ptr->begin() + frameBase + src;
			std::copy (from, from + n, planes[z - z0].begin() + y * fw + x0);
		},
		[]() {});

	return depth;
}

// Planes one read of 'fl' delivers: the entire x*y*z*t blob on a loader that keeps the whole
// time series in its tile (NIfTI), otherwise the tile layer, which the last layer of a volume
// may cut short. tileTimestamps is the base class's 1 on every loader that does not override
// it -- OME-Zarr reports its time extent through fullTimestamps and does not -- so it says
// "frames per read" only where a loader sets it.
static size_t planes_per_read_of (const AbstractTileLoader<uint32_t>* fl, int lvl)
{
	const size_t fd = fl->fullDepth (lvl);
	if (fl->tileTimestamps (lvl) > 1)
		return fd * fl->fullTimestamps (lvl);
	return (std::min) (fl->tileDepth (lvl), fd);
}

// True when one read of 'fl' hands back every plane a streaming pass would walk, so there is
// nothing smaller to stream. A pass walks one (channel, timeframe) volume, which is fullDepth
// planes; a read that covers them all, and is more than a single plane, leaves nothing to
// bound. Both the decision and the number a refusal reports come from planes_per_read_of, so
// they cannot drift apart.
//
// Each loader is measured against its own extents: an intensity volume and its mask need not
// have the same number of time frames.
static bool tile_is_whole_volume (const AbstractTileLoader<uint32_t>* fl, int lvl)
{
	const size_t per_read = planes_per_read_of (fl, lvl);
	return per_read > 1 && per_read >= fl->fullDepth (lvl);
}

bool ImageLoader::unstreamable_read (size_t& planes, bool& of_mask) const
{
	if (tile_is_whole_volume (intFL, lvl))
	{
		planes = planes_per_read_of (intFL, lvl);
		of_mask = false;
		return true;
	}
	if (segFL != nullptr && tile_is_whole_volume (segFL, lvl))
	{
		planes = planes_per_read_of (segFL, lvl);
		of_mask = true;
		return true;
	}
	return false;
}

bool ImageLoader::streams_bounded() const
{
	size_t planes = 0;
	bool of_mask = false;
	return ! unstreamable_read (planes, of_mask);
}
void ImageLoader::stream_volume_planes (size_t channel, size_t timeframe,
	const std::function<void(size_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&)>& sink)
{
	cur_channel = channel;
	cur_timeframe = timeframe;

	// the mask plane that pairs with this (channel, timeframe)
	size_t mask_channel = 0, mask_tf = 0;
	if (segFL != nullptr)
		Nyxus::mask_plane_for (channel, timeframe, intFL->fullTimestamps (lvl),
			segFL->numberChannels(), segFL->fullTimestamps (lvl), mask_channel, mask_tf);

	// open() requires the mask's tile grid to be the intensity's, so both assemble the same
	// planes from each tile layer
	const std::vector<uint32_t> noMask;
	std::vector<std::vector<uint32_t>> intPlanes, segPlanes;
	const size_t tileLayers = intFL->numberTileDepth (lvl),
		tileDepth = intFL->tileDepth (lvl);

	for (size_t lz = 0; lz < tileLayers; lz++)
	{
		const size_t depth = assemble_tile_layer (intFL, ptrI, intPlanes, lz, channel, timeframe);
		if (segFL != nullptr)
			assemble_tile_layer (segFL, ptrL, segPlanes, lz, mask_channel, mask_tf);

		for (size_t k = 0; k < depth; k++)
			sink (lz * tileDepth + k, intPlanes[k], segFL != nullptr ? segPlanes[k] : noMask);
	}
}

const std::vector<uint32_t>& ImageLoader::get_int_tile_buffer()
{
	return *ptrI;
}

const std::vector<uint32_t>& ImageLoader::get_seg_tile_buffer()
{
	return *ptrL;
}

const std::shared_ptr<std::vector<uint32_t>> & ImageLoader::get_seg_tile_sptr()
{
	return ptrL;
}

size_t ImageLoader::get_tile_size()
{
	return tileSize;
}

size_t ImageLoader::get_tile_x (size_t pixel_col)
{
	size_t tx = pixel_col / tw;
	return tx;
}

size_t ImageLoader::get_tile_y (size_t pixel_row)
{
	size_t ty = pixel_row / th;
	return ty;
}

size_t ImageLoader::get_within_tile_idx (size_t pixel_row, size_t pixel_col)
{
	size_t wtx = pixel_col % tw,
		wty = pixel_row % th,
		idx = wty * tw + wtx;
	return idx;
}

size_t ImageLoader::get_num_tiles_vert()
{
	return nth;
}

size_t ImageLoader::get_num_tiles_hor()
{
	return ntw;
}

size_t ImageLoader::get_tile_height()
{
	return th;
}

size_t ImageLoader::get_tile_width()
{
	return tw;
}

size_t ImageLoader::get_full_width()
{
	return fw;
}

size_t ImageLoader::get_full_height()
{
	return fh;
}

size_t ImageLoader::get_full_depth()
{
	return fd;
}

size_t ImageLoader::get_inten_time()
{
	return intFL->fullTimestamps(0);
}

size_t ImageLoader::get_mask_time()
{
	if (segFL)
		return segFL->fullTimestamps(0);	// masked mode
	else
		return 0;	// whole-slide mode
}
