#include <iostream>

#define NOMINMAX 

#include "dirs_and_files.h"
#include "helpers/fsystem.h"
#include "raw_image_loader.h"
#include "raw_dicom.h"
#include "raw_nifti.h"
#include "raw_omezarr.h"
#include "raw_tiff.h"
#include "ome/format_detect.h"		// FIX: unified content-sniffing loader dispatch

RawImageLoader::RawImageLoader() {}

bool RawImageLoader::open (const std::string& int_fpath, const std::string& seg_fpath, bool preserve_hu)
{
	try
	{
		// FIX: classify via detect_input_format(). Defects fixed: (1) matched only ".zarr" so
		// ".ome.zarr" mis-routed to TIFF; (2) `ext==".dcm" | ext==".dicom"` used bitwise-OR (works by luck).
		Nyxus::InputFormat fmt = Nyxus::detect_input_format (int_fpath);

		if (fmt.kind == Nyxus::ContainerKind::OmeZarr)		// FIX: was `ext==".zarr"` only (dropped .ome.zarr)
		{
#ifdef OMEZARR_SUPPORT
			intFL = new RawOmezarrLoader (int_fpath);
#else
			std::cout << "This version of Nyxus was not build with OmeZarr support." << std::endl;
#endif
		}
		else
			if (fmt.kind == Nyxus::ContainerKind::Dicom) {		// FIX: was bitwise `ext==".dcm" | ext==".dicom"`
#ifdef DICOM_SUPPORT
			intFL = new RawDicomLoader (int_fpath, preserve_hu);		// CT/HU: scan in Hounsfield domain
#else
			std::cout << "This version of Nyxus was not build with DICOM support." << std::endl;
#endif
		}
			else
				if (fmt.kind == Nyxus::ContainerKind::Nifti)		// FIX: was `ext==".nii"||".nii.gz"`
				{
					intFL = new RawNiftiLoader (int_fpath, preserve_hu);		// FIX: CT/HU: scan in Hounsfield domain (matches DICOM)
				}
				else
				{
					// flavors of TIFF
					if (Nyxus::check_tile_status(int_fpath))
					{
 						intFL = new RawTiffTileLoader (int_fpath);
					}
					else
					{
						intFL = new RawTiffStripLoader (1/*n_threads*/, int_fpath);
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

	// File #1 (intensity)
	th = intFL->tileHeight(lvl);
	tw = intFL->tileWidth(lvl);
	td = intFL->tileDepth(lvl);
	tileSize = th * tw;

	fh = intFL->fullHeight(lvl);
	fw = intFL->fullWidth(lvl);
	fd = intFL->fullDepth(lvl);

	ntw = intFL->numberTileWidth(lvl);
	nth = intFL->numberTileHeight(lvl);
	ntd = intFL->numberTileDepth(lvl);

	// wholeslide
	if (seg_fpath.empty())
		return true;

	// segmented slide

	try {
		// FIX: unified seg dispatch — same defects as the intensity path (".ome.zarr" dropped,
		// bitwise-OR on the DICOM test) fixed here via detect_input_format().
		Nyxus::InputFormat fmt = Nyxus::detect_input_format (seg_fpath);
		if (fmt.kind == Nyxus::ContainerKind::OmeZarr)		// FIX: was `ext==".zarr"` only (dropped .ome.zarr)
		{
#ifdef OMEZARR_SUPPORT
			segFL = new RawOmezarrLoader (seg_fpath);
#else
			std::cout << "This version of Nyxus was not build with OmeZarr support." << std::endl;
#endif
		}
		else
			if (fmt.kind == Nyxus::ContainerKind::Dicom)		// FIX: was bitwise `ext==".dcm" | ext==".dicom"`
			{
#ifdef DICOM_SUPPORT
				segFL = new RawDicomLoader (seg_fpath);
#else
				std::cout << "This version of Nyxus was not build with DICOM support." << std::endl;
#endif
			}
			else
				if (fmt.kind == Nyxus::ContainerKind::Nifti)		// FIX: was `ext==".nii"||".nii.gz"`
				{
					segFL = new RawNiftiLoader (seg_fpath);
				}
				else
				{
					// flavors of TIFF
					if (Nyxus::check_tile_status(seg_fpath))
					{
						segFL = new RawTiffTileLoader(seg_fpath);
					}
					else
					{
						segFL = new RawTiffStripLoader (1/*n_threads*/, seg_fpath);
					}
				}
	}
	catch (std::exception const& e)
	{
		std::cout << "Error while initializing the image loader for mask image file " << seg_fpath << ": " << e.what() << "\n";
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
		std::cout << "\terror: INT: " << int_fpath << " SEG: " << seg_fpath << " :  mismatch in tile height, width, or dep-th TH " << th << ":" << th_seg << " TW " << tw << ":" << tw_seg << " TD " << td << ":" << td_seg << "\n";
		return false;
	}

	return true;
}

void RawImageLoader::free_tile_buffers()
{
	if (intFL)
		intFL->free_tile();
	if (segFL)
		segFL->free_tile();
}

void RawImageLoader::close()
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

bool RawImageLoader::load_tile(size_t tile_idx)
{
	if (tile_idx >= ntw * nth * ntd)
		return false;

	auto row = tile_idx / ntw;
	auto col = tile_idx % ntw;

	intFL->loadTileFromFile (row, col, lyr, cur_channel, cur_timeframe, lvl);

	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (row, col, lyr, cur_channel, cur_timeframe, lvl);

	return true;
}

bool RawImageLoader::load_tile(size_t tile_row, size_t tile_col)
{
	if (tile_row >= nth || tile_col >= ntw)
		return false;

	intFL->loadTileFromFile (tile_row, tile_col, lyr, cur_channel, cur_timeframe, lvl);

	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (tile_row, tile_col, lyr, cur_channel, cur_timeframe, lvl);

	return true;
}

uint32_t RawImageLoader::get_cur_tile_seg_pixel (size_t idx)
{
	return segFL->get_uint32_pixel (idx);
}

double RawImageLoader::get_cur_tile_dpequiv_pixel (size_t idx)
{
	return intFL->get_dpequiv_pixel (idx);
}

bool RawImageLoader::load_volume (size_t channel, size_t timeframe)
{
	const size_t sliceSize = fw * fh,
		volSize = sliceSize * fd;

	if (vol_int_.size() != volSize)
		vol_int_.resize (volSize);
	const bool haveSeg = (segFL != nullptr);
	if (haveSeg && vol_seg_.size() != volSize)
		vol_seg_.resize (volSize);

	// Use each loader's OWN layout: per-plane loaders (OME-Zarr, multi-page/OME-TIFF) deliver
	// one Z-plane per read (tileDepth==1); a whole-4D loader (NIfTI) delivers the entire
	// x*y*z*t blob in one read and ignores the layer arg, so the requested frame is slabbed
	// out via frameBase. Mirrors ImageLoader::assemble_volume.
	const size_t ltd = intFL->tileDepth (lvl),
		ltt = intFL->tileTimestamps (lvl),
		lntd = intFL->numberTileDepth (lvl),
		frameStride = ltd * th * tw,
		frameBase = (ltt > 1) ? timeframe * frameStride : 0;

	// The mask is channel-agnostic unless it genuinely has that many channels
	const size_t maskChannel = (haveSeg && channel < segFL->numberChannels()) ? channel : 0;

	for (size_t lz = 0; lz < lntd; lz++)
	{
		intFL->loadTileFromFile (0, 0, lz, channel, timeframe, lvl);
		if (haveSeg)
			segFL->loadTileFromFile (0, 0, lz, maskChannel, timeframe, lvl);

		for (size_t pz = 0; pz < ltd && (lz * ltd + pz) < fd; pz++)
		{
			const size_t gz = lz * ltd + pz;
			for (size_t row = 0; row < fh; row++)
				for (size_t col = 0; col < fw; col++)
				{
					const size_t src = frameBase + (pz * th + row) * tw + col,
						dst = gz * sliceSize + row * fw + col;
					vol_int_[dst] = intFL->get_dpequiv_pixel (src);
					if (haveSeg)
						vol_seg_[dst] = segFL->get_uint32_pixel (src);
				}
		}

		// the raw TIFF loaders malloc their tile buffer on each read (no-op elsewhere)
		intFL->free_tile();
		if (haveSeg)
			segFL->free_tile();
	}

	return true;
}

size_t RawImageLoader::get_tile_size()
{
	return tileSize;
}

size_t RawImageLoader::get_tile_x(size_t pixel_col)
{
	size_t tx = pixel_col / tw;
	return tx;
}

size_t RawImageLoader::get_tile_y(size_t pixel_row)
{
	size_t ty = pixel_row / th;
	return ty;
}

size_t RawImageLoader::get_within_tile_idx(size_t pixel_row, size_t pixel_col)
{
	size_t wtx = pixel_col % tw,
		wty = pixel_row % th,
		idx = wty * tw + wtx;
	return idx;
}

size_t RawImageLoader::get_num_tiles_vert()
{
	return ntw;
}

size_t RawImageLoader::get_num_tiles_hor()
{
	return nth;
}

size_t RawImageLoader::get_tile_height()
{
	return th;
}

size_t RawImageLoader::get_tile_width()
{
	return tw;
}

size_t RawImageLoader::get_full_width()
{
	return fw;
}

size_t RawImageLoader::get_full_height()
{
	return fh;
}

size_t RawImageLoader::get_full_depth()
{
	return fd;
}

size_t RawImageLoader::get_inten_time()
{
	return intFL->fullTimestamps(0);
}

size_t RawImageLoader::get_mask_time()
{
	if (segFL)
		return segFL->fullTimestamps(0);	// masked mode
	else
		return 0;	// whole-slide mode
}

size_t RawImageLoader::get_inten_channels()
{
	return intFL->numberChannels();		// FIX (IO): OME loaders report the real C; others default to 1
}

double RawImageLoader::get_physical_size_x()
{
	return intFL->physicalSizeX();		// FIX (IO): OME PhysicalSizeX; 1.0 if uncalibrated
}

double RawImageLoader::get_physical_size_y()
{
	return intFL->physicalSizeY();
}

double RawImageLoader::get_physical_size_z()
{
	return intFL->physicalSizeZ();
}

std::string RawImageLoader::get_physical_size_unit()
{
	return intFL->physicalSizeUnit();
}

std::string RawImageLoader::get_slide_descr()
{
	std::string s = get_fp_phys_pixvoxels() ? "R-" : "N-";
	s += intFL->get_infostring() + " I ";
	if (segFL != nullptr)
		s += segFL->get_infostring() + " M ";
	return s;
}

bool RawImageLoader::get_fp_phys_pixvoxels()
{
	return intFL->get_fp_pixels();
}

