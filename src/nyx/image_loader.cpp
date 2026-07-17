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
#include "ome/format_detect.h"		// FIX: unified content-sniffing loader dispatch

ImageLoader::ImageLoader() {}

bool ImageLoader::open (SlideProps & p, const FpImageOptions & fpopts)
{
	int n_threads = 1;

	std::string & int_fpath = p.fname_int,
		& seg_fpath = p.fname_seg;

	// intensity image
  
	try 
	{
		// FIX: classify by detect_input_format() (extension + OME content sniff) instead of
		// raw extension compares, so dispatch is identical across all 3 loaders and OME is recognized.
		Nyxus::InputFormat fmt = Nyxus::detect_input_format (int_fpath);

		if (fmt.kind == Nyxus::ContainerKind::OmeZarr)		// FIX: was `ext==".zarr"||".ome.zarr"`
		{
			#ifdef OMEZARR_SUPPORT
				intFL = new NyxusOmeZarrLoader<uint32_t>(n_threads, int_fpath);
			#else
				std::string erm = "This version of Nyxus was not build with OmeZarr support";
				#ifdef WITH_PYTHON_H
					throw std::runtime_error (erm);
				#endif	
				std::cerr << erm << "\n";
			#endif
		}
		else
			if (fmt.kind == Nyxus::ContainerKind::Dicom)		// FIX: was `ext==".dcm"||".dicom"`
			{
				#ifdef DICOM_SUPPORT
					// HU offset base must be the scanned (HU-domain) slide min. In preserve_hu
					// mode the slope-1 map bypasses fp min/max/dr, so we must NOT take
					// fpopts.min_intensity() (defaults to 0 when --fpimgmin is absent) — that
					// would clamp every negative HU to 0. Only the non-HU float path uses the
					// fp override min.
					intFL = new NyxusGrayscaleDicomLoader<uint32_t>(n_threads, int_fpath,
						(fpopts.preserve_hu() || fpopts.empty()) ? p.min_preroi_inten : (double)fpopts.min_intensity(),
						fpopts.preserve_hu());
				#else
					std::string erm = "This version of Nyxus was not build with DICOM support";
					#ifdef WITH_PYTHON_H
						throw std::runtime_error(erm);
					#endif	
						std::cerr << erm << "\n";
					#endif
			}
			else
				if (fmt.kind == Nyxus::ContainerKind::Nifti)		// FIX: was `ext==".nii"||".nii.gz"`
				{
					intFL = new NiftiLoader<uint32_t> (int_fpath,
							(fpopts.preserve_hu() || fpopts.empty()) ? p.min_preroi_inten : (double)fpopts.min_intensity(),		// HU offset base = scanned HU-domain slide min; ignore fp min in preserve_hu mode (else negative HU clamps to 0)
							fpopts.preserve_hu());		// CT/HU mode: offset-preserving map (matches DICOM/TIFF)
				}
				else 
				{
					// flavors of TIFF (TIFF, OME.TIFF)
					
					// automatic or overriden FP dynamic range
					double fpmin = p.min_preroi_inten,
						fpmax = p.max_preroi_inten;
					// Only the non-HU float path honors the fp override min/max. In preserve_hu
					// mode fpmin is the HU offset base and must stay the scanned slide min —
					// taking fpopts.min_intensity() (0 by default) would clamp every negative
					// HU to 0. hu_offset() ignores fpmax entirely.
					if (! fpopts.empty() && ! fpopts.preserve_hu())
					{
						fpmin = fpopts.min_intensity();
						fpmax = fpopts.max_intensity();
					}

					if (Nyxus::check_tile_status(int_fpath))
					{
						intFL = new NyxusGrayscaleTiffTileLoader<uint32_t> (
							n_threads,
							int_fpath,
							true,
							fpmin,
							fpmax,
							fpopts.target_dyn_range(),
							fpopts.preserve_hu());		// CT/HU mode: offset-preserving map
					} 
					else 
					{
						intFL = new NyxusGrayscaleTiffStripLoader<uint32_t>(n_threads, int_fpath);
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
		// FIX: unify seg dispatch with detect_input_format(). Defect fixed: the seg path only
		// matched ".zarr", so a ".ome.zarr" mask mis-routed to the TIFF path (intensity path matched both).
		Nyxus::InputFormat fmt = Nyxus::detect_input_format (seg_fpath);

		if (fmt.kind == Nyxus::ContainerKind::OmeZarr)		// FIX: was `ext==".zarr"` only (dropped .ome.zarr)
		{
			#ifdef OMEZARR_SUPPORT
				segFL = new NyxusOmeZarrLoader<uint32_t>(n_threads, seg_fpath);
			#else
				std::cout << "This version of Nyxus was not build with OmeZarr support." <<std::endl;
			#endif
		}
		else
			if (fmt.kind == Nyxus::ContainerKind::Dicom)		// FIX: was `ext==".dcm"||".dicom"`
			{
				#ifdef DICOM_SUPPORT
					segFL = new NyxusGrayscaleDicomLoader<uint32_t>(n_threads, seg_fpath);
				#else
					std::cout << "This version of Nyxus was not build with DICOM support." <<std::endl; 
				#endif
			}
			else
				if (fmt.kind == Nyxus::ContainerKind::Nifti)		// FIX: was `ext==".nii"||".nii.gz"`
				{
					segFL = new NiftiLoader <uint32_t> (seg_fpath);
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

void ImageLoader::assemble_volume (AbstractTileLoader<uint32_t>* fl,
	std::shared_ptr<std::vector<uint32_t>>& ptr,
	std::vector<uint32_t>& dst, size_t channel, size_t timeframe)
{
	const size_t sliceSize = (size_t)fw * fh;

	// Use THIS loader's own layout: per-plane loaders (OME-Zarr, multi-page TIFF)
	// deliver one Z-plane per read (tileDepth==1, tileTimestamps==1) with the
	// timeframe chosen by the loadTileFromFile arg; a whole-4D loader (NIfTI)
	// delivers the entire x*y*z*t blob ([t][z][y][x]) in one read and ignores the
	// timeframe arg, so the requested frame is slabbed out via frameBase.
	const size_t ltd = fl->tileDepth (lvl);
	const size_t ltt = fl->tileTimestamps (lvl);
	const size_t lntd = fl->numberTileDepth (lvl);
	const size_t frameStride = ltd * th * tw;
	const size_t frameBase = (ltt > 1) ? timeframe * frameStride : 0;

	// FIX: walk the whole tile GRID of each plane, not just tile (0,0). A plane commonly
	// spans several tiles/chunks (OME-Zarr chunks are typically 512x512, tiled OME-TIFF
	// 256x256), and the old code read tile (0,0) once and then copied fh*fw out of that
	// single tile's buffer -- wrong data past the first tile plus an out-of-bounds read.
	// Edge tiles are partial, so each tile contributes only its valid [validH x validW].
	const size_t lnth = fl->numberTileHeight (lvl);
	const size_t lntw = fl->numberTileWidth (lvl);

	for (size_t lz = 0; lz < lntd; ++lz)
	{
		for (size_t tr = 0; tr < lnth; ++tr)
		for (size_t tc = 0; tc < lntw; ++tc)
		{
			fl->loadTileFromFile (ptr, tr, tc, lz, channel, timeframe, lvl);

			const size_t row0 = tr * th, col0 = tc * tw;
			if (row0 >= fh || col0 >= fw)
				continue;
			const size_t validH = (std::min) (th, fh - row0),
				validW = (std::min) (tw, fw - col0);

			for (size_t pz = 0; pz < ltd && (lz * ltd + pz) < fd; ++pz)
			{
				const size_t gz = lz * ltd + pz;
				for (size_t row = 0; row < validH; ++row)
				{
					const size_t src = frameBase + (pz * th + row) * tw;
					const size_t d = gz * sliceSize + (row0 + row) * fw + col0;
					std::copy (ptr->begin() + src, ptr->begin() + src + validW, dst.begin() + d);
				}
			}
		}
	}
}

bool ImageLoader::load_volume (size_t channel, size_t timeframe, size_t mask_timeframe)
{
	cur_channel = channel;
	cur_timeframe = timeframe;

	const size_t volSize = (size_t)fw * fh * fd;   // the whole X*Y*Z volume

	if (vol_int_.size() != volSize)
		vol_int_.resize (volSize);
	assemble_volume (intFL, ptrI, vol_int_, channel, timeframe);

	if (segFL != nullptr)
	{
		if (vol_seg_.size() != volSize)
			vol_seg_.resize (volSize);
		// FIX (IO): the mask is usually channel-agnostic (a single-channel segmentation that
		// applies to every intensity channel), so index into the mask's OWN channels only when
		// it actually has that many; otherwise fall back to channel 0. Without this, featurizing
		// intensity channel c>0 against a 1-channel mask read the mask out of range and dropped
		// the ROI. Mirrors the mask_timeframe separation for the 1-mask : N-intensity case.
		size_t mask_channel = (channel < segFL->numberChannels()) ? channel : 0;
		// FIX: clamp the mask TIMEFRAME too (callers already pass 0 for a single-timeframe mask,
		// but make load_volume self-defending like it is for the channel -- an unclamped t>0
		// would read the TIFF mask past its last IFD and throw uncaught). Symmetric with above.
		size_t mask_tf = (mask_timeframe < segFL->fullTimestamps (lvl)) ? mask_timeframe : 0;
		assemble_volume (segFL, ptrL, vol_seg_, mask_channel, mask_tf);
	}
	return true;
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
