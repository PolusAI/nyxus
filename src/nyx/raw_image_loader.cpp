#include <iostream>

#define NOMINMAX 

#include "dirs_and_files.h"
#include "helpers/fsystem.h"
#include "raw_image_loader.h"
#include "raw_dicom.h"
#include "raw_nifti.h"
#include "raw_omezarr.h"
#include "raw_tiff.h"

RawImageLoader::RawImageLoader() {}

bool RawImageLoader::open (const std::string& int_fpath, const std::string& seg_fpath)
{
	try
	{
		if (fs::path(int_fpath).extension() == ".zarr")
		{
#ifdef OMEZARR_SUPPORT
			intFL = new RawOmezarrLoader (int_fpath);
#else
			std::cout << "This version of Nyxus was not build with OmeZarr support." << std::endl;
#endif
		}
		else 
			if (fs::path(int_fpath).extension() == ".dcm" | fs::path(int_fpath).extension() == ".dicom") {
#ifdef DICOM_SUPPORT
			intFL = new RawDicomLoader (int_fpath);
#else
			std::cout << "This version of Nyxus was not build with DICOM support." << std::endl;
#endif
		}
			else
				if (fs::path(int_fpath).extension() == ".nii" || fs::path(int_fpath).extension() == ".nii.gz")
				{
					intFL = new RawNiftiLoader (int_fpath);
				}

		else
		{
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
		if (fs::path(seg_fpath).extension() == ".zarr")
		{
#ifdef OMEZARR_SUPPORT
			segFL = new RawOmezarrLoader (seg_fpath);
#else
			std::cout << "This version of Nyxus was not build with OmeZarr support." << std::endl;
#endif
		}
		else if (fs::path(seg_fpath).extension() == ".dcm" | fs::path(seg_fpath).extension() == ".dicom") {
#ifdef DICOM_SUPPORT
			segFL = new RawDicomLoader (seg_fpath);
#else
			std::cout << "This version of Nyxus was not build with DICOM support." << std::endl;
#endif
		}
		else
			if (fs::path(int_fpath).extension() == ".nii" || fs::path(int_fpath).extension() == ".nii.gz")
			{
				intFL = new RawNiftiLoader (int_fpath);
			}
			else
		{
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

	intFL->loadTileFromFile (row, col, lyr, lvl);

	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (row, col, lyr, lvl);

	return true;
}

bool RawImageLoader::load_tile(size_t tile_row, size_t tile_col)
{
	if (tile_row >= nth || tile_col >= ntw)
		return false;

	intFL->loadTileFromFile (tile_row, tile_col, lyr, lvl);
	
	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (tile_row, tile_col, lyr, lvl);
	
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

std::string RawImageLoader::get_slide_descr()
{
	std::string rv = get_fp_phys_pixvoxels() ? "R-" : "N-";
	rv += intFL->get_infostring() + " I ";
	if (segFL != nullptr)
		rv += segFL->get_infostring() + " M ";
	return rv;
}

bool RawImageLoader::get_fp_phys_pixvoxels()
{
	return intFL->get_fp_pixels();
}

