#include <iostream>
#include "environment.h"
#include "image_loader.h"
#include "grayscale_tiff.h"
#include "raw_tiff.h"
#include "omezarr.h"
#include "nyxus_dicom_loader.h"
#include "dirs_and_files.h"
#include "helpers/fsystem.h"
#include "raw_nifti.h"


ImageLoader::ImageLoader() {}

bool ImageLoader::open (SlideProps & p)	
{
	int n_threads = 1;

	std::string & int_fpath = p.fname_int,
		& seg_fpath = p.fname_seg;

	// intensity image
	try 
	{
		std::string ext = Nyxus::get_big_extension (int_fpath);

		if (ext == ".zarr")
		{
			#ifdef OMEZARR_SUPPORT
				intFL = new NyxusOmeZarrLoader<uint32_t>(n_threads, int_fpath);
			#else
				std::cout << "This version of Nyxus was not build with OmeZarr support." <<std::endl; 
			#endif
		}
		else 
			if (ext == ".dcm" || ext == ".dicom")
			{
				#ifdef DICOM_SUPPORT
					intFL = new NyxusGrayscaleDicomLoader<uint32_t>(n_threads, int_fpath);
				#else
					std::cout << "This version of Nyxus was not build with DICOM support." <<std::endl; 
				#endif
			}
			else
				if (ext == ".nii" || ext == ".nii.gz")
				{
					intFL = new NiftiLoader<uint32_t> (int_fpath);
				}
				else 
				{
					// flavors of TIFF (TIFF, OME.TIFF)
					
					// automatic or overriden FP dynamic range
					double fpmin = p.min_preroi_inten,
						fpmax = p.max_preroi_inten;
					if (!Nyxus::theEnvironment.fpimageOptions.empty())
					{
						fpmin = Nyxus::theEnvironment.fpimageOptions.min_intensity();
						fpmax = Nyxus::theEnvironment.fpimageOptions.max_intensity();
					}

					if (Nyxus::check_tile_status(int_fpath))
					{
						intFL = new NyxusGrayscaleTiffTileLoader<uint32_t> (
							n_threads, 
							int_fpath, 
							true,
							fpmin,
							fpmax,
							Nyxus::theEnvironment.fpimageOptions.target_dyn_range());
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
	th = intFL->tileHeight(lvl);
	tw = intFL->tileWidth(lvl);
	td = intFL->tileDepth(lvl);
	tileSize = th * tw * td;

	fh = intFL->fullHeight(lvl);
	fw = intFL->fullWidth(lvl);
	fd = intFL->fullDepth(lvl);

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
		std::string ext = Nyxus::get_big_extension(seg_fpath);

		if (ext == ".zarr")
		{
			#ifdef OMEZARR_SUPPORT
				segFL = new NyxusOmeZarrLoader<uint32_t>(n_threads, seg_fpath);
			#else
				std::cout << "This version of Nyxus was not build with OmeZarr support." <<std::endl;
			#endif
		}
		else 
			if (ext == ".dcm" || ext == ".dicom")
			{
				#ifdef DICOM_SUPPORT
					segFL = new NyxusGrayscaleDicomLoader<uint32_t>(n_threads, seg_fpath);
				#else
					std::cout << "This version of Nyxus was not build with DICOM support." <<std::endl; 
				#endif
			}
			else
				if (ext == ".nii" || ext == ".nii.gz")
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
							Nyxus::theEnvironment.fpimageOptions.target_dyn_range());
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
	
	intFL->loadTileFromFile (ptrI, tRow, tCol, lyr, lvl);

	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (ptrL, tRow, tCol, lyr, lvl);
	
	return true;
}

bool ImageLoader::load_tile (size_t tile_row, size_t tile_col)
{
	if (tile_row >= nth || tile_col >= ntw)
		return false;

	intFL->loadTileFromFile (ptrI, tile_row, tile_col, lyr, lvl);

	// segmentation loader is not available in wholeslide
	if (segFL)
		segFL->loadTileFromFile (ptrL, tile_row, tile_col, lyr, lvl);

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
	return ntw;
}

size_t ImageLoader::get_num_tiles_hor()
{
	return nth;
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
