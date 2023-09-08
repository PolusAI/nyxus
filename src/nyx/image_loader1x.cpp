#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include <iostream>
#include "image_loader1x.h"
#include "grayscale_tiff.h"
#include "omezarr.h"
#include "nyxus_dicom_loader.h"
#include "dirs_and_files.h"

ImageLoader1x::ImageLoader1x() {}

bool ImageLoader1x::open(const std::string& fpath)
{
	int n_threads = 1;

	try
	{
		if 	(fs::path(fpath).extension() == ".zarr")
		{
			#ifdef OMEZARR_SUPPORT
			FL = std::make_unique<NyxusOmeZarrLoader<uint32_t>>(n_threads, fpath);
			#else
			std::cout << "This version of Nyxus was not build with OmeZarr support." <<std::endl;
			#endif
		}
		else if(fs::path(fpath).extension() == ".dcm" | fs::path(fpath).extension() == ".dicom"){
			#ifdef DICOM_SUPPORT
			FL = std::make_unique<NyxusGrayscaleDicomLoader<uint32_t>>(n_threads, fpath);
			#else
			std::cout << "This version of Nyxus was not build with DICOM support." <<std::endl;
			#endif
		}
		else
		{
			if (Nyxus::check_tile_status(fpath))
				FL = std::make_unique<NyxusGrayscaleTiffTileLoader<uint32_t>> (n_threads, fpath);
			else
			{
				FL = std::make_unique<NyxusGrayscaleTiffStripLoader<uint32_t>> (n_threads, fpath);
			}

		}
	}
	catch (std::exception const& e)
	{
		std::cout << "Error while initializing the image loader for intensity image file " << fpath << ": " << e.what() << "\n";
		return false;
	}

	if (FL == nullptr)
		return false;

	// File #1 (intensity)
	th = FL->tileHeight(lvl);
	tw = FL->tileWidth(lvl);
	td = FL->tileDepth(lvl);
	tileSize = th * tw;

	fh = FL->fullHeight(lvl);
	fw = FL->fullWidth(lvl);
	fd = FL->fullDepth(lvl);

	ntw = FL->numberTileWidth(lvl);
	nth = FL->numberTileHeight(lvl);
	ntd = FL->numberTileDepth(lvl);

	ptr = std::make_shared<std::vector<uint32_t>>(tileSize);

	return true;
}

void ImageLoader1x::close()
{
	if (FL)
	{
		FL = nullptr;
	}
}

bool ImageLoader1x::load_tile(size_t tile_idx)
{
	if (tile_idx >= ntw * nth * ntd)
		return false;

	auto row = tile_idx / ntw;
	auto col = tile_idx % ntw;
	FL->loadTileFromFile(ptr, row, col, lyr, lvl);

	return true;
}

bool ImageLoader1x::load_tile(size_t tile_row, size_t tile_col)
{
	if (tile_row >= nth || tile_col >= ntw)
		return false;

	FL->loadTileFromFile(ptr, tile_row, tile_col, lyr, lvl);

	return true;
}

const std::vector<uint32_t>& ImageLoader1x::get_tile_buffer()
{
	return *ptr;
}

size_t ImageLoader1x::get_tile_size()
{
	return tileSize;
}

size_t ImageLoader1x::get_tile_x(size_t pixel_col)
{
	size_t tx = pixel_col / tw;
	return tx;
}

size_t ImageLoader1x::get_tile_y(size_t pixel_row)
{
	size_t ty = pixel_row / th;
	return ty;
}

size_t ImageLoader1x::get_within_tile_idx(size_t pixel_row, size_t pixel_col)
{
	size_t wtx = pixel_col % tw,
		wty = pixel_row % th,
		idx = wty * tw + wtx;
	return idx;
}

size_t ImageLoader1x::get_num_tiles_vert()
{
	return ntw;
}

size_t ImageLoader1x::get_num_tiles_hor()
{
	return nth;
}

size_t ImageLoader1x::get_tile_height()
{
	return th;
}

size_t ImageLoader1x::get_tile_width()
{
	return tw;
}
