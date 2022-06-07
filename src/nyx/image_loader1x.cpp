#include "image_loader1x.h"

ImageLoader1x::ImageLoader1x() {}

bool ImageLoader1x::open(const std::string& fpath)
{
	int n_threads = 1;

	try
	{
		if (checkTileStatus(fpath))
			FL = std::make_unique<NyxusGrayscaleTiffTileLoader<uint32_t>> (n_threads, fpath); 
		else
		{
			// since the file is not tiled, we provide the tile dimensions
			auto [tw, th, td] = calculate_tile_dimensions(fpath);
			FL = std::make_unique<NyxusGrayscaleTiffStripLoader<uint32_t>> (n_threads, fpath, tw, th, td); 
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

bool ImageLoader1x::checkTileStatus(const std::string& filePath)
{
	TIFF* tiff_ = TIFFOpen(filePath.c_str(), "r");
	if (tiff_ != nullptr)
	{
		if (TIFFIsTiled(tiff_) == 0)
		{
			TIFFClose(tiff_);
			return false;
		}
		else
		{
			TIFFClose(tiff_);
			return true;
		}
	}
	else { throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); }
}

std::tuple<uint32_t, uint32_t, uint32_t>  ImageLoader1x::get_image_dimensions(const std::string& filePath)
{
	TIFF* tiff_ = TIFFOpen(filePath.c_str(), "r");
	if (tiff_ != nullptr)
	{
		uint32_t w, l, ndirs;
		TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &w);
		TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &l);
		ndirs = TIFFNumberOfDirectories(tiff_);
		TIFFClose(tiff_);
		return { w, l, ndirs };
	}
	else
	{
		throw (std::runtime_error("Tile Loader ERROR: The file can not be opened."));
	}
}

std::tuple<uint32_t, uint32_t, uint32_t>  ImageLoader1x::calculate_tile_dimensions(const std::string& filePath)
{
	auto [w, h, d] = get_image_dimensions(filePath);
	uint32_t defaultWidthSize = 1024;
	uint32_t defaultHeightSize = 1024;
	uint32_t defaultDepthSize = 1;
	w = std::min({ w, defaultWidthSize });
	h = std::min({ h, defaultHeightSize });
	d = std::min({ d, defaultDepthSize });
	return { w, h, d };
}

