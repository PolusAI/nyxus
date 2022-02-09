#include "image_loader.h"

ImageLoader::ImageLoader()
{
}

bool ImageLoader::open(const std::string& int_fpath, const std::string& seg_fpath)
{
	int n_threads = 1;

	try {
		intFL = new GrayscaleTiffTileLoader<uint32_t>(n_threads, int_fpath);
	}
	catch (std::exception const& e)	
	{
		std::stringstream ss;
		ss << "Error while initializing the image loader for intensity image file " << int_fpath << ": " << e.what();
		#ifdef WITH_PYTHON_H
			throw ss.str().c_str();
		#endif
		std::cerr << ss.str() << std::endl;

		return false;
	}

	if (intFL == nullptr)
		return false;

	try {
		segFL = new GrayscaleTiffTileLoader<uint32_t>(n_threads, seg_fpath);
	}
	catch (std::exception const& e)	
	{
		std::stringstream ss;
		ss << "Error while initializing the image loader for mask image file " << seg_fpath << ": " << e.what();
		#ifdef WITH_PYTHON_H
			throw ss.str().c_str();
		#endif
		std::cerr << ss.str() << std::endl;

		return false;
	}

	if (segFL == nullptr)
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

	// File #2 (labels)

	// -- check whole file consistency
	if (fh != segFL->fullHeight(lvl) || fw != segFL->fullWidth(lvl) || fd != segFL->fullDepth(lvl))
	{
		#ifdef WITH_PYTHON_H
			throw "Error: mismatch in full height, width, or depth";
		#endif
		std::cerr << "Error: mismatch in full height, width, or depth\n";
		return false;
	}

	// -- check tile consistency
	if (th != segFL->tileHeight(lvl) || tw != segFL->tileWidth(lvl) || td != segFL->tileDepth(lvl))
	{
		#ifdef WITH_PYTHON_H
			throw "Error: mismatch in tile height, width, or depth";
		#endif
		std::cerr << "Error: mismatch in tile height, width, or depth\n";
		return false;
	}

	ptrI = std::make_shared<std::vector<uint32_t>>(tileSize);
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

	auto row = tile_idx / ntw;
	auto col = tile_idx % ntw;
	intFL->loadTileFromFile (ptrI, row, col, lyr, lvl);
	segFL->loadTileFromFile (ptrL, row, col, lyr, lvl);
	
	return true;
}

bool ImageLoader::load_tile (size_t tile_row, size_t tile_col)
{
	if (tile_row >= nth || tile_col >= ntw)
		return false;

	intFL->loadTileFromFile (ptrI, tile_row, tile_col, lyr, lvl);
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

