#include "image_loader.h"

ImageLoader::ImageLoader() {}

bool ImageLoader::open(const std::string& int_fpath, const std::string& seg_fpath)
{
	int n_threads = 1;

	try 
	{
		if (checkTileStatus(int_fpath))
		{
			intFL = new NyxusGrayscaleTiffTileLoader<uint32_t>(n_threads, int_fpath);
		} 
		else 
		{
			// since the file is not tiled, we provide the tile dimensions
			auto [tw, th, td] = calculate_tile_dimensions (int_fpath);
			intFL = new NyxusGrayscaleTiffStripLoader<uint32_t>(n_threads, int_fpath, tw, th, td);
		}
	}
	catch (std::exception const& e)	
	{
		std::cout << "Error while initializing the image loader for intensity image file " << int_fpath << ": " << e.what() << "\n";
		return false;
	}

	if (intFL == nullptr)
		return false;

	try {
		if (checkTileStatus(seg_fpath))
		{
			segFL = new NyxusGrayscaleTiffTileLoader<uint32_t>(n_threads, seg_fpath);
		} 
		else 
		{
			// since the file is not tiled, we provide the tile dimensions
			auto [tw, th, td] = calculate_tile_dimensions(seg_fpath); 
			segFL = new NyxusGrayscaleTiffStripLoader<uint32_t>(n_threads, seg_fpath, tw, th, td);
		}

	}
	catch (std::exception const& e)	
	{
		std::cout << "Error while initializing the image loader for mask image file " <<  seg_fpath << ": " << e.what() << "\n";
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

#if 0 // Tests
	ptrI = std::make_shared<std::vector<uint32_t>>(tileSize);
	// Experiment
	ptrL = std::make_shared<std::vector<uint32_t>>(tileSize);
	segFL->loadTileFromFile(ptrL, 
		0, //row, 
		0, //col, 
		0, //lyr, 
		0); // lvl);
	auto& dataL = *ptrL;
#endif

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

bool ImageLoader::checkTileStatus(const std::string& filePath)
{
	TIFF *tiff_ = TIFFOpen(filePath.c_str(), "r");
	if (tiff_ != nullptr) 
	{
		if (TIFFIsTiled(tiff_) == 0) 
		{ 
			TIFFClose(tiff_);
			return false;
			} else 
			{
			TIFFClose(tiff_);
			return true;
			}
	} else { throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); }
}

std::tuple<uint32_t, uint32_t, uint32_t>  ImageLoader::get_image_dimensions (const std::string& filePath)
{
	TIFF* tiff_ = TIFFOpen(filePath.c_str(), "r");
	if (tiff_ != nullptr)
	{
		uint32_t w, l, ndirs;
		TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &w);
		TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &l);
		ndirs = TIFFNumberOfDirectories(tiff_);
		TIFFClose(tiff_);
		return {w, l, ndirs};
	}
	else 
	{ 
		throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); 
	}
}

std::tuple<uint32_t, uint32_t, uint32_t>  ImageLoader::calculate_tile_dimensions (const std::string& filePath)
{
	auto [w, h, d] = get_image_dimensions (filePath);
	uint32_t defaultWidthSize = 1024;
	uint32_t defaultHeightSize = 1024;
	uint32_t defaultDepthSize = 1;
	w = std::min ({ w, defaultWidthSize });
	h = std::min ({ h, defaultHeightSize });
	d = std::min ({ d, defaultDepthSize });
	return {w, h, d};
}