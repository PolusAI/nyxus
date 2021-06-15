#include <string>
#include <specialised_tile_loader/grayscale_tiff_tile_loader.h>


bool scanFastloaderWay (std::string & fpath, int num_threads)
{
	GrayscaleTiffTileLoader<uint32_t> fl (num_threads, fpath);
	
	// Allocate the tile buffer
	auto th = fl.tileHeight(0),
		tw = fl.tileWidth(0),
		td = fl.tileDepth(0);
	auto tileSize = th*tw;

	std::shared_ptr<std::vector<uint32_t>> data = std::make_shared<std::vector<uint32_t>> (tileSize);

	// Just for information
	auto fh = fl.fullHeight(0);
	auto fw = fl.fullWidth(0);
	auto fd = fl.fullDepth(0);

	// Read the TIFF tile by tile
	auto ntw = fl.numberTileWidth(0);
	auto nth = fl.numberTileHeight(0);
	auto ntd = fl.numberTileDepth(0);

	// -- checksum
	unsigned long chkSum = 0;

	for (int row=0; row<nth; row++)
		for (int col = 0; col < ntw; col++)
		{
			std::cout << "\treading tile " << row*ntw+col+1 << " of " << nth*ntw << std::endl;
			fl.loadTileFromFile (data, row, col, 0 /*layer*/, 0 /*level*/);

			// Update the checksum
			for (unsigned long i = 0; i < tileSize; i++)
			{
				auto &v = *data;
				chkSum += v[i];			
			}
		}

	std::cout << "\tfile checksum = " << chkSum << std::endl;

	return true;
}