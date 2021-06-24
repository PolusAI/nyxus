#include <string>
#include <fast_loader/specialised_tile_loader/grayscale_tiff_tile_loader.h>
#include <map>
#include "sensemaker.h"

#include <array>
#include "virtual_file_tile_channel_loader.h"

// Sanity
#include<windows.h>
#include <chrono>

int ingestDataset1 (std::vector<std::string> &intensFiles, std::vector<std::string> &labelFiles, int numFastloaderThreads)
{
	bool ok = true;

	for (auto & fp : intensFiles)
	{
		ok = scanViaFastloader (fp, numFastloaderThreads);
		//ok = TraverseViaFastloader1 (fp, numFastloaderThreads);
		if (ok == false)
			return 1;
	}

	for (auto & fp : labelFiles)
	{
		ok = scanViaFastloader (fp, numFastloaderThreads);
		//ok = TraverseViaFastloader1 (fp, numFastloaderThreads);
		if (ok == false)
			return 1;
	}

	return 0; // success
}

bool scanFilePair (const std::string& intens_fpath, const std::string& label_fpath, int num_threads)
{
	std::cout << std::endl << "Processing pair " << intens_fpath << " -- " << label_fpath << std::endl;

	int lvl = 0;

	// File #1 (intensity)
	GrayscaleTiffTileLoader<uint32_t> I (num_threads, intens_fpath);

	auto th = I.tileHeight(lvl),
		tw = I.tileWidth(lvl),
		td = I.tileDepth(lvl);
	auto tileSize = th * tw;

	auto fh = I.fullHeight(lvl);
	auto fw = I.fullWidth(lvl);
	auto fd = I.fullDepth(lvl);

	auto ntw = I.numberTileWidth(lvl);
	auto nth = I.numberTileHeight(lvl);
	auto ntd = I.numberTileDepth(lvl);

	// File #2 (labels)
	GrayscaleTiffTileLoader<uint32_t> L(num_threads, label_fpath);

	// -- check whole file consistency
	if (fh != L.fullHeight(lvl) || fw != L.fullWidth(lvl) || fd != L.fullDepth(lvl))
	{
		std::cout << "\terror: mismatch in full height, width, or depth";
		return false;
	}

	// -- check tile consistency
	if (th != L.tileHeight(lvl) || tw != L.tileWidth(lvl) || td != L.tileDepth(lvl))
	{
		std::cout << "\terror: mismatch in tile height, width, or depth";
		return false;
	}

	// Read the TIFF tile by tile 
	// 
	// -- allocate the tile buffer
	std::shared_ptr<std::vector<uint32_t>> ptrI = std::make_shared<std::vector<uint32_t>> (tileSize);
	std::shared_ptr<std::vector<uint32_t>> ptrL = std::make_shared<std::vector<uint32_t>> (tileSize);

	for (int row = 0; row < nth; row++)
		for (int col = 0; col < ntw; col++)
		{
			std::cout << "\tt." << row * ntw + col + 1 << "/" << nth * ntw << std::endl;

			I.loadTileFromFile (ptrI, row, col, 0 /*layer*/, lvl);
			L.loadTileFromFile (ptrL, row, col, 0 /*layer*/, lvl);
			auto& dataI = *ptrI;
			auto& dataL = *ptrL;

			// Calculate statistics
			for (unsigned long i = 0; i < tileSize; i++)
			{
				auto label = dataL[i];
				if (label != 0)
				{
					update_label_stats (label, dataI[i]);
				}
			}
		}

	return true;
}

int ingestDataset (std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles, int numFastloaderThreads)
{
	// Sanity
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	bool ok = true;

	int nf = intensFiles.size();
	for (int i = 0; i < nf; i++)
	{
		// Clear label stats buffers
		clearLabelStats();

		auto &ifp = intensFiles[i], 
			&lfp = labelFiles[i];

		// Scan a label-intensity pair and calculate features
		ok = scanFilePair (ifp, lfp, numFastloaderThreads);
		if (ok == false)
			return 1;

		// Execute calculations requiring reduction
		do_partial_stats_reduction();

		// Sanity check
		print_label_stats();
	}

	// Sanity
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Elapsed time (s) " << elapsed_seconds.count() << std::endl;

	return 0; // success
}


bool scanViaFastloader (const std::string & fpath, int num_threads)
{
	// Sanity check
	std::vector <uint32_t> histo; // histogram channels
	std::vector <uint32_t> counts; // channel counts
	//
	std::cout << std::endl << "Processing file " << fpath << std::endl;

	// Get the TIFF file info
	GrayscaleTiffTileLoader<uint32_t> fl (num_threads, fpath);
	
	auto th = fl.tileHeight(0),
		tw = fl.tileWidth(0),
		td = fl.tileDepth(0);
	auto tileSize = th*tw;

	// Just for information
	auto fh = fl.fullHeight(0);
	auto fw = fl.fullWidth(0);
	auto fd = fl.fullDepth(0);

	// Read the TIFF tile by tile
	auto ntw = fl.numberTileWidth(0);
	auto nth = fl.numberTileHeight(0);
	auto ntd = fl.numberTileDepth(0);

	// Allocate the tile buffer
	std::shared_ptr<std::vector<uint32_t>> data = std::make_shared<std::vector<uint32_t>>(tileSize);

	for (int row=0; row<nth; row++)
		for (int col = 0; col < ntw; col++)
		{
			std::cout << "\tt." << row*ntw+col+1 << " of " << nth*ntw << std::endl;
			fl.loadTileFromFile (data, row, col, 0 /*layer*/, 0 /*level*/);

			// Calculate statistics
			for (unsigned long i = 0; i < tileSize; i++)
			{
				// Sanity
				auto &v = *data;
				auto iter = std::find (histo.begin(), histo.end(), v[i]);
				if (iter == histo.end())
				{
					// The pixel value is not known, add it
					histo.push_back(v[i]);	// pixel value is not in the histogram so save it 
					counts.push_back(1);
				}
				else
				{
					// The pixel value is previously known, increment its counter
					int index = iter - histo.begin();
					counts[index] = counts[index] + 1;
				}

			}
		}

	// Sanity
	std::cout << "\t"<< histo.size() << " histogram entries" << std::endl;
	// --Show a histogram fragment
	for (int i = 0; i < 100 && i < histo.size(); i++)
	{
		std::cout << i << "\t" << histo[i] << "\t" << counts[i] << std::endl;
	}

	return true;
}


bool TraverseViaFastloader1 (const std::string & fpath, int num_threads)
{
	// Sanity
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof (statex);
	GlobalMemoryStatusEx (&statex);
	std::cout << "memory load, %=" << statex.dwMemoryLoad
		<< "\ntotal physical memory, Mb=" << statex.ullTotalPhys / 1048576
		<< "\ntotal available meory, Mb=" << statex.ullAvailPhys / 1048576
		<< std::endl;

	// Get the TIFF file info
	GrayscaleTiffTileLoader<uint32_t> gfl(num_threads, fpath);

	auto th = gfl.tileHeight(0),
		tw = gfl.tileWidth(0),
		td = gfl.tileDepth(0);
	auto tileSize = th * tw;

	// Just for information
	auto fh = gfl.fullHeight(0);
	auto fw = gfl.fullWidth(0);
	auto fd = gfl.fullDepth(0);

	// Read the TIFF tile by tile
	auto ntw = gfl.numberTileWidth(0);
	auto nth = gfl.numberTileHeight(0);
	auto ntd = gfl.numberTileDepth(0);

	// Radius
	const int radiusDepth = 1, radiusHeight = 1, radiusWidth = 1;

	// Instanciate a Tile loader
	auto tl = std::make_shared<VirtualFileTileChannelLoader>(
		num_threads,
		fh, fw, fd,
		th, tw, td,
		1 /*numberChannels*/);

	// Create the Fast Loader configuration
	auto options = std::make_unique<fl::FastLoaderConfiguration<fl::DefaultView<int>>>(tl);
	
	// Set the configuration
	options->radius(radiusDepth, radiusHeight, radiusWidth);
	options->ordered(true);
	options->borderCreatorConstant(0);
	// Create the Fast Loader Graph
	auto fl = fl::FastLoaderGraph<fl::DefaultView<int >>(std::move(options));
	// Execute the graph
	fl.executeGraph();
	// Request all the views in the graph
	fl.requestAllViews();
	// Indicate no other view will be requested
	fl.finishRequestingViews();

	// For each of the view
	while (auto view = fl.getBlockingResult()) {
		// Do stuff

		// Return the view to the Fast Loader Graph
		view->returnToMemoryManager();
	}
	// Wait for the graph to terminate
	fl.waitForTermination();

	return true;
}


bool TraverseViaFastloader2 (const std::string & fpath, int num_threads)
{
	// Sanity
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof (statex);
	GlobalMemoryStatusEx (&statex);
	std::cout << "memory load, %=" << statex.dwMemoryLoad
		<< "\ntotal physical memory, Mb=" << statex.ullTotalPhys / 1048576
		<< "\ntotal available meory, Mb=" << statex.ullAvailPhys / 1048576
		<< std::endl;

	// Get the TIFF file info
	GrayscaleTiffTileLoader<int> gfl(num_threads, fpath);

	auto th = gfl.tileHeight(0),
		tw = gfl.tileWidth(0),
		td = gfl.tileDepth(0);
	auto tileSize = th * tw;

	// Just for information
	auto fh = gfl.fullHeight(0);
	auto fw = gfl.fullWidth(0);
	auto fd = gfl.fullDepth(0);

	// Read the TIFF tile by tile
	auto ntw = gfl.numberTileWidth(0);
	auto nth = gfl.numberTileHeight(0);
	auto ntd = gfl.numberTileDepth(0);

	uint16_t fileHeight = fh, //10,
		fileWidth = fw, //10,
		fileDepth = 1, //10,
		tileHeight = th, //2,
		tileWidth = tw, //2,
		tileDepth = 1, //2,
		numberChannels = 1;

	uint32_t radiusDepth = 1,
		radiusHeight = 1,
		radiusWidth = 1;

	uint32_t numberThreads = 1; // 5;

	// Instanciate a Tile loader
	auto tl = std::make_shared<VirtualFileTileChannelLoader>(
		numberThreads,
		fileHeight, fileWidth, fileDepth,
		tileHeight, tileWidth, tileDepth,
		numberChannels);

	// Create the Fast Loader configuration
	auto options = std::make_unique<fl::FastLoaderConfiguration<fl::DefaultView<int>>>(tl);
	// Set the configuration
	options->radius(radiusDepth, radiusHeight, radiusWidth);
	options->ordered(true);
	options->borderCreatorConstant(0);

	// Create the Fast Loader Graph
	auto fl = fl::FastLoaderGraph<fl::DefaultView<int>>(std::move(options));
	// Execute the graph
	fl.executeGraph();
	// Request all the views in the graph
	fl.requestAllViews();
	// Indicate no other view will be requested
	fl.finishRequestingViews();

	// For each of the view
	while (auto view = fl.getBlockingResult()) {
		// Do stuff
		//std::cout << *view << std::endl;

		// Do stuff #2
		// Get information on the tile and looping through it 
		for (int32_t y = 0; y < view->tileHeight(); y++) 
		{
			for (int32_t x = 0; x < view->tileWidth(); x++) 
			{
				// Get a pixel within the tile
				//---	uint32_t pixel = view->getPixelValue (y,x);
				//---	std::cout << pixel << std::endl;
			}
		}

		// Return the view to the Fast Loader Graph
		view->returnToMemoryManager();
	}
	// Wait for the graph to terminate
	fl.waitForTermination();

	return true;
}


/* --- This code is pending a descussion with FastLoader people ---
bool TraverseViaFastloader3 (const std::string& fpath, int num_threads)
{
	// Sanity
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof(statex);
	GlobalMemoryStatusEx(&statex);
	std::cout << "memory load, %=" << statex.dwMemoryLoad
		<< "\ntotal physical memory, Mb=" << statex.ullTotalPhys / 1048576
		<< "\ntotal available meory, Mb=" << statex.ullAvailPhys / 1048576
		<< std::endl;

	// Get the TIFF file info
	GrayscaleTiffTileLoader<uint32_t> gfl(num_threads, fpath);

	auto th = gfl.tileHeight(0),
		tw = gfl.tileWidth(0),
		td = gfl.tileDepth(0);
	auto tileSize = th * tw;

	// Just for information
	auto fh = gfl.fullHeight(0);
	auto fw = gfl.fullWidth(0);
	auto fd = gfl.fullDepth(0);

	// Read the TIFF tile by tile
	auto ntw = gfl.numberTileWidth(0);
	auto nth = gfl.numberTileHeight(0);
	auto ntd = gfl.numberTileDepth(0);

	uint16_t fileHeight = fh, //10,
		fileWidth = fw, //10,
		fileDepth = 1, //10,
		tileHeight = th, //2,
		tileWidth = tw, //2,
		tileDepth = 1, //2,
		numberChannels = 1;

	uint32_t radiusDepth = 1,
		radiusHeight = 1,
		radiusWidth = 1;

	uint32_t numberThreads = 1; // 5;

	// Instanciate a Tile loader
	auto tl = std::make_shared<VirtualFileTileChannelLoader>(
		numberThreads,
		fileHeight, fileWidth, fileDepth,
		tileHeight, tileWidth, tileDepth,
		numberChannels);

	// Create the Fast Loader configuration
	auto options = std::make_unique<fl::FastLoaderConfiguration<fl::DefaultView<uint32_t>>>(tl);
	// Set the configuration
	options->radius(radiusDepth, radiusHeight, radiusWidth);
	options->ordered(true);
	options->borderCreatorConstant(0);

	// Create the Fast Loader Graph
	auto fl = fl::FastLoaderGraph<fl::DefaultView<uint32_t>>(std::move(options));
	// Execute the graph
	fl.executeGraph();
	// Request all the views in the graph
	fl.requestAllViews();
	// Indicate no other view will be requested
	fl.finishRequestingViews();

	// Show pixel values on the whole image diagonal
	if (fileHeight == fileWidth) // Make sure we deal with a square image
		for (uint32_t i = 0; i < fileHeight; i++) // Iterate diagonal elements
		{
			// Global pixel position
			uint32_t globalRow = i, 
				globalCol = i;

			// Get the view containing the pixel with global coordinates. The corresponding tile data is taken from cache or loaded via file i/o
			auto view = fl.getContainingView (globalRow, globalCol);

			// Learn the corresponding position local within the view/tile
			auto [localRow, localCol] = view->castGlobalPosition (globalRow, globalCol);
			
			// Finally, read the pixel value
			uint32_t pixel = view->getPixelValue (localRow, localCol);
		}

		// Return the view to the Fast Loader Graph
		view->returnToMemoryManager();
	}
	// Wait for the graph to terminate
	fl.waitForTermination();

	return true;
}
*/
