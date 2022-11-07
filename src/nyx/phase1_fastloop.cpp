#include <cstdint>
#include <popcntintrin.h>
#include <ratio>
#include <smmintrin.h>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <array>
#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#endif
#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"
#include "rle.hpp"

namespace Nyxus
{
	bool gatherRoisMetricsFast (const std::string& intens_fpath, const std::string& label_fpath, int num_FL_threads)
	{

		int lvl = 0, // Pyramid level
			lyr = 0; //	Layer

		// Read the tiff. The image loader is put in the open state in processDataset()
		size_t nth = theImLoader.get_num_tiles_hor(),
			ntv = theImLoader.get_num_tiles_vert(),
			fw = theImLoader.get_tile_width(),
			th = theImLoader.get_tile_height(),
			tw = theImLoader.get_tile_width(),
			tileSize = theImLoader.get_tile_size(),
			fullwidth = theImLoader.get_full_width(),
			fullheight = theImLoader.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = theImLoader.load_tile(row, col);
				if (!ok)
				{
					std::stringstream ss;
					ss << "Error fetching tile row=" << row << " col=" << col;
					#ifdef WITH_PYTHON_H
						throw ss.str();
					#endif	
					std::cerr << ss.str() << "\n";
					return false;
				}

				// Get ahold of tile's pixel buffer
				auto tileIdx = row * nth + col;
				auto dataI = theImLoader.get_int_tile_buffer(),
					dataL = theImLoader.get_seg_tile_buffer();

				// Iterate pixels
                int y = 0;
				int y_max = (fullheight < y+th) ? fullheight - row*th : th;
				while (y < y_max)
				{

                    size_t i = y*tw;
                    
                    int x = 0;
					int x_max = (fullwidth < x+tw) ? fullwidth - col*tw : tw;


					int x_stream = 32 * (x_max / 32);

					// Compress row to an RLE stream
					RLEStream<uint32_t, uint16_t> stream = rle_encode_long_stream_32(dataL.data()+i,x_stream);

					// Loop over row objects in RLE stream
					unsigned int ind = 0;
					while (ind+1 < stream.offsets.size()) {
						uint16_t x1 = stream.offsets[ind++];
						auto label = dataL[i+x1];

						// If not background, store the object
						if (label > 0) {
							// Collapse all the labels to one if single-ROI mde is requested
							if (theEnvironment.singleROI)
								label = 1;

							uint16_t x2 = stream.offsets[ind];
							auto minInt = dataI[i+x1];
							auto maxInt = dataI[i+x1];

							// Find the min and max intensity
							for (int xi=i+x1; xi<i+x2; xi++) {
								if (maxInt < dataI[xi]) {
									maxInt = dataI[xi];
								} else if (minInt > dataI[xi]) {
									minInt = dataI[xi];
								}

							}

							feed_pixel_2_metrics_fast (x1, x2, y, maxInt, minInt, label, tileIdx); // Updates 'uniqueLabels' and 'roiData'
						}
					}

					i += x_stream;
					x = x_stream;
					while (x < x_max) {

                        // Skip non-mask pixels
                        auto label = dataL[i];
                        if (label == 0) {
                            ++x, ++i;
                            continue;
                        }

                        // Collapse all the labels to one if single-ROI mde is requested
                        if (theEnvironment.singleROI)
                            label = 1;
                        
                        // Update pixel's ROI metrics
                        feed_pixel_2_metrics (x, y, dataI[i], label, tileIdx); // Updates 'uniqueLabels' and 'roiData'
                        ++x, ++i;
                    }
                    ++y;
				}

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

				// Show stayalive progress info
				if (cnt++ % 4 == 0)
					std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
			}

		return true;
	}

}
