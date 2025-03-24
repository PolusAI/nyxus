#include <string>
#include <vector>
#include <map>
#include <array>
#ifdef WITH_PYTHON_H
#include <pybind11/pybind11.h>
#endif
#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

namespace Nyxus
{
	/// @brief Feeds a pixel to image measurement object to gauge the image RAM footprint without caching the pixel. Updates 'uniqueLabels' and 'roiData'.
	/// @param x -- x-coordinate of the pixel in the image
	/// @param y -- y-coordinate of the pixel in the image
	/// @param label -- label of pixel's segment 
	/// @param intensity -- pixel's intensity
	/// @param tile_index -- index of pixel's tile in the image
	void feed_pixel_2_metrics(int x, int y, PixIntens intensity, int label, unsigned int tile_index)
	{
		if (uniqueLabels.find(label) == uniqueLabels.end())
		{
			// Remember this label
			uniqueLabels.insert(label);

			// Initialize the ROI label record
			LR newData (label);
			init_label_record_2(newData, theSegFname, theIntFname, x, y, label, intensity, tile_index);
			roiData[label] = newData;
		}
		else
		{
			// Update basic ROI info (info that doesn't require costly calculations)
			LR& existingData = roiData[label];
			update_label_record_2(existingData, x, y, label, intensity, tile_index);
		}
	}

	void feed_pixel_2_metrics_3D (int x, int y, int z, PixIntens intensity, int label, unsigned int tile_index)
	{
		if (uniqueLabels.find(label) == uniqueLabels.end())
		{
			// Remember this label
			uniqueLabels.insert(label);

			// Initialize the ROI label record
			LR newData (label);
			init_label_record_3D (newData, theSegFname, theIntFname, x, y, z, label, intensity, tile_index);
			roiData[label] = newData;
		}
		else
		{
			// Update basic ROI info (info that doesn't require costly calculations)
			LR& existingData = roiData[label];
			update_label_record_3D (existingData, x, y, z, label, intensity, tile_index);
		}
	}

	void feed_pixel_2_cache_LR (int x, int y, PixIntens intensity, LR& r)
	{
		r.raw_pixels.push_back (Pixel2(x, y, intensity));
	}	
	
	/// @brief Copies a pixel to the ROI's cache. 
	/// @param x -- x-coordinate of the pixel in the image
	/// @param y -- y-coordinate of the pixel in the image
	/// @param label -- label of pixel's segment 
	/// @param intensity -- pixel's intensity
	void feed_pixel_2_cache (int x, int y, PixIntens intensity, int label)
	{
		// Update basic ROI info (info that doesn't require costly calculations)
		LR& r = roiData[label];
		feed_pixel_2_cache_LR(x, y, intensity, r);
	}

	void feed_pixel_2_cache_3D (int x, int y, int z, PixIntens intensity, int label)
	{
		// Update basic ROI info (info that doesn't require costly calculations)
		LR& r = roiData[label];
		r.raw_pixels_3D.push_back (Pixel3(x, y, z, intensity));
	}
}
