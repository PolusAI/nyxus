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
	void feed_pixel_2_metrics (
		// modified
		std::unordered_set<int> & uniqueLabels,
		std::unordered_map <int, LR> & roiData,
		// in
		int x, int y, PixIntens intensity, int label, int sidx)
	{
		if (uniqueLabels.find(label) == uniqueLabels.end())
		{
			// Remember this label
			uniqueLabels.insert(label);

			// Initialize the ROI label record
			LR newRoi (label);
			newRoi.slide_idx = sidx;
			init_label_record_3 (newRoi, x, y, intensity);
			roiData[label] = newRoi;
		}
		else
		{
			// Update basic ROI info (info that doesn't require costly calculations)
			LR& existingRoi = roiData[label];
			update_label_record_3 (existingRoi, x, y, intensity);
		}
	}

	void feed_pixel_2_metrics_3D (
		// modified
		Uniqueids & uniqueLabels,
		Roidata & roiData,
		// in
		int x, int y, int z, PixIntens intensity, int label, int sidx)
	{
		if (uniqueLabels.find(label) == uniqueLabels.end())
		{
			// Remember this label
			uniqueLabels.insert(label);

			// Initialize the ROI label record
			LR newData (label);
			newData.slide_idx = sidx;
			init_label_record_3D (newData, x, y, z, label, intensity);
			roiData[label] = newData;
		}
		else
		{
			// Update basic ROI info (info that doesn't require costly calculations)
			LR& existingData = roiData[label];
			update_label_record_3D (existingData, x, y, z, label, intensity);
		}
	}

	void feed_pixel_2_cache_LR (int x, int y, PixIntens intensity, LR& r)
	{
		r.raw_pixels.push_back (Pixel2(x, y, intensity));
	}	
	
	void feed_pixel_2_cache_3D_LR (int x, int y, int z, PixIntens intensity, LR& r)
	{
		// update basic ROI info (info that doesn't require costly calculations)
		r.raw_pixels_3D.push_back(Pixel3(x, y, z, intensity));

		// save the index in its z-plane
		size_t idx = r.raw_pixels_3D.size() - 1;
		auto itr = r.zplanes.find(z);
		if (itr == r.zplanes.end())
		{
			std::vector<size_t> newPlane;
			newPlane.push_back(idx);
			r.zplanes[z] = newPlane;
		}
		else
		{
			std::vector<size_t>& existingPlane = (*itr).second;
			existingPlane.push_back(idx);
		}
	}
}
