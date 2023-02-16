#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <sstream>
#include "environment.h"
#include "globals.h"
#include "features/chords.h"
#include "features/ellipse_fitting.h"
#include "features/euler_number.h"
#include "features/circle.h"
#include "features/extrema.h"
#include "features/fractal_dim.h"
#include "features/erosion.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/geodetic_len_thickness.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/glszm.h"
#include "features/gldm.h"
#include "features/hexagonality_polygonality.h"
#include "features/ngtdm.h"
#include "features/image_moments.h"
#include "features/moments.h"
#include "features/neighbors.h"
#include "features/caliper.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/timing.h"
#include "parallel.h"

namespace Nyxus
{
	// Label buffer size related constants
	constexpr int N2R = 100 * 1000;
	constexpr int N2R_2 = 100 * 1000;

	// Preallocates the intensely accessed main containers
	void init_feature_buffers()
	{
		uniqueLabels.reserve(N2R);
		roiData.reserve(N2R);
		zero_background_area = 0;
		labelMutexes.reserve(N2R);
	}

	// Resets the main containers
	void clear_feature_buffers()
	{
		uniqueLabels.clear();
		roiData.clear();
		labelMutexes.clear();
	}

	// Label Record (structure 'LR') is where the state of label's pixels scanning and feature calculations is maintained. This function initializes an LR instance for the 1st pixel.
	void init_label_record(LR& r, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity)
	{
		r.segFname = segFile;
		r.intFname = intFile;

		// Allocate the value matrix
		for (int i = 0; i < AvailableFeatures::_COUNT_; i++)
		{
			std::vector<StatsReal> row{ 0.0 };	// One value initialy. More values can be added for Haralick and Zernike type methods
			r.fvals.push_back(row);
		}

		r.label = label;

		// Initialize the AABB
		r.init_aabb(x, y);

		// Save the pixel
		r.raw_pixels.push_back(Pixel2(x, y, intensity));

		r.fvals[AREA_PIXELS_COUNT][0] = 1;
		if (theEnvironment.xyRes == 0.0)
			r.fvals[AREA_UM2][0] = 0;
		else
			r.fvals[AREA_UM2][0] = std::pow(theEnvironment.pixelSizeUm, 2);

		// Min
		r.fvals[MIN][0] = r.aux_min = intensity;
		// Max
		r.fvals[MAX][0] = r.aux_max = intensity;
		// Moments
		r.fvals[MEAN][0] = intensity;
		r.aux_M2 = 0;
		r.aux_M3 = 0;
		r.aux_M4 = 0;
		r.fvals[ENERGY][0] = intensity * intensity;
		r.aux_variance = 0.0;
		r.fvals[MEAN_ABSOLUTE_DEVIATION][0] = 0.0;
		// Weighted centroids x and y. 1-based for compatibility with Matlab and WNDCHRM
		r.fvals[CENTROID_X][0] = StatsReal(x) + 1;
		r.fvals[CENTROID_Y][0] = StatsReal(y) + 1;

		// Other fields
		r.fvals[MEDIAN][0] = 0;
		r.fvals[STANDARD_DEVIATION][0] = 0;
		r.fvals[SKEWNESS][0] = 0;
		r.fvals[KURTOSIS][0] = 0;
		r.fvals[ROOT_MEAN_SQUARED][0] = 0;
		r.fvals[P10][0] = r.fvals[P25][0] = r.fvals[P75][0] = r.fvals[90][0] = 0;
		r.fvals[INTERQUARTILE_RANGE][0] = 0;
		r.fvals[ENTROPY][0] = 0;
		r.fvals[MODE][0] = 0;
		r.fvals[UNIFORMITY][0] = 0;
		r.fvals[ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = 0;
	}

	void init_label_record_2 (LR& r, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity, unsigned int tile_index)
	{
		// Initialize basic counters
		r.aux_area = 1;
		r.aux_min = r.aux_max = intensity;
		r.init_aabb (x,y);

		// Cache the ROI label
		r.label = label;

		// File names
		r.segFname = segFile;
		r.intFname = intFile;
	}

	// This function 'digests' the 2nd and the following pixel of a label and updates the label's feature calculation state - the instance of structure 'LR'
	void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity)
	{
		// Save the pixel
		if (lr.caching_permitted())
			lr.raw_pixels.push_back(Pixel2(x, y, intensity));
		else
			lr.clear_pixels_cache();

		// Update ROI intensity range
		lr.aux_min = std::min(lr.aux_min, intensity);
		lr.aux_max = std::max(lr.aux_max, intensity);

		// Update ROI bounds
		lr.update_aabb(x, y);
	}

	void update_label_record_2 (LR& lr, int x, int y, int label, PixIntens intensity, unsigned int tile_index)
	{
		// Initialize basic counters
		lr.aux_area++;
		lr.aux_min = std::min(lr.aux_min, intensity);
		lr.aux_max = std::max(lr.aux_max, intensity);
		lr.update_aabb (x,y);
	}

}