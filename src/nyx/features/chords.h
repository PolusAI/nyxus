#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>
#include "aabb.h"
#include "pixel.h"
#include "../roi_cache.h"
#include "../feature_method.h"

/// @brief Class encapsulating calculating "allchords" and "maxchors" features.
/// An "all chord" refers to all the chords for all ROI rotations.
/// A max chord is the max of all chords for one ROI rotation.
///

#if 0
class Chords_feature
{
public:
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled({
				MAXCHORDS_MAX,
				MAXCHORDS_MAX_ANG,
				MAXCHORDS_MIN,
				MAXCHORDS_MIN_ANG,
				MAXCHORDS_MEDIAN,
				MAXCHORDS_MEAN,
				MAXCHORDS_MODE,
				MAXCHORDS_STDDEV,
				ALLCHORDS_MAX,
				ALLCHORDS_MAX_ANG,
				ALLCHORDS_MIN,
				ALLCHORDS_MIN_ANG,
				ALLCHORDS_MEDIAN,
				ALLCHORDS_MEAN,
				ALLCHORDS_MODE,
				ALLCHORDS_STDDEV, });
	}

	Chords_feature (const std::vector<Pixel2> & raw_pixels, const AABB & bb, const double cenx, const double ceny);

	/// @brief Calculated maxchords statistics
	/// @return Tuple of [0] max, [1] min, [2] median, [3] mean, [4] mode, [5] std, [6] min_angle, [7] max_angle
	std::tuple<double, double, double, double, double, double, double, double> get_maxchords_stats();

	/// @brief Calculated allchords statistics
	/// @return Tuple of [0] max, [1] min, [2] median, [3] mean, [4] mode, [5] std, [6] min_angle, [7] max_angle
	std::tuple<double, double, double, double, double, double, double, double> get_allchords_stats();

	/// @brief Calculates "maxchords" and "allchords" features for a range of ROI labels
	/// @param start First ROI label index
	/// @param end Last ROI label index
	/// @param ptrLabels Vector of ROI labels
	/// @param ptrLabelData Map of numeric ROI labels to ROI data
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	double
		allchords_max = 0,
		allchords_min = 0,
		allchords_median = 0,
		allchords_mean = 0,
		allchords_mode = 0,
		allchords_stddev = 0,
		allchords_min_angle = 0,
		allchords_max_angle = 0,
		maxchords_max = 0,
		maxchords_min = 0,
		maxchords_median = 0,
		maxchords_mean = 0,
		maxchords_mode = 0,
		maxchords_stddev = 0,
		maxchords_min_angle = 0,
		maxchords_max_angle = 0;
};
#endif

class ChordsFeature : public FeatureMethod
{
public:
	ChordsFeature();

	// Trivial
	void calculate(LR& r);

	// Non-trivial
	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {}
	void osized_calculate (LR& r, ImageLoader& imloader);
	void save_value (std::vector<std::vector<double>>& feature_vals);
	static void process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Support of "manual" phase 2
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled({
				MAXCHORDS_MAX,
				MAXCHORDS_MAX_ANG,
				MAXCHORDS_MIN,
				MAXCHORDS_MIN_ANG,
				MAXCHORDS_MEDIAN,
				MAXCHORDS_MEAN,
				MAXCHORDS_MODE,
				MAXCHORDS_STDDEV,
				ALLCHORDS_MAX,
				ALLCHORDS_MAX_ANG,
				ALLCHORDS_MIN,
				ALLCHORDS_MIN_ANG,
				ALLCHORDS_MEDIAN,
				ALLCHORDS_MEAN,
				ALLCHORDS_MODE,
				ALLCHORDS_STDDEV });
	}

private:
	const int n_angle_segments = 20;
	const int n_side_segments = 100;

	double allchords_max = 0,
		allchords_min = 0,
		allchords_median = 0,
		allchords_mean = 0,
		allchords_mode = 0,
		allchords_stddev = 0,
		allchords_min_angle = 0,
		allchords_max_angle = 0,
		maxchords_max = 0,
		maxchords_min = 0,
		maxchords_median = 0,
		maxchords_mean = 0,
		maxchords_mode = 0,
		maxchords_stddev = 0,
		maxchords_min_angle = 0,
		maxchords_max_angle = 0;
};
