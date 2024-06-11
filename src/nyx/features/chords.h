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
 
class ChordsFeature : public FeatureMethod
{
public:
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
				Nyxus::Feature2D::MAXCHORDS_MAX,
				Nyxus::Feature2D::MAXCHORDS_MAX_ANG,
				Nyxus::Feature2D::MAXCHORDS_MIN,
				Nyxus::Feature2D::MAXCHORDS_MIN_ANG,
				Nyxus::Feature2D::MAXCHORDS_MEDIAN,
				Nyxus::Feature2D::MAXCHORDS_MEAN,
				Nyxus::Feature2D::MAXCHORDS_MODE,
				Nyxus::Feature2D::MAXCHORDS_STDDEV,
				Nyxus::Feature2D::ALLCHORDS_MAX,
				Nyxus::Feature2D::ALLCHORDS_MAX_ANG,
				Nyxus::Feature2D::ALLCHORDS_MIN,
				Nyxus::Feature2D::ALLCHORDS_MIN_ANG,
				Nyxus::Feature2D::ALLCHORDS_MEDIAN,
				Nyxus::Feature2D::ALLCHORDS_MEAN,
				Nyxus::Feature2D::ALLCHORDS_MODE,
				Nyxus::Feature2D::ALLCHORDS_STDDEV
	};

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
		return fs.anyEnabled (ChordsFeature::featureset);
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

