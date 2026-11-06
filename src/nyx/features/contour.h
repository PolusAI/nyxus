#pragma once
#include <list>
#include <tuple>
#include <vector>
#include <unordered_map>
#include "../dataset.h"
#include "../featureset.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "image_matrix.h"

/// @brief A ROI's contour is the set of its borders, each a closed walk over the ROI pixels that have a 4-neighbour outside the ROI (LR::multicontour_).
class ContourFeature: public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::PERIMETER,
		Nyxus::Feature2D::DIAMETER_EQUAL_PERIMETER,
		Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY,
		Nyxus::Feature2D::EDGE_MAX_INTENSITY,
		Nyxus::Feature2D::EDGE_MIN_INTENSITY,
		Nyxus::Feature2D::EDGE_MEAN_INTENSITY,
		Nyxus::Feature2D::EDGE_STDDEV_INTENSITY
	};

	ContourFeature();
	void calculate (LR& roi, const Fsettings& settings);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& roi, const Fsettings& settings, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void cleanup_instance();
	static void extract (LR& roi, const Fsettings& settings); // extracts the feature of- and saves to ROI
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & fst, const Dataset & ds);

	static bool required(const FeatureSet& fs);

	// contour LEGO:

	static void buildRegularContour (LR &r, const Fsettings &s);

	/// @brief Traces every border of the ROI in a padded image -- the outer border of each 8-connected
	/// fragment and the border of each hole -- as a closed walk of 8-connected steps over the ROI
	/// pixels that have a 4-neighbour outside the ROI.
	/// @param padded_img (w+2) x (h+2) image whose nonzero pixels are the ROI, each holding its
	/// intensity + 1; its one-pixel frame is read as background
	/// @param multicountour one closed walk per border, in padded-image coordinates
	/// @param is_hole when given, (*is_hole)[k] tells whether border k is the border of a hole
	static void gather_multicontour (
		// out
		std::vector<std::vector<Pixel2>> &multicountour,
		// in
		const std::vector<PixIntens>& padded_img,
		int w,
		int h,
		int verbose,
		// out, optional
		std::vector<bool>* is_hole = nullptr);

private:
	void buildRegularContour_nontriv (LR& roi, const Fsettings& settings);
	void buildWholeSlideContour(LR& r);
	std::tuple<double, double, double, double> calc_min_max_mean_stddev_intensity (const std::vector<Pixel2> & contour_pixels);
	double
		fval_PERIMETER = 0, 
		fval_DIAMETER_EQUAL_PERIMETER = 0,
		fval_EDGE_MEAN_INTENSITY = 0, 
		fval_EDGE_STDDEV_INTENSITY = 0, 
		fval_EDGE_MAX_INTENSITY = 0, 
		fval_EDGE_MIN_INTENSITY = 0,
		fval_EDGE_INTEGRATEDINTENSITY = 0;
};

