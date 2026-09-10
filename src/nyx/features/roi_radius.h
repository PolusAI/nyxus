#pragma once

#include <vector>
#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include "moments.h"
#include "pixel.h"
#include "../feature_method.h"
#include "../feature_settings.h"

/// @brief Statistics of ROI pixels' distance to the edge.
class RoiRadiusFeature: public FeatureMethod
{
public:
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::ROI_RADIUS_MEAN,
		Nyxus::Feature2D::ROI_RADIUS_MAX,
		Nyxus::Feature2D::ROI_RADIUS_MEDIAN
	};

	RoiRadiusFeature();
	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void extract (LR& roi, const Fsettings& s);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);

	/// @brief Whether a distance transform over the ROI's bounding box beats scanning the contour
	/// per pixel. Both engines are exact and return the same distances, so this decides only how
	/// long a ROI takes to measure, never what it reports. Its three inputs are counts known before
	/// any distance is computed, so reaching the decision costs nothing.
	/// @param cells Bounding box area in raster cells.
	/// @param n_pixels ROI pixel count.
	/// @param n_contour Contour pixel count.
	static bool edt_is_cheaper (size_t cells, size_t n_pixels, size_t n_contour);

	// Compatibility with manual reduce
	static bool required (const FeatureSet& fs) 
	{
		return fs.anyEnabled (RoiRadiusFeature::featureset);
	}

private:
	double max_r = 0, mean_r = 0, median_r = 0;
};