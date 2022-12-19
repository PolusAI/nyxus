#pragma once
#include "../feature_method.h"

/// @brief Encapsulate basic morphological features: area, bounding box, aspect ratio, centroid, weighted centroid, mass displacement, and extent
class BasicMorphologyFeatures : public FeatureMethod
{
public:
	BasicMorphologyFeatures();
	void calculate (LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();

	static bool required(const FeatureSet& fs) {
		return fs.anyEnabled({
			AREA_PIXELS_COUNT,
			AREA_UM2,
			ASPECT_RATIO,
			BBOX_XMIN,
			BBOX_YMIN,
			BBOX_WIDTH,
			BBOX_HEIGHT,
			CENTROID_X,
			CENTROID_Y,
			COMPACTNESS,
			DIAMETER_EQUAL_AREA,
			EXTENT,
			MASS_DISPLACEMENT,
			WEIGHTED_CENTROID_X,
			WEIGHTED_CENTROID_Y });
	}
private:
	double
		val_AREA_PIXELS_COUNT = 0, 
		val_AREA_UM2 = 0, 
		val_ASPECT_RATIO = 0,
		val_BBOX_XMIN = 0,
		val_BBOX_YMIN = 0,
		val_BBOX_WIDTH = 0,
		val_BBOX_HEIGHT = 0,
		val_CENTROID_X = 0,
		val_CENTROID_Y = 0,
		val_COMPACTNESS = 0,
		val_DIAMETER_EQUAL_AREA = 0,
		val_EXTENT = 0,
		val_MASS_DISPLACEMENT = 0,
		val_WEIGHTED_CENTROID_X = 0,
		val_WEIGHTED_CENTROID_Y = 0;
};
