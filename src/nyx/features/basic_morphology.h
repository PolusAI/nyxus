#pragma once
#include "../feature_method.h"

/// @brief Encapsulate basic morphological features: area, bounding box, aspect ratio, centroid, weighted centroid, mass displacement, and extent
class BasicMorphologyFeatures : public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::AREA_PIXELS_COUNT,
		Nyxus::Feature2D::AREA_UM2,
		Nyxus::Feature2D::ASPECT_RATIO,
		Nyxus::Feature2D::BBOX_XMIN,
		Nyxus::Feature2D::BBOX_YMIN,
		Nyxus::Feature2D::BBOX_WIDTH,
		Nyxus::Feature2D::BBOX_HEIGHT,
		Nyxus::Feature2D::CENTROID_X,
		Nyxus::Feature2D::CENTROID_Y,
		Nyxus::Feature2D::COMPACTNESS,
		Nyxus::Feature2D::DIAMETER_EQUAL_AREA,
		Nyxus::Feature2D::EXTENT,
		Nyxus::Feature2D::MASS_DISPLACEMENT,
		Nyxus::Feature2D::WEIGHTED_CENTROID_X,
		Nyxus::Feature2D::WEIGHTED_CENTROID_Y
	};


	BasicMorphologyFeatures();
	void calculate (LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void extract(LR& roi); // extracts the feature of- and saves to ROI
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();
	static bool required(const FeatureSet& fs);

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
