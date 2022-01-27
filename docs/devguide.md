# Developer's guide

## Adding a new feature

Adding a feature is a 5-step procedure.

Step 1: Come up with an internal c++ compliant identifier for the feature and its user-facing counterpart, if different from the identifier. Edit enum AvailableFeatures putting the identifier in the end of existing features' identifiers:
```
```

Step 2: edit the integer to string feature identifier mapping
Edit 

Step 2: create a feature method class, say MyFeature, deriving it from class FeatureMethod. Provide implementation of FeatureMethod's pure virtual methods. The class's header and source files are suggested to be placed in directory "features".
```
#include "../feature_method.h"

/// @brief Class MyFeature encapsulates my new feature...
class MyFeature : public FeatureMethod
{
public:
	MyFeature();
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
		val_EXTENT = 0,
		val_MASS_DISPLACEMENT = 0,
		val_WEIGHTED_CENTROID_X = 0,
		val_WEIGHTED_CENTROID_Y = 0;
};

```

