#pragma once

#include <string>
#include <unordered_set>
#include <vector>
#include "features/aabb.h"
#include "features/image_matrix.h"
#include "features/image_matrix_nontriv.h"
#include "features/pixel.h"
#include "featureset.h"

// Label record - structure aggregating label's cached data and calculated features
#define DFLT0 -0.0	// default unassigned value
#define DFLT0i -0	// default unassigned value

enum RoiDataCacheItem
{
	RAW_PIXELS = 0,
	CONTOUR, 
	CONVEX_HULL, 
	IMAGE_MATRIX, 
	NEIGHBOR_ROI_LABELS
};

/// @brief Encapsulates data cached per each ROI
class LR
{
public:
	static constexpr const RoiDataCacheItem CachedObjects[] = { RAW_PIXELS,	CONTOUR, CONVEX_HULL, IMAGE_MATRIX, NEIGHBOR_ROI_LABELS };

	LR();
	bool nontrivial_roi (size_t memory_limit);
	bool has_bad_data();
	size_t get_ram_footprint_estimate();
	void recycle_aux_obj (RoiDataCacheItem itm);
	bool have_oversize_roi();
	bool caching_permitted();
	void clear_pixels_cache();

	int label;
	std::string segFname, intFname;

	bool roi_disabled = false;

	// Helper objects
	std::vector <Pixel2> raw_pixels;
	OutOfRamPixelCloud osized_pixel_cloud;
	unsigned int aux_area;
	PixIntens aux_min, aux_max;
	AABB aabb;
	std::vector<Pixel2> contour;	// Contour contour;
	std::vector<Pixel2> convHull_CH; // ConvexHull convHull;

	// Replaced with a faster version (class TrivialHistogram)	std::shared_ptr<Histo> aux_Histogram;
	StatsInt
		aux_PrevCount,
		aux_PrevIntens;

	std::vector<std::vector<StatsReal>> fvals;
	std::vector<StatsReal> getFeatureValues(AvailableFeatures af) { return fvals[af]; }

	StatsReal aux_M2,
		aux_M3,
		aux_M4,
		aux_variance;

	std::vector<int> aux_neighboring_labels;

	ImageMatrix aux_image_matrix;	// Needed by Contour, Erosions, GLCM, GLRLM, GLSZM, GLDM, NGTDM, Radial distribution(via Contour), Gabor, Moments, ROI radius(via Contour)

	std::unordered_set <unsigned int> host_tiles;

	void init_aabb(StatsInt x, StatsInt y);
	void update_aabb(StatsInt x, StatsInt y);

	double getValue (AvailableFeatures f);
	void reduce_pixel_intensity_features();
};

