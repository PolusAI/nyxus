#pragma once

#include <string>
#include <unordered_set>
#include <vector>
#include "features/aabb.h"
#include "features/image_matrix.h"
#include "features/image_matrix_nontriv.h"
#include "features/pixel.h"
#include "featureset.h"
#include "roi_cache_basic.h"

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
class LR: public BasicLR
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

	bool blacklisted = false;

	std::vector <Pixel2> raw_pixels;
	OutOfRamPixelCloud raw_pixels_NT;
	unsigned int aux_area = 0;
	PixIntens aux_min, aux_max;
	std::vector<Pixel2> contour;
	std::vector<Pixel2> convHull_CH;

	std::vector<std::vector<StatsReal>> fvals;
	std::vector<StatsReal> get_fvals(AvailableFeatures af);
	void initialize_fvals();

	StatsReal aux_M2,
		aux_M3,
		aux_M4,
		aux_variance;

	std::vector<int> aux_neighboring_labels;

	// needed by Contour, Erosions, GLCM, GLRLM, GLSZM, GLDM, NGTDM, Radial distribution, Gabor, Moments, ROI radius
	ImageMatrix aux_image_matrix;
	size_t im_buffer_offset;

	// Stats across ROI pixels of the whole image
	static PixIntens global_min_inten;
	static PixIntens global_max_inten;
	static void reset_global_stats();
};

/// @brief Encapsulates ROI data related to ROI nesting
class NestedLR: public BasicLR
{
public:
	NestedLR(const LR& r)
	{
		this->aabb = r.aabb;
		this->label = r.label;
		this->segFname = r.segFname;
		this->intFname = r.intFname;
	}
	NestedLR() {}
	std::vector<int> children;
	std::string segFname;
};
