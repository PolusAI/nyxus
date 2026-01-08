#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "features/aabb.h"
#include "features/image_matrix.h"
#include "features/image_matrix_nontriv.h"
#include "features/image_cube.h"
#include "features/pixel.h"
#include "featureset.h"
#include "roi_cache_basic.h"
#include "slideprops.h"

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

/// @brief Encapsulates ROI properties
class LR: public BasicLR
{
public:
	static constexpr const RoiDataCacheItem CachedObjects[] = { RAW_PIXELS,	CONTOUR, CONVEX_HULL, IMAGE_MATRIX, NEIGHBOR_ROI_LABELS };

	LR (int roi_label);
	LR() : BasicLR(-1) {}	
	bool nontrivial_roi (size_t n_rois, size_t max_ram_bytes);
	bool has_bad_data();
	size_t get_ram_footprint_estimate (size_t n_slide_rois) const;
	size_t get_ram_footprint_estimate_3D (size_t n_volume_rois) const;
	void recycle_aux_obj (RoiDataCacheItem itm);
	bool have_oversize_roi();
	bool caching_permitted();
	void clear_pixels_cache();

	bool blacklisted = false;

	std::vector <Pixel2> raw_pixels;
	std::vector <Pixel3> raw_pixels_3D;
	std::unordered_map<int, std::vector<size_t>> zplanes;  

	OutOfRamPixelCloud raw_pixels_NT;
	unsigned int aux_area = 0;
	PixIntens aux_min, aux_max;

	std::vector<std::vector<Pixel2>> multicontour_;
	void merge_multicontour (std::vector<Pixel2> &flattened_contour) const;

	std::vector<std::vector<size_t>> contours_3D;

	std::vector<Pixel2> convHull_CH;

	std::vector<std::vector<StatsReal>> fvals;
	std::vector<StatsReal> get_fvals (int fcode) const;
	void initialize_fvals();

	std::vector<int> aux_neighboring_labels;

	// 2D
	ImageMatrix aux_image_matrix;	// helper for contour, erosions, texture features, radial distribution, Gabor, moments, ROI radius

	// 3D
	SimpleCube<PixIntens> aux_image_cube;	// helper for texture features and moments
};

/// @brief Encapsulates ROI data related to ROI nesting
class NestedLR: public BasicLR
{
public:
	NestedLR(const LR& r):
		BasicLR (r.label)
	{
		this->aabb = r.aabb;
		this->slide_idx = r.slide_idx;
	}
	NestedLR(): BasicLR(-1) {} // use default label '-1'
	std::vector<int> children;
	std::string segFname;
};
