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

// Label record - structure aggregating label's cached data and calculated features
#define DFLT0 -0.0	// default unassigned value
#define DFLT0i -0	// default unassigned value

struct SlideProps
{
	PixIntens min_inten;
	PixIntens max_inten;
	size_t max_roi_area;
	size_t n_rois;
	size_t max_roi_w;
	size_t max_roi_h;
	std::string fname_int;
	std::string fname_seg;
};

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
	size_t get_ram_footprint_estimate_3D();
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
	std::vector<Pixel2> contour;	
	std::unordered_map<int, std::vector<size_t>> contours_3D; // std::unordered_map<int, std::vector<Pixel3>> contours_3D;
	std::vector<Pixel2> convHull_CH;

	std::vector<std::vector<StatsReal>> fvals;
	std::vector<StatsReal> get_fvals (int fcode);
	void initialize_fvals();

	StatsReal aux_M2,
		aux_M3,
		aux_M4,
		aux_variance;

	std::vector<int> aux_neighboring_labels;

	// 2D
	// needed by Contour, Erosions, GLCM, GLRLM, GLSZM, GLDM, NGTDM, Radial distribution, Gabor, Moments, ROI radius
	ImageMatrix aux_image_matrix;
	size_t im_buffer_offset;

	// 3D
	SimpleCube<PixIntens> aux_image_cube;

	// Stats across ROI pixels of the whole image
	static PixIntens global_min_inten;
	static PixIntens global_max_inten;
	static void reset_global_stats();
	// Dataset properties
	static std::vector<SlideProps> dataset_props;
	static size_t dataset_max_combined_roicloud_len;
	static size_t dataset_max_n_rois;
	static size_t dataset_max_roi_area;
	static size_t dataset_max_roi_w;
	static size_t dataset_max_roi_h;
	static void reset_dataset_props();
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
