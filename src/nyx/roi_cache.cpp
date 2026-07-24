#include <limits.h>
#include "globals.h"
#include "roi_cache.h"

LR::LR (int lab):
	BasicLR(lab)
{
	slide_idx = -1;
}

bool LR::nontrivial_roi (size_t n_rois, size_t limit) 
{ 
	size_t footprint = get_ram_footprint_estimate (n_rois);
	bool nonTriv = (footprint >= limit);
	return nonTriv; 
}

size_t LR::get_ram_footprint_estimate (size_t n_slide_rois) const
{
	// n_slide_rois==0 means zero OTHER rois to ever hold as neighbors, so that term is 0 bytes --
	// not (size_t)(0-1)*sizeof(int), which underflows to SIZE_MAX and then overflows the multiply
	// to another huge wrapped value. A caller passing an in-progress batch count (which starts at
	// 0) hits this on every batch's first item.
	size_t neighborsBytes = (n_slide_rois > 0) ? (n_slide_rois - 1) * sizeof(int) : 0;
	size_t sz =
		int(Nyxus::FeatureIMQ::_COUNT_) * 10 * sizeof(double) + // feature values (approximately 10 each)
		aabb.get_width() * aabb.get_height() * sizeof(Pixel2) +	// image matrix
		aux_area * sizeof(Pixel2) +	// raw pixels
		neighborsBytes;
	return sz;
}

size_t LR::get_ram_footprint_estimate_3D (size_t n_volume_rois) const
{
	// see get_ram_footprint_estimate() above for why n==0 is guarded rather than left to underflow
	size_t neighborsBytes = (n_volume_rois > 0) ? (n_volume_rois - 1) * sizeof(int) : 0;
	size_t sz =
		int(Nyxus::FeatureIMQ::_COUNT_) * 10 * sizeof(double) + // feature values (approximately 10 each)
		aabb.get_width() * aabb.get_height() * aabb.get_z_depth() * sizeof(Pixel2) +	// image matrix
		aux_area * sizeof(Pixel2) +	// raw pixels
		neighborsBytes;
	return sz;
}

bool LR::have_oversize_roi()
{
	return raw_pixels.size() == 0;
}

bool LR::has_bad_data()
{
	bool bad = (aux_min == aux_max);
	return false; //<-- Permitting nans and infs-- bad;
}

void LR::recycle_aux_obj (RoiDataCacheItem itm)
{
	switch (itm)
	{
	case RAW_PIXELS:
		raw_pixels.clear();
		break;
	case CONTOUR:
		multicontour_.clear();
		break;
	case CONVEX_HULL:
		convHull_CH.clear();
		break;
	case IMAGE_MATRIX:
		aux_image_matrix.clear();
		break;
	case NEIGHBOR_ROI_LABELS:
		aux_neighboring_labels.clear();
		break;
	}
}

bool LR::caching_permitted()
{
	return raw_pixels.size() <= 10000; // some threshold e.g. 10K pixels,  or global flag check, or system RAM resource check
}

void LR::clear_pixels_cache()
{
	recycle_aux_obj(RAW_PIXELS);
}

void LR::rebuild_raw_pixels_from_cloud()
{
	raw_pixels.clear();
	raw_pixels.reserve (raw_pixels_NT.size());
	for (size_t i = 0; i < raw_pixels_NT.size(); i++)
		raw_pixels.push_back (raw_pixels_NT.get_at(i));
}

void LR::rebuild_aux_image_matrix_from_cloud()
{
	std::vector<Pixel2> cloud;
	cloud.reserve (raw_pixels_NT.size());
	for (size_t i = 0; i < raw_pixels_NT.size(); i++)
		cloud.push_back (raw_pixels_NT.get_at(i));

	aux_image_matrix.allocate (aabb.get_width(), aabb.get_height());	// calculate_from_pixelcloud reshapes but does not allocate; size the buffer first
	aux_image_matrix.calculate_from_pixelcloud (cloud, aabb);
}

std::vector<StatsReal> LR::get_fvals (int fcode) const
{
	return fvals[fcode];
}

void LR::initialize_fvals()
{
	fvals.resize ((int)Nyxus::FeatureIMQ::_COUNT_);
	for (auto& valVec : fvals)
		valVec.push_back(0.0);
}

void LR::merge_multicontour (std::vector<Pixel2> &super_c) const
{
	super_c.clear();

	for (const std::vector<Pixel2> &c : multicontour_)
		super_c.insert (std::end(super_c), std::begin(c), std::end(c));
}



