#include <limits.h>
#include "globals.h"
#include "roi_cache.h"

PixIntens LR::global_min_inten = 0;
PixIntens LR::global_max_inten = 0;

void LR::reset_global_stats()
{
	LR::global_min_inten = UINT_MAX;
	LR::global_max_inten = 0;
}

LR::LR()
{}

bool LR::nontrivial_roi (size_t memory_limit) 
{ 
	size_t footprint = get_ram_footprint_estimate();
	bool nonTriv = footprint >= memory_limit;
	return nonTriv; 
}

size_t LR::get_ram_footprint_estimate()
{
	size_t sz =
		Nyxus::AvailableFeatures::_COUNT_ * 10 * sizeof(double) + // feature values (approximately 10 each)
		aabb.get_width() * aabb.get_height() * sizeof(Pixel2) +	// image matrix
		aux_area * sizeof(Pixel2) +	// raw pixels
		(uniqueLabels.size() - 1) * sizeof(int);	// neighbors
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
		contour.clear();
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

std::vector<StatsReal> LR::get_fvals(AvailableFeatures af)
{
	return fvals[af];
}

void LR::initialize_fvals()
{
	fvals.resize(AvailableFeatures::_COUNT_);
	for (auto& valVec : fvals)
		valVec.push_back(0.0);
}


