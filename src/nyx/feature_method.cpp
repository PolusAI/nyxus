#include "feature_method.h"

FeatureMethod::FeatureMethod() 
{
}

bool FeatureMethod::pending()
{
	return pending_calculation;
}

FeatureMgr::FeatureMgr()
{}

void FeatureMgr::sort_by_num_dependencies()
{
}

bool FeatureMgr::roi_cache_item_needed (RoiDataCacheItem item)
{
	for (const auto f : Methods)
		if (f->pending() && f->roi_cache_item_needed(item))
			return true;
	return false;
}

const std::vector<FeatureMethod*>& FeatureMgr::get_requested_features() const
{
	return Methods;
}


