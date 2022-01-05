#pragma once
#include "roi_cache.h"

/// @brief Abstract class encapsulating basic feature functionality e.g. dependency on other features, dependency on helper objects, state (calculated or pending), etc.
class FeatureMethod
{
public:
	FeatureMethod();
	virtual void calculate (LR& roi_cache) = 0;
	bool pending();
	virtual bool roi_cache_item_needed (RoiDataCacheItem item) = 0;

protected:
	bool pending_calculation = true;
};

/// @brief Dispatcher class arranging individual features calculation in the correct order.
class FeatureMgr
{
public:
	FeatureMgr();
	void sort_by_num_dependencies();
	const std::vector<FeatureMethod*>& get_requested_features() const;
	bool roi_cache_item_needed(RoiDataCacheItem item);
private:
	std::vector<FeatureMethod*> Methods;
};

