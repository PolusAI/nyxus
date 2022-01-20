#include "feature_method.h"

FeatureMethod::FeatureMethod() 
{
}

void FeatureMethod::provide_features (const std::initializer_list<Nyxus::AvailableFeatures>& F)
{
	for (auto f : F)
		provided_features.push_back(f);
}

void FeatureMethod::add_dependencies (const std::initializer_list<Nyxus::AvailableFeatures>& F)
{
	for (auto f : F)
		dependencies.push_back(f);
}

void FeatureMethod::osized_scan_whole_image (LR& r, ImageLoader& imloader)
{
	this->osized_calculate (r, imloader);
	this->save_value (r.fvals);
}

