#include "feature_method.h"

FeatureMethod::FeatureMethod(const std::string& _featureinfo)
{
	feature_info = _featureinfo;
}

FeatureMethod::~FeatureMethod(){}

void FeatureMethod::provide_features (const std::initializer_list<Nyxus::Feature2D> & F)
{
	for (auto f : F)
		provided_features.push_back((int)f);
}

void FeatureMethod::provide_features (const std::initializer_list<Nyxus::Feature3D> & F)
{
	for (auto f : F)
		provided_features.push_back((int)f);
}

void FeatureMethod::add_dependencies (const std::initializer_list<Nyxus::Feature2D>& F)
{
	for (auto f : F)
		dependencies.push_back((int)f);
}

void FeatureMethod::add_dependencies(const std::initializer_list<Nyxus::Feature3D>& F)
{
	for (auto f : F)
		dependencies.push_back((int)f);
}

void FeatureMethod::osized_scan_whole_image (LR& r, ImageLoader& imloader)
{
	this->osized_calculate (r, imloader);
	this->save_value (r.fvals);
}

bool FeatureMethod::provides (int fcode) const
{
	return std::find(provided_features.begin(), provided_features.end(), (int)fcode) != provided_features.end();
}

bool FeatureMethod::depends (Nyxus::Feature2D fcode)
{
	return std::find (dependencies.begin(), dependencies.end(), (int)fcode) != dependencies.end();
}

bool FeatureMethod::depends (Nyxus::Feature3D fcode)
{
	return std::find(dependencies.begin(), dependencies.end(), (int)fcode) != dependencies.end();
}
