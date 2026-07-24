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

void FeatureMethod::provide_features (const std::initializer_list<Nyxus::FeatureIMQ> & F)
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

void FeatureMethod::add_dependencies (const std::initializer_list<Nyxus::FeatureIMQ>& F)
{
	for (auto f : F)
		dependencies.push_back((int)f);
}

void FeatureMethod::add_dependencies(const std::initializer_list<Nyxus::Feature3D>& F)
{
	for (auto f : F)
		dependencies.push_back((int)f);
}

// Default out-of-core dispatch: ignore the Dataset and run the Dataset-less osized_calculate,
// so features that don't need slide props are unaffected; only features that do (intensity,
// intensity-histogram) override this and consume the Dataset.
void FeatureMethod::osized_scan_whole_image (LR& r, const Fsettings& s, const Dataset& ds, ImageLoader& ldr)
{
	this->osized_calculate (r, s, ldr);
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

bool FeatureMethod::depends (Nyxus::FeatureIMQ fcode)
{
	return std::find (dependencies.begin(), dependencies.end(), (int)fcode) != dependencies.end();
}

bool FeatureMethod::depends (Nyxus::Feature3D fcode)
{
	return std::find(dependencies.begin(), dependencies.end(), (int)fcode) != dependencies.end();
}
