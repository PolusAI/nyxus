#include "feature_mgr.h"
#include "featureset.h"


void FeatureManager::register_feature (FeatureMethod* fm)
{
	full_featureset.push_back(fm);
}

bool FeatureManager::compile()
{
	if (! check_11_correspondence())
	{
		std::cout << "Error compiling features: the 1:1 correspondence check failed\n";
		return false;
	}

	if (!check_cycles())
	{
		std::cout << "Error compiling features: the cycle check failed\n";
		return false;
	}

	build_requested_set();
	return true;
}

// After compiling, returns the number of user-requested features
int FeatureManager::get_num_requested_features()
{
	return user_requested_features.size();
}

// Returns the pointer to a feature method instance
FeatureMethod* FeatureManager::get_feature_method(int idx)
{
	return user_requested_features[idx];
}

void FeatureManager::apply_user_selection()
{
	user_requested_features.clear();
	for (int i=0; i< Nyxus::AvailableFeatures::_COUNT_; i++)
		if (theFeatureSet.isEnabled(i))
		{
			// Iterate ALL the feature methods looking for one that provides user-requested feature [i]
			FeatureMethod* foundFM = nullptr;
			for (FeatureMethod* m : full_featureset)
			{
				// Iterate the 
				bool found = false;		// feature is found to be provided by this feature method
				for (Nyxus::AvailableFeatures pf : m->provided_features)
				{
					if (pf == (Nyxus::AvailableFeatures)i)
					{
						found = true;
						break;
					}
				}

				if (found)
				{
					foundFM = m;
					break;
				}
			}

			// Sanity check
			if (foundFM == nullptr)
				std::cout << "Error: feature " << i << " is not provided by any feature method\n";
			else
			{
				user_requested_features.push_back (foundFM);
				break;
			}
		}
}

void FeatureManager::clear()
{
	full_featureset.clear();

	for (; user_requested_features.size(); )
		user_requested_features.pop_back();
}

bool FeatureManager::check_11_correspondence()
{
	return true;
}

// This test checks for cyclic feature dependencies
bool FeatureManager::check_cycles()
{
	return true;
}

// Builds the requested set by copying items of 'featureset' requested via the command line into 'requested_features'
void FeatureManager::build_requested_set()
{}

#if 0
void FeatureManager::external_test_init()
{
	pixelIntensityFeatures = new PixelIntensityFeatures();
	contourFeature = new ContourFeature();
	convhullFeature = new ConvexHullFeature();
	ellipsefitFeature = new EllipseFittingFeature();
	extremaFeature = new ExtremaFeature();
	eulerNumberFeature = new EulerNumberFeature();
	caliperNassensteinFeature = new CaliperNassensteinFeature();
	caliperFeretFeature = new CaliperFeretFeature();
	caliperMartinFeature = new CaliperMartinFeature();
	chordsFeature = new ChordsFeature();
	gaborFeature = new GaborFeature();
}

namespace Nyxus
{
	FeatureMethod* pixelIntensityFeatures,
		* contourFeature,
		* convhullFeature,
		* ellipsefitFeature,
		* extremaFeature,
		* eulerNumberFeature, 		
		* chordsFeature,
		* caliperNassensteinFeature, 
		* caliperFeretFeature, 
		* caliperMartinFeature,
		* gaborFeature;
}

#endif