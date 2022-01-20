#pragma once

#include <vector>
#include "feature_method.h"

class FeatureManager
{
public:
	FeatureManager();

	// Registers a feature method
	void register_feature (FeatureMethod*);
	
	// Performs the 1:1 correspondence and cyclic dependency checks and sets up the set of user-requested features
	bool compile();

	void apply_user_selection();

	// After compiling, returns the number of user-requested features
	int get_num_requested_features();

	// Returns the pointer to a feature method instance
	FeatureMethod* get_feature_method (int idx);

private:
	// This test checks if there exists a feature code in Nyxus::AvailableFeatures implemented by multiple feature methods
	bool check_11_correspondence();

	// This test checks for cyclic feature dependencies
	bool check_cycles();

	// Builds the requested set by copying items of 'featureset' requested via the command line into 'requested_features'
	void build_requested_set();

	void external_test_init();
	void clear();
	std::vector<FeatureMethod*> full_featureset;
	std::vector<FeatureMethod*> user_requested_features;
};

namespace Nyxus
{
	extern FeatureMethod
		* pixelIntensityFeatures,
		* contourFeature,
		* convhullFeature,
		* ellipsefitFeature,
		* extremaFeature,
		* eulerNumberFeature, 
		* caliperNassensteinFeature, 
		* caliperFeretFeature, 
		* caliperMartinFeature, 
		* chordsFeature,
		* gaborFeature;
}