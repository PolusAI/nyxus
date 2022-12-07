#pragma once

#include <vector>
#include "feature_method.h"

/// @brief Dispatcher of feature calculation in the order of their mutual dependence
class FeatureManager
{
public:
	FeatureManager();
	~FeatureManager();

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

	// This test checks for cyclic feature dependencies and populates 'xdeps' 
	bool gather_dependencies();

	int get_num_fmethods_dependencies(FeatureMethod* fm, std::vector<Nyxus::AvailableFeatures> & parent_dependencies);

	// Builds the requested set by copying items of 'featureset' requested via the command line into 'user_requested_features' along with their depended feature methods
	void build_user_requested_set();

	void external_test_init();
	std::vector<FeatureMethod*> full_featureset;
	std::vector<FeatureMethod*> user_requested_features;	// Ordered set of FMs implementing user's feature selection

	std::vector<std::vector<Nyxus::AvailableFeatures>> xdeps;	// Vector of 'full_featureset' items' extended dependencies (as feature codes)

};

