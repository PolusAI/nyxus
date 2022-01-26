#include <string>
#include "feature_mgr.h"
#include "featureset.h"
#include "environment.h"


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

	if (!gather_dependencies())	// The result will be in 'xdeps'
	{
		std::cout << "Error compiling features: the cycle check failed\n";
		return false;
	}

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
	/*
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
				throw std::invalid_argument("Error: feature " + std::to_string(i) + " is not provided by any feature method\n");
			else
			{
				user_requested_features.push_back (foundFM);
				break;
			}
		}
	*/

	build_user_requested_set();	// The result is 'user_requested_features'
}

void FeatureManager::clear()
{
	for (auto f : full_featureset)
		delete f;
	full_featureset.clear();

	user_requested_features.clear();
}

// This test checks every feature code 
bool FeatureManager::check_11_correspondence()
{
	bool success = true;

	for (int i_fcode = 0; i_fcode < Nyxus::AvailableFeatures::_COUNT_; i_fcode++)
	{
		int nProviders = 0;
		for (const auto fm : full_featureset)
		{
			Nyxus::AvailableFeatures fcode = (Nyxus::AvailableFeatures) i_fcode;
			if (fm->provides (fcode))
				nProviders++;
		}

		if (nProviders == 1)
			continue;	// OK
		else
			if (nProviders > 1)	// error - ambiguous provider
			{
				success = false;
				std::cout << "Error: ambiguous provider of feature " << theFeatureSet.findFeatureNameByCode((AvailableFeatures)i_fcode) << " (code " << i_fcode << ").  (Feature is provided by multiple feature methods.) \n";
			}
			else	// error - no providers
			{
				success = false;
				std::cout << "Error: feature " << theFeatureSet.findFeatureNameByCode((AvailableFeatures)i_fcode) << " (code " << i_fcode << ") is not provided by any feature method \n";
			}
	}

	return success;
}

// This test checks for cyclic feature dependencies
bool FeatureManager::gather_dependencies()
{
	xdeps.clear();

	bool success = true;	// Success in terms of no cycles

	for (const auto fm : full_featureset)
	{
		std::vector<Nyxus::AvailableFeatures> extendedDependencies;
		int n_deps = get_num_fmethods_dependencies (fm, extendedDependencies);

		// Any cycle (negative number of depends) ?
		if (n_deps < 0)
		{
			std::cout << "Error: feature method " << fm->feature_info << " has a cyclic dependency \n";
			success = false;
			continue;
		}

		// Feature method instance is good
		if (theEnvironment.verbosity_level)
			std::cout << fm->feature_info << ": " << n_deps << " depends\n";

		// Show the user method's extended dependencies 
		for (auto fcode : extendedDependencies)
		{
			std::cout << "\t" << theFeatureSet.findFeatureNameByCode(fcode) << "\n";
		}

		// Bind 'fm' to feature methods implementing fm's extended dependency set
		xdeps.push_back (extendedDependencies);
	}

	return success;
}

int FeatureManager::get_num_fmethods_dependencies (FeatureMethod * fm, std::vector<Nyxus::AvailableFeatures> & parent_dependencies)
{
	// Sanity check
	if (fm == nullptr)
	{
		std::cout << "Invalid feature method passed to FeatureManager::get_num_fmethods_dependencies()\n";
		return 0;
	}

	int n_deps = 0;

	for (auto fcode : fm->dependencies)
	{
		// Check if dependency 'fcode' is cyclic. If so, there's no need to count the number of dependencies. Instead, return -1.
		bool cyclic = std::find(parent_dependencies.begin(), parent_dependencies.end(), fcode) != parent_dependencies.end();
		if (cyclic)
			return -1;

		// Account for the dependency itself (without children)
		n_deps++;


		// Add dependency fcode to the dependency list to be checked
		//?	std::vector<Nyxus::AvailableFeatures> parent_deps2 = parent_dependencies;
		parent_dependencies.push_back(fcode); //? parent_deps2.push_back(fcode);


		// Find the feature method providing the dependency fcode (the child feature method)
		FeatureMethod* providerFM = nullptr;
		for (const auto fm : full_featureset)
		{
			if (fm->provides(fcode))
			{
				providerFM = fm;
				break;
			}
		}

		// No provider?
		if (providerFM == nullptr)
		{
			std::cout << "Error: no registered provider for feature " << fcode << " referenced as dependency by feature method " << fm->feature_info << "\n";
		}

		// Analyze the child
		int n_child_deps = get_num_fmethods_dependencies(
			providerFM,
			parent_dependencies); //? parent_deps2);
		n_deps += n_child_deps;
	}

	return n_deps;
}

// Builds the requested set by copying items of 'featureset' requested via the command line into 'requested_features'
void FeatureManager::build_user_requested_set()
{
	user_requested_features.clear();

	std::vector<std::tuple<FeatureMethod*, int>> requestedWithDepths;

	// iterate FMs
	for (int i = 0; i < full_featureset.size(); i++)
	{
		auto fm = full_featureset[i];

		// figure out if the FM is user-requested (that is, at least any of its FCodes is user-selected)
		bool fmRequested = false;

		// --iterate provided FCodes
		for (auto fcode : fm->provided_features)
			if (theFeatureSet.isEnabled(fcode))
			{
				fmRequested = true;
				break;
			}

		// add requested FM to the execute-list 'requestedWithDepths'
		if (fmRequested)
		{
			auto& deps = xdeps[i];
			std::tuple<FeatureMethod*, int> oneD = { fm, deps.size() };
			if (std::find(requestedWithDepths.begin(), requestedWithDepths.end(), oneD) == requestedWithDepths.end())
				requestedWithDepths.push_back(oneD);

			// iterate dependency FCodes and add corresponding FMs to the execute list
			for (auto dfc : fm->dependencies)
			{
				// find the provider
				for (int k = 0; k < full_featureset.size(); k++)
				{
					auto provFM = full_featureset[k];
					auto& provDeps = xdeps[k];
					oneD = { provFM, provDeps.size() };
					if (std::find(requestedWithDepths.begin(), requestedWithDepths.end(), oneD) == requestedWithDepths.end())
						requestedWithDepths.push_back(oneD);
					break;	// stop searching the provider, proceed to the next dependency-fcode
				}
			}
		}
	}

	// List
	if (theEnvironment.verbosity_level)
	{
		std::cout << "Unsorted:\n";
		for (auto& oneD : requestedWithDepths)
			std::cout << std::get<0>(oneD)->feature_info << " " << std::get<1>(oneD) << " deps \n";
	}

	// Sort by independence
	std::sort(requestedWithDepths.begin(), requestedWithDepths.end(),
		[](const std::tuple<FeatureMethod*, int>& a, const std::tuple<FeatureMethod*, int>& b)
		{
			return std::get<1>(a) < std::get<1>(b);
		});

	if (theEnvironment.verbosity_level)
	{
		// List
		std::cout << "Sorted:\n";
		for (auto& oneD : requestedWithDepths)
			std::cout << std::get<0>(oneD)->feature_info << " " << std::get<1>(oneD) << " deps \n";
	}
}

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