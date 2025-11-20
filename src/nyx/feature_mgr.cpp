#include <string>
#include "feature_mgr.h"
#include "featureset.h"
#include "environment.h"

using namespace Nyxus;

FeatureManager::~FeatureManager()
{
	for (auto f : full_featureset)
		delete f;
	full_featureset.clear();

	user_requested_features.clear();
}

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
	return (int) user_requested_features.size();
}

// Returns the pointer to a feature method instance
FeatureMethod* FeatureManager::get_feature_method(int idx)
{
	return user_requested_features[idx];
}

void FeatureManager::apply_user_selection (FeatureSet & fset)
{
	build_user_requested_set (fset);	// the result will be stored in member 'user_requested_features'
}

// This test checks every feature code 
bool FeatureManager::check_11_correspondence()
{
	FeatureSet fset;
	bool success = true;

	// check the 2D featureset	//xxxx what about 3D and IMQ?
	for (int i_fcode = 0; i_fcode < (int) Nyxus::Feature2D::_COUNT_; i_fcode++)
	{
		int nProviders = 0;
		for (const auto fm : full_featureset)
		{
			if (fm->provides (i_fcode))
				nProviders++;
		}

		if (nProviders == 1)
			continue;	// OK
		else
			if (nProviders > 1)	// error - ambiguous provider (as a class 'XYZ_feature') of a feature (as a code)
			{
				success = false;
				std::cout << "Error: ambiguous provider of feature " << fset.findFeatureNameByCode((Feature2D)i_fcode) << " (code " << i_fcode << ").  (Feature is provided by multiple feature methods.) \n";
			}
			else	// error - no providers
			{
				success = false;
				std::cout << "Error: feature " << fset.findFeatureNameByCode((Feature2D)i_fcode) << " (code " << i_fcode << ") is not provided by any feature method. Check constructor of class FeatureManager\n";
			}
	}

	return success;
}

// This test checks for cyclic feature dependencies
bool FeatureManager::gather_dependencies ()
{
	FeatureSet fset;

	xdeps.clear();

	bool success = true;	// Success in terms of no cycles

	for (const auto fm : full_featureset)
	{
		std::vector<int> extendedDependencies;
		int n_deps = get_num_fmethods_dependencies (fm, extendedDependencies);

		// Any cycle (negative number of depends) ?
		if (n_deps < 0)
		{
			std::cout << "Error: feature method " << fm->feature_info << " has a cyclic dependency \n";
			success = false;
			continue;
		}

#ifdef _DEBUG
		// Feature method instance is good
		std::cout << fm->feature_info << ": " << n_deps << " depends\n";

		// Show the user method's extended dependencies 
		for (auto fcode : extendedDependencies)
		{
			std::string fn = fcode < (int)Feature2D::_COUNT_ ? fset.findFeatureNameByCode((Feature2D)fcode) : fset.findFeatureNameByCode((Feature3D)fcode);
			std::cout << "\t" << fn << "\n";
		}
#endif

		// Bind 'fm' to feature methods implementing fm's extended dependency set
		xdeps.push_back (extendedDependencies);
	}

	return success;
}

int FeatureManager::get_num_fmethods_dependencies (const FeatureMethod * fm, std::vector<int> & parent_dependencies)
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
		parent_dependencies.push_back(fcode); 

		// Find the feature method providing the dependency fcode (the child feature method)
		const FeatureMethod* providerFM = nullptr;
		for (const FeatureMethod* fm : full_featureset)
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
			std::cout << "Error: no registered provider for feature " << (int)fcode << " referenced as dependency by feature method " << fm->feature_info << "\n";
		}

		// Analyze the child
		int n_child_deps = get_num_fmethods_dependencies(
			providerFM,
			parent_dependencies); 
		n_deps += n_child_deps;
	}

	return n_deps;
}

FeatureMethod* FeatureManager::get_feature_method_by_code (int fcode)
{
	for (FeatureMethod* fm : full_featureset)
	{
		for (int providedFcode : fm->provided_features)
			if (providedFcode == fcode)
				return fm;
	}

	return nullptr;
}

// Builds the requested set by copying items of 'featureset' requested via the command line into 'requested_features'
void FeatureManager::build_user_requested_set (FeatureSet & fset)
{
	user_requested_features.clear();

	// Requested feature codes (as integer constants)
	std::vector<std::tuple<std::string, int>> rfc = fset.getEnabledFeatures();

	// Find feature methods implementing them
	for (auto f_info : rfc)
	{
		int fc = std::get<1>(f_info);
		FeatureMethod* fm = get_feature_method_by_code (fc);

		if (fm == nullptr)
			throw (std::runtime_error("Feature " + std::to_string(fc) + " is not provided by any feature method. Check constructor of class FeatureManager"));

		// first, save feature methods of fm's dependencies
		for (auto depend_fc : fm->dependencies)
		{
			FeatureMethod* depend_fm = get_feature_method_by_code ((int)depend_fc);

			if (depend_fm == nullptr)
				throw (std::runtime_error("Feature " + std::to_string((int)depend_fc) + " is not provided by any feature method. Check constructor of class FeatureManager"));

			// save this fm if it's not yet saved
			if (std::find(user_requested_features.begin(), user_requested_features.end(), depend_fm) == user_requested_features.end())
				user_requested_features.push_back(depend_fm);
		}

		// second, save fm itself if it's not yet saved
		if (std::find(user_requested_features.begin(), user_requested_features.end(), fm) == user_requested_features.end())
			user_requested_features.push_back(fm);
	}
}

