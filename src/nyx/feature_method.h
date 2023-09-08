#pragma once
#include <unordered_map>
#include "roi_cache.h"
#include "image_loader.h"
#include "features/image_matrix_nontriv.h"

/// @brief Abstract class encapsulating basic feature functionality e.g. dependency on other features, dependency on helper objects, state (calculated or pending), etc.
class FeatureMethod
{
	friend class FeatureManager;
public:
	/// @brief User-facing feature name info displayed by class FeatureManager in case of dependency errors
	std::string feature_info;

	FeatureMethod (const std::string& _featureinfo = __FILE__);

	virtual ~FeatureMethod();

	//=== Trivial ROI
	// Calculate the feature for one ROI using cached data and probably caching data
	virtual void calculate (LR& r) = 0;
	// Calculate the feature for a vector of ROIs
	virtual void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)  {}

	//=== Oversized ROI
	virtual void osized_scan_whole_image (LR& r, ImageLoader& imloader);

	virtual void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) = 0;	// Called each time the ROI pixel is being scanned in the raster order
	virtual void osized_reduce() final {};	// Get rid of this method in all derived
	virtual void osized_calculate (LR& r, ImageLoader& imloader) = 0;	// Called once right after having scanned the ROI in the raster order. Put your reduction or summarization of data gathered in osized_add_online_pixel()

	// Put method-dependent set of calculation results in the standard feature results list further saveable as CSV-file
	virtual void save_value(std::vector<std::vector<double>>& feature_vals) = 0;

	// Feature-specific cache clean-up
	virtual void cleanup_instance() {}

	/// @brief
	/// @param F
	void provide_features (const std::initializer_list<Nyxus::AvailableFeatures>& F);	// Queried with provides()

	/// @brief Checks if a feature code belongs to the provided set
	/// @param
	/// @return
	bool provides (Nyxus::AvailableFeatures);	// Looks up in 'provided_features'

	/// @brief
	/// @param F
	void add_dependencies (const std::initializer_list<Nyxus::AvailableFeatures>& F);	// Queried with depends()

	/// @brief Checks if a feature code is among dependencies
	/// @param
	/// @return
	bool depends (Nyxus::AvailableFeatures);	// Looks up in 'dependencies'

private:
	// Dependency manager support
	std::vector<Nyxus::AvailableFeatures> provided_features;
	std::vector<Nyxus::AvailableFeatures> dependencies;
	bool pending_calculation = true;
};
