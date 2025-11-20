#pragma once

#include <vector>
#include "slideprops.h"

class Dataset
{
public:

	Dataset() {}

	std::vector<SlideProps> dataset_props;
	size_t dataset_max_combined_roicloud_len;
	size_t dataset_max_n_rois;
	size_t dataset_max_roi_area;
	size_t dataset_max_roi_w;
	size_t dataset_max_roi_h;
	size_t dataset_max_roi_d;

	void update_dataset_props_extrema();

	// clears dataset's slide list
	void reset_dataset_props();

private:
};
