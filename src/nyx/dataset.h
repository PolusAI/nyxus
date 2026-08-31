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

	// The load-time intensity map of the slide a ROI came from, given as its inverse:
	// intensity = offset + scale * grey_level (see SlideProps::inten_scale). Identity when the
	// slide is unknown -- an in-memory ROI, or one of the scenarios that leaves slide_idx unset.
	void intensity_domain_map (int slide_idx, double & scale, double & offset) const
	{
		scale = 1.0;
		offset = 0.0;
		const SlideProps * p = scanned_slide (slide_idx);
		if (! p)
			return;
		scale = p->inten_scale;
		offset = p->inten_offset;
	}

	// The scanned properties of a slide, or nullptr when the index names no scanned slide.
	// Both callers hand over an index taken from a ROI, which some scenarios leave unset.
	const SlideProps * scanned_slide (int slide_idx) const
	{
		if (slide_idx < 0 || (size_t)slide_idx >= dataset_props.size())
			return nullptr;
		return &dataset_props[slide_idx];
	}

	void update_dataset_props_extrema();

	// clears dataset's slide list
	void reset_dataset_props();

private:
};
