#pragma once
#include <string>

struct SlideProps
{
	// pre-ROI intensity range in DP
	double min_preroi_inten;
	double max_preroi_inten;
	// geometric
	size_t max_roi_area;
	size_t n_rois;
	size_t max_roi_w;
	size_t max_roi_h;

	std::string fname_int;
	std::string fname_seg;
};

