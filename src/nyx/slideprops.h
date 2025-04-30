#pragma once
#include <string>

struct SlideProps
{
	// pre-ROI intensity range in DP
	double min_preroi_inten;
	double max_preroi_inten;

	// geometric
	size_t slide_w;
	size_t slide_h;
	size_t max_roi_area;
	size_t n_rois;
	size_t max_roi_w;
	size_t max_roi_h;

	// slide file names
	std::string fname_int, fname_seg;

	// low-level slide description (tiled or stripe, pixel intensity type, etc)
	std::string lolvl_slide_descr;

	// annotation
	std::vector<std::string> annots;

};

namespace Nyxus
{
	// Scans segmented slide p.fname_int / p.fname_seg and fills other fields of 'p'
	bool scan_slide_props (SlideProps & p, bool need_annotations);
}