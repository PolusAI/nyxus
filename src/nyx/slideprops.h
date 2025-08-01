#pragma once
#include <string>

struct SlideProps
{
	// pre-ROI intensity range in DP
	double min_preroi_inten;
	double max_preroi_inten;

	// unsigned int grey-minning
	static unsigned int uint_friendly_dynrange;
	bool fp_phys_pivoxels = false;

	// casting x in real range [min_preroi_inten, max_preroi_inten] -> uint [0, uint_dynrange]
	unsigned int uint_friendly_inten (double x, double uint_dynrange) const
	{
		if (fp_phys_pivoxels)
		{
			double y = x < min_preroi_inten ? min_preroi_inten : x;
			y = y > max_preroi_inten ? max_preroi_inten : y;
			y = uint_dynrange * (y - min_preroi_inten) / (max_preroi_inten - min_preroi_inten);
			return (unsigned int)y;
		}
		else
			return (unsigned int) x;
	}

	// geometric
	size_t slide_w,
		slide_h,
		volume_d,
		max_roi_area,
		n_rois,
		max_roi_w,
		max_roi_h,
		max_roi_d;

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
	bool scan_slide_props (SlideProps & p, int dim, bool need_annotations);
}