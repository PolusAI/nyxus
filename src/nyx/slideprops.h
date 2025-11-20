#pragma once
#include <string>
#include "cli_anisotropy_options.h"

class SlideProps
{
public:

	SlideProps (const std::string & ifile, const std::string & mfile):
		SlideProps()
	{
		fname_int = ifile;
		fname_seg = mfile;
	}
	SlideProps()
	{
		fname_int = "";
		fname_seg = "";
		min_preroi_inten = -1;
		max_preroi_inten = -1;
		fp_phys_pivoxels = false;
		slide_w = slide_h = volume_d = 0;
		max_roi_area = 0;
		n_rois = 0;
		max_roi_w = 0;
		max_roi_h = 0;
		max_roi_d = 0;
		inten_time = mask_time = 0;
	}

	// pre-ROI intensity range in DP
	double min_preroi_inten;
	double max_preroi_inten;

	// unsigned int grey-minning
	bool fp_phys_pivoxels;

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

	// time series
	size_t inten_time, mask_time;	// number of time frames
};

namespace Nyxus
{
	// Scans segmented slide p.fname_int / p.fname_seg and fills other fields of 'p'
	bool scan_slide_props (SlideProps & p, int dim, const AnisotropyOptions & aniso, bool need_annotations);
}