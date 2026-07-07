#pragma once
#include <cmath>          // HU mode: std::floor / std::llround for the offset map
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
		preserve_hu = false;			// HU/CT raw-intensity mode off by default
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

	// HU/CT raw-intensity preservation. When set, the load-time map keeps
	// 1 grey level == 1 intensity unit (offset by the floored global min)
	// instead of min-max rescaling, so absolute Hounsfield values survive and
	// negative CT values no longer wrap on the unsigned cast. Inverted for
	// reporting by IntensityHistogramFeatures::float_domain_map.
	bool preserve_hu;

	// casting x in real range [min_preroi_inten, max_preroi_inten] -> uint [0, uint_dynrange]
	unsigned int uint_friendly_inten (double x, double uint_dynrange) const
	{
		if (preserve_hu)
		{
			// slope-1 offset map: u = round(x - floor(min)). e.g. min=-1024 => HU -1024->0, 0->1024, 3071->4095
			double y = x - std::floor(min_preroi_inten);
			if (y < 0.0) y = 0.0;			// clamp rare sub-min outliers to 0 (stay in unsigned range)
			return (unsigned int) std::llround(y);
		}
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