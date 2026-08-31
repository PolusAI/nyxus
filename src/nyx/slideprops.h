#pragma once
#include <string>
#include "cli_anisotropy_options.h"
#include "cli_fpimage_options.h"

// How a loader turned a slide's own intensities into the unsigned grey levels the pipeline stores.
enum class IntenMap
{
	native,		// stored as they are (in-memory montage input, OME-Zarr)
	offset,		// shifted by the floored slide minimum so negatives survive the unsigned cast
	quantized	// a floating-point range hard-clamped and rescaled into [0, target dynamic range]
};

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
		min_allpix_inten = 0;
		fp_phys_pivoxels = false;
		preserve_hu = false;			// float slides: offset map instead of min-max rescale
		inten_scale = 1.0;			// identity until the scan records the load-time map
		inten_offset = 0.0;
		inten_map = IntenMap::native;
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

	// The minimum over every pixel of the slide, masked or not. The load-time offset is driven
	// by this rather than by min_preroi_inten: it has to keep the whole loaded buffer
	// non-negative, and offsetting by a within-mask minimum instead would clamp every off-mask
	// pixel below it and move the ROI's own minimum onto grey level 0, which the grey binning
	// reads as background.
	double min_allpix_inten;

	// unsigned int grey-minning
	bool fp_phys_pivoxels;

	// Asks a floating-point slide to be carried as 1 grey level == 1 intensity unit
	// (offset by the floored slide minimum) instead of being min-max rescaled into
	// [0, target dynamic range], so absolute intensities survive the load. Integer
	// slides and medical volumes (DICOM, NIfTI) are offset-preserved regardless.
	bool preserve_hu;

	// The load-time map that turned this slide's own intensities into the unsigned grey
	// levels the pipeline stores, held as its inverse: intensity = inten_offset +
	// inten_scale * grey_level. Identity (1, 0) when the loader stored values natively.
	// Recorded by Nyxus::record_intensity_domain_map() once the scan knows the slide's
	// range, handed to the tile loader by ImageLoader::open(), and applied on the way out
	// by the intensity and intensity-histogram families so reported location statistics
	// are in the slide's own domain (Hounsfield units for CT) rather than in grey levels.
	double inten_scale, inten_offset;

	// Which of the three load-time maps produced them. The inverse above is all the feature
	// side needs; the loaders need the branch itself, since the quantized map also clamps at
	// its upper end while the offset map only clamps below.
	IntenMap inten_map;

	// The forward map: one intensity of this slide -> the grey level the loader stores for it.
	// Sub-minimum outliers clamp to 0 rather than wrapping on the unsigned cast; the truncating
	// cast is the loaders' own, so this stays their exact mirror.
	unsigned int to_grey_level (double x) const
	{
		double y = (x - inten_offset) / inten_scale;
		if (y < 0.0) y = 0.0;
		return (unsigned int) y;
	}

	// The inverse: one stored grey level -> this slide's own intensity domain.
	double to_source_intensity (double u) const
	{
		return inten_offset + inten_scale * u;
	}

	// Copies the scanned slide's intensity range and load-time map onto a bare SlideProps built
	// to hand a file pair to ImageLoader. Without it the loader would map that pass differently
	// from the prescan -- a bare instance defaults min_preroi_inten to -1 and the map to identity,
	// which puts the pass's grey levels in a domain the features would then misreport.
	void inherit_intensity_domain (const SlideProps & scanned)
	{
		min_preroi_inten = scanned.min_preroi_inten;
		max_preroi_inten = scanned.max_preroi_inten;
		min_allpix_inten = scanned.min_allpix_inten;
		fp_phys_pivoxels = scanned.fp_phys_pivoxels;
		preserve_hu = scanned.preserve_hu;
		inten_scale = scanned.inten_scale;
		inten_offset = scanned.inten_offset;
		inten_map = scanned.inten_map;
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
	bool scan_slide_props (SlideProps & p, int dim, const AnisotropyOptions & aniso, const FpImageOptions & fpo, bool need_annotations);

	// Fills p.inten_scale / p.inten_offset with the map the tile loader will apply to this slide.
	// Reads p.fname_int (the loader is picked from the same extension), p.fp_phys_pivoxels,
	// p.preserve_hu and the range the scan just measured, so it runs after scan_slide_props().
	void record_intensity_domain_map (SlideProps & p, const FpImageOptions & fpo);
}