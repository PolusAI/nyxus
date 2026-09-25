#pragma once
#include <cmath>
#include <string>
#include "grey_level_cast.h"
#include "cli_anisotropy_options.h"
#include "cli_fpimage_options.h"

// How a loader turned a slide's own intensities into the unsigned grey levels the pipeline stores.
enum class IntenMap
{
	native,		// stored as they are (in-memory montage input, which no tile loader ever mapped)
	offset,		// shifted by the floored slide minimum so negatives survive the unsigned cast
	quantized,	// a floating-point range hard-clamped and rescaled into [0, target dynamic range]
	stored		// a header-rescaled integer slide's stored samples, shifted; the rescale lives in the inverse
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
		inten_top_grey = 0.0;
		inten_stored_shift = 0.0;
		inten_map = IntenMap::native;
		integer_rescale = false;
		rescale_slope = 1.0;
		rescale_intercept = 0.0;
		slide_w = slide_h = volume_d = 0;
		max_roi_area = 0;
		n_rois = 0;
		max_roi_w = 0;
		max_roi_h = 0;
		max_roi_d = 0;
		inten_time = mask_time = 0;
		inten_channels = 1;			// >=1 so the channel loop runs at least once; set from the loader in scan_slide_props
		phys_x = phys_y = phys_z = 1.0;		// physical voxel spacing (1.0 == uncalibrated); set in scan_slide_props
		phys_unit = "";
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

	// Asks a slide to be carried as 1 grey level == 1 intensity unit (offset by the floored
	// slide minimum). A floating-point slide takes that map instead of being min-max rescaled
	// into [0, target dynamic range]; a header-rescaled integer slide (DICOM, NIfTI) takes it
	// instead of the stored map. Other integer slides take the offset map regardless.
	bool preserve_hu;

	// Set by the scan when the slide stores integer samples that a header rescale carries into
	// physical units: physical = rescale_slope * stored + rescale_intercept (DICOM RescaleSlope /
	// RescaleIntercept, NIfTI scl_slope / scl_inter). The scanned extrema are physical either way.
	bool integer_rescale;
	double rescale_slope, rescale_intercept;

	// The load-time map that turned this slide's own intensities into the unsigned grey
	// levels the pipeline stores, held as its inverse: intensity = inten_offset +
	// inten_scale * grey_level. Identity (1, 0) when the loader stored values natively.
	// Recorded by Nyxus::record_intensity_domain_map() once the scan knows the slide's
	// range, handed to the tile loader by ImageLoader::open(), and applied on the way out
	// by the intensity and intensity-histogram families so reported location statistics
	// are in the slide's own domain (Hounsfield units for CT) rather than in grey levels.
	double inten_scale, inten_offset;

	// The highest grey level the quantized map can store -- the target dynamic range the scan
	// was given. Unused (0) on the other two maps, which have no upper end. The forward map
	// needs it because the quantized loaders clamp above as well as below, and the inverse
	// alone cannot say where that clamp sits.
	double inten_top_grey;

	// The stored map's shift, in stored units: the loader keeps grey level = stored - shift, with
	// the header rescale left off. Unused (0) on the other maps, whose loaders shift by
	// inten_offset in the slide's own domain.
	double inten_stored_shift;

	// Which of the load-time maps produced them. The inverse above is all the feature side needs;
	// the loaders need the branch itself, since the quantized map also clamps at its upper end,
	// the offset map only clamps below, and the stored map shifts before any rescale.
	IntenMap inten_map;

	// The forward map: one intensity of this slide -> the grey level the loader stores for it,
	// mirroring each loader map branch for branch. The quantized branch hard-clamps to
	// [fpmin, fpmax] and rescales, exactly as NyxusGrayscaleTiffTileLoader::map_real_intensity
	// and NyxusOmeZarrLoader::map_intensity do -- clamping only below would let an intensity
	// above fpmax map past the top grey level the loader can actually store. The offset branch
	// clamps below only, since that is all its loaders do, and narrows the way they narrow:
	// rounding to nearest under preserve_hu, truncating otherwise. A forward map that rounded
	// where its loader truncates would put the vROI's grey range a level away from the levels
	// actually stored. The stored branch rounds: its loaders store exact integers, and dividing a
	// physical value by the slope recovers one only to within floating-point error.
	unsigned int to_grey_level (double x) const
	{
		// The loaders store a non-finite sample as grey level 0; so does this. It has to run ahead of
		// the quantized clamp, which would otherwise store +Inf as the top grey level.
		if (! std::isfinite (x))
			return 0u;
		if (inten_map == IntenMap::stored)
			return Nyxus::grey_level_rounded<unsigned int> ((x - inten_offset) / inten_scale);
		if (inten_map == IntenMap::quantized)
		{
			// fpmax as ImageLoader::open reconstructs it, so both sides clamp at the same value
			double fpmax = inten_offset + inten_scale * inten_top_grey;
			double t = x < inten_offset ? inten_offset : x;
			t = t > fpmax ? fpmax : t;
			return Nyxus::grey_level_truncated<unsigned int> (inten_top_grey * (t - inten_offset) / (fpmax - inten_offset));
		}
		// The same call the offset loaders make, with the rounding ImageLoader::open hands them.
		return Nyxus::grey_level<unsigned int> ((x - inten_offset) / inten_scale, preserve_hu);
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
		integer_rescale = scanned.integer_rescale;
		rescale_slope = scanned.rescale_slope;
		rescale_intercept = scanned.rescale_intercept;
		inten_scale = scanned.inten_scale;
		inten_offset = scanned.inten_offset;
		inten_top_grey = scanned.inten_top_grey;
		inten_stored_shift = scanned.inten_stored_shift;
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

	// channels (number of intensity channels; drives the per-channel output rows)
	size_t inten_channels;

	// physical voxel spacing (OME PhysicalSize*; 1.0 == uncalibrated). Used only
	// when --use-physical-spacing is set; also emitted as phys_x/y/z + phys_unit output columns.
	double phys_x, phys_y, phys_z;
	std::string phys_unit;
};

namespace Nyxus
{
	// Scans segmented slide p.fname_int / p.fname_seg and fills other fields of 'p'
	bool scan_slide_props (SlideProps & p, int dim, const AnisotropyOptions & aniso, bool use_physical_spacing, const FpImageOptions & fpo, bool need_annotations);

	// Writes the intensity range a scan measured onto 'p'. A scan that met no finite sample --
	// an all-NaN real-valued slide, or a montage, which never scans at all -- leaves its extrema
	// at the sentinels it started from, which is the one way max can end up below min. Such a
	// slide holds no intensity to describe, so a flat zero range is recorded rather than the
	// sentinels: they would otherwise reach the recorded map (whose offset the intensity
	// families add back, reporting DBL_MAX as MIN, MAX and MEAN -- finite, so the output
	// sanitizer passes it through), the whole-slide vROI's grey-level range via
	// to_grey_level(), and COVERED_IMAGE_INTENSITY_RANGE's divisor.
	void record_scanned_intensity_range (SlideProps & p, double slide_I_min, double slide_I_max, double allpix_I_min);

	// Fills p.inten_scale / p.inten_offset with the map the tile loader will apply to this slide.
	// Reads p.fname_int (the loader is picked from the same extension), p.fp_phys_pivoxels,
	// p.preserve_hu, the header rescale and the range the scan just measured, so it runs after
	// scan_slide_props().
	void record_intensity_domain_map (SlideProps & p, const FpImageOptions & fpo);
}