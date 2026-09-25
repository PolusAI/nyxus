#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "dirs_and_files.h"
#include "globals.h"
#include "helpers/fsystem.h"
#include "helpers/timing.h"
#include "raw_image_loader.h"
#include "ome/format_detect.h"

namespace Nyxus
{
	// An OME-TIFF or OME-Zarr file addresses its channels and timepoints plane by plane, and 2D
	// featurization reads one channel and one timepoint of it (3D featurizes every one). A file
	// carrying more of either is refused in 2D, so no channel or timepoint is left out without a word.
	static bool check_single_channel_timepoint (RawImageLoader& ilo, const SlideProps& p)
	{
		Nyxus::ContainerKind kind = Nyxus::detect_container_family (p.fname_int);
		if (kind != Nyxus::ContainerKind::Tiff && kind != Nyxus::ContainerKind::OmeZarr)
			return true;

		size_t n_chan = ilo.get_inten_channels(),
			n_time = ilo.get_inten_time();
		if (n_chan <= 1 && n_time <= 1)
			return true;

		std::string erm = "Error: " + p.fname_int + " carries " + std::to_string(n_chan) + " channels and "
			+ std::to_string(n_time) + " timepoints; 2D featurization reads one channel and one timepoint of an OME-TIFF or OME-Zarr file";
#ifdef WITH_PYTHON_H
		throw std::runtime_error (erm);
#endif
		std::cerr << erm << "\n";
		return false;
	}

	bool gatherRoisMetrics_2_slideprops_2D_montage (
		// in
		const AnisotropyOptions& aniso,
		// out
		SlideProps& p)
	{
		// low-level slide properties (intensity and mask, if available)
		p.lolvl_slide_descr = "from montage";
		p.fp_phys_pivoxels = false;
		p.integer_rescale = false;

		// time series
		p.inten_time = 0;
		p.mask_time = 0;

		// scan intensity slide's data

		bool wholeslide = false;

		double slide_I_max = (std::numeric_limits<double>::lowest)(),
			slide_I_min = (std::numeric_limits<double>::max)(),
			allpix_I_min = (std::numeric_limits<double>::max)();

		std::unordered_set<int> U;	// unique ROI mask labels
		std::unordered_map <int, LR> R;	// ROI data

		//****** fix ROIs' AABBs with respect to anisotropy

		if (!aniso.customized())
		{
			for (auto& pair : R)
			{
				LR& r = pair.second;
				r.make_nonanisotropic_aabb();
			}
		}
		else
		{
			for (auto& pair : R)
			{
				LR& r = pair.second;
				r.make_anisotropic_aabb(aniso.get_aniso_x(), aniso.get_aniso_y());
			}
		}

		//****** Analysis

		// slide-wide (max ROI area) x (number of ROIs)
		size_t maxArea = 0;
		size_t max_w = 0, max_h = 0;
		for (const auto& pair : R)
		{
			const LR& r = pair.second;
			maxArea = maxArea > r.aux_area ? maxArea : r.aux_area; //std::max (maxArea, r.aux_area);
			const AABB& bb = r.aabb;
			auto w = bb.get_width();
			auto h = bb.get_height();
			max_w = max_w > w ? max_w : w;
			max_h = max_h > h ? max_h : h;
		}

		p.slide_w = 2;
		p.slide_h = 2;

		// a montage never went through a tile loader, so no sample was ever measured
		record_scanned_intensity_range (p, slide_I_min, slide_I_max, allpix_I_min);

		p.max_roi_area = maxArea;
		p.n_rois = R.size();
		p.max_roi_w = max_w;
		p.max_roi_h = max_h;

		return true;
	}

	bool gatherRoisMetrics_2_slideprops_2D(
		// in
		RawImageLoader& ilo,
		const AnisotropyOptions& aniso,
		// out
		SlideProps& p)
	{
		// low-level slide properties (intensity and mask, if available)
		p.lolvl_slide_descr = ilo.get_slide_descr();
		p.fp_phys_pivoxels = ilo.get_fp_phys_pixvoxels();
		p.integer_rescale = ilo.get_integer_rescale (p.rescale_slope, p.rescale_intercept);

		// time series
		p.inten_time = ilo.get_inten_time();
		p.mask_time = ilo.get_mask_time();
		p.inten_channels = ilo.get_inten_channels();		// number of channels (>=1)
		p.phys_x = ilo.get_physical_size_x();				// physical voxel spacing (1.0 if uncalibrated)
		p.phys_y = ilo.get_physical_size_y();
		p.phys_z = ilo.get_physical_size_z();
		p.phys_unit = ilo.get_physical_size_unit();

		if (! check_single_channel_timepoint (ilo, p))
			return false;

		// scan intensity slide's data

		bool wholeslide = p.fname_seg.empty();

		double slide_I_max = (std::numeric_limits<double>::lowest)(),
			slide_I_min = (std::numeric_limits<double>::max)(),
			// the minimum over EVERY pixel, mask or not: the load-time offset has to keep the
			// whole buffer non-negative, not just the part a mask happens to cover
			allpix_I_min = (std::numeric_limits<double>::max)();

		std::unordered_set<int> U;	// unique ROI mask labels
		std::unordered_map <int, LR> R;	// ROI data

		int lvl = 0, // pyramid level
			lyr = 0; //	layer

		// Read the image/volume. The image loader is put in the open state in processDataset_XX_YY ()
		size_t ntHor = ilo.get_num_tiles_hor(),	// tiles across a row
			ntVert = ilo.get_num_tiles_vert(),	// tiles down a column
			fw = ilo.get_tile_width(),
			th = ilo.get_tile_height(),
			tw = ilo.get_tile_width(),
			tileSize = ilo.get_tile_size(),
			fullwidth = ilo.get_full_width(),
			fullheight = ilo.get_full_height();

		// iterate abstract tiles (in a tiled slide /e.g. tiled tiff/ they correspond to physical tiles, in a nontiled slide /e.g. scanline tiff or strip tiff/ they correspond to )
		int cnt = 1;
		for (unsigned int row = 0; row < ntVert; row++)
			for (unsigned int col = 0; col < ntHor; col++)
			{
				// Fetch the tile
				if (!ilo.load_tile(row, col))
				{
#ifdef WITH_PYTHON_H
					throw "Error fetching tile";
#endif	
					std::cerr << "Error fetching tile\n";
					return false;
				}

				// Iterate pixels
				for (size_t i = 0; i < tileSize; i++)
				{
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// the offset the loader will apply is driven by every pixel, so this runs
					// before the mask filter below
					double dxequiv_I = ilo.get_cur_tile_dpequiv_pixel(i);

					// A real-valued slide may hold a non-finite sample. It carries no intensity
					// to measure, and letting one into the extrema takes the whole slide with it:
					// a single infinity gives the quantized map an infinite span, on which every
					// finite pixel maps to NaN. The extrema describe the finite samples; the
					// loaders store a non-finite one as grey level 0.
					bool finite_I = std::isfinite (dxequiv_I);
					if (finite_I)
						allpix_I_min = (std::min)(allpix_I_min, dxequiv_I);

					// Mask
					uint32_t msk = 1; // wholeslide by default
					if (!wholeslide)
						msk = ilo.get_cur_tile_seg_pixel(i);

					// Skip non-mask pixels
					if (!msk)
						continue;

					// dynamic range within- and off-ROI
					if (finite_I)
					{
						slide_I_max = (std::max)(slide_I_max, dxequiv_I);
						slide_I_min = (std::min)(slide_I_min, dxequiv_I);
					}

					// Update pixel's ROI metrics
					//		- the following block mocks feed_pixel_2_metrics (x, y, dataI[i], msk, tidx)
					if (U.find(msk) == U.end())
					{
						// Remember this label
						U.insert(msk);

						// Initialize the ROI label record
						LR r(msk);

						//		- mocking init_label_record_3 (roi, theSegFname, theIntFname, x, y, label, intensity, tile_index)
						// Initialize basic counters
						r.aux_area = 1;
						r.aux_min = r.aux_max = 0; //we don't have uint-cast intensities at this moment
						r.init_aabb(x, y);

						//		- not storing file names (r.segFname = segFile, r.intFname = intFile) but will do so in the future

						// Attach
						R[msk] = r;
					}
					else
					{
						// Update basic ROI info (info that doesn't require costly calculations)
						LR& r = R[msk];

						//		- mocking update_label_record_2 (r, x, y, label, intensity, tile_index)

						// Per-ROI 
						r.aux_area++;

						// save
						r.update_aabb(x, y);
					}
				} // scan tile

				// free tile buffers
				ilo.free_tile_buffers();

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

			} // foreach tile

		//****** fix ROIs' AABBs with respect to anisotropy

		if (!aniso.customized())
		{
			for (auto& pair : R)
			{
				LR& r = pair.second;
				r.make_nonanisotropic_aabb();
			}
		}
		else
		{
			for (auto& pair : R)
			{
				LR& r = pair.second;
				r.make_anisotropic_aabb(aniso.get_aniso_x(), aniso.get_aniso_y());
			}
		}

		//****** Analysis

		// slide-wide (max ROI area) x (number of ROIs)
		size_t maxArea = 0;
		size_t max_w = 0, max_h = 0;
		for (const auto& pair : R)
		{
			const LR& r = pair.second;
			maxArea = maxArea > r.aux_area ? maxArea : r.aux_area; //std::max (maxArea, r.aux_area);
			const AABB& bb = r.aabb;
			auto w = bb.get_width();
			auto h = bb.get_height();
			max_w = max_w > w ? max_w : w;
			max_h = max_h > h ? max_h : h;
		}

		p.slide_w = fullwidth;
		p.slide_h = fullheight;

		record_scanned_intensity_range (p, slide_I_min, slide_I_max, allpix_I_min);

		p.max_roi_area = maxArea;
		p.n_rois = R.size();
		p.max_roi_w = max_w;
		p.max_roi_h = max_h;

		return true;
	}

	bool gatherRoisMetrics_2_slideprops_3D(
		// in
		RawImageLoader& ilo,
		const AnisotropyOptions& aniso,
		bool use_physical_spacing,
		// out
		SlideProps& p)
	{
		// low-level slide properties (intensity and mask, if available)
		p.lolvl_slide_descr = ilo.get_slide_descr();
		p.fp_phys_pivoxels = ilo.get_fp_phys_pixvoxels();
		p.integer_rescale = ilo.get_integer_rescale (p.rescale_slope, p.rescale_intercept);

		// time series
		p.inten_time = ilo.get_inten_time();
		p.mask_time = ilo.get_mask_time();
		p.inten_channels = ilo.get_inten_channels();		// number of channels (>=1)
		p.phys_x = ilo.get_physical_size_x();				// physical voxel spacing (1.0 if uncalibrated)
		p.phys_y = ilo.get_physical_size_y();
		p.phys_z = ilo.get_physical_size_z();
		p.phys_unit = ilo.get_physical_size_unit();

		// scan intensity slide's data

		double slide_I_max = (std::numeric_limits<double>::lowest)(),
			slide_I_min = (std::numeric_limits<double>::max)(),
			// the minimum over EVERY voxel, mask or not: the load-time offset has to keep the
			// whole buffer non-negative, not just the part a mask happens to cover
			allpix_I_min = (std::numeric_limits<double>::max)();

		std::unordered_set<int> U;	// unique ROI mask labels
		std::unordered_map <int, LR> R;	// ROI data

		// The image loader is in the open state, opened by processDataset_XX_YY ().
		size_t fullW = ilo.get_full_width(),
			fullH = ilo.get_full_height(),
			fullD = ilo.get_full_depth();

		// Scan the whole X*Y*Z volume of every (channel, timeframe). The pipeline featurizes every
		// (c,t) plane, so the slide intensity range covers all of them; intensity-indexed buffers
		// are sized from it. An ROI's bounding box covers it in every pass and its size is its
		// largest in any one pass: a mask with a frame per timepoint may differ between frames,
		// and one shared by every plane must not be counted once per plane. for_each_voxel streams
		// each volume tile by tile across every Z-plane's tile grid and hands back (x,y,z) directly.
		const size_t n_chan = (std::max)((size_t)1, p.inten_channels),
			n_time = (std::max)((size_t)1, p.inten_time);
		std::unordered_map <int, unsigned int> passArea;	// each ROI's voxel count in the current pass
		bool ok = true;

		for (size_t scan_c = 0; ok && scan_c < n_chan; scan_c++)
			for (size_t scan_t = 0; ok && scan_t < n_time; scan_t++)
			{
				passArea.clear();
				ok = ilo.for_each_voxel (scan_c, scan_t,
					[&](size_t x, size_t y, size_t z, double dxequiv_I, uint32_t msk)
				{
					// the offset the loader will apply is driven by every voxel, so this runs before
					// the mask filter below
					// Non-finite voxels stay out of the extrema, as in the 2D scan above: one infinity
					// would give the quantized map an infinite span and map every finite voxel to NaN.
					bool finite_I = std::isfinite (dxequiv_I);
					if (finite_I)
						allpix_I_min = (std::min)(allpix_I_min, dxequiv_I);

					// Skip non-mask voxels
					if (!msk)
						return;

					// dynamic range within- and off-ROI
					if (finite_I)
					{
						slide_I_max = (std::max)(slide_I_max, dxequiv_I);
						slide_I_min = (std::min)(slide_I_min, dxequiv_I);
					}

					// Update the ROI's bounding box over all passes and its voxel count in this one
					if (U.find(msk) == U.end())
					{
						// Remember this label
						U.insert(msk);

						// Initialize the ROI label record
						LR r(msk);
						r.aux_min = r.aux_max = 0; //we don't have uint-cast intensities at this moment
						r.init_aabb_3D((int)x, (int)y, (int)z);
						R[msk] = r;
					}
					else
						R[msk].update_aabb_3D((int)x, (int)y, (int)z);

					passArea[msk]++;

#ifdef WITH_PYTHON_H
					// keyboard interrupt
					if (PyErr_CheckSignals() != 0)
						throw pybind11::error_already_set();
#endif

				}); //- all voxels

				for (const auto& pa : passArea)
				{
					LR& r = R[pa.first];
					r.aux_area = (std::max) (r.aux_area, pa.second);
				}
			} //- all (channel, timeframe) volumes

		if (!ok)
		{
#ifdef WITH_PYTHON_H
			throw std::runtime_error ("Error fetching volume");
#endif
			std::cerr << "Error fetching volume\n";
			return false;
		}

		//****** fix ROIs' AABBs with respect to anisotropy, on the spacing every pass over this
		// volume resolves -- explicit --aniso* or, opted in, the slide's physical voxel size. The
		// ROI sizes recorded below drive the memory estimate, so they have to describe the same
		// (resampled) geometry the scans will cache.

		double ax, ay, az;
		bool anisotropic = resolve_anisotropy (aniso, use_physical_spacing, p, ax, ay, az);
		for (auto& pair : R)
		{
			LR& r = pair.second;
			if (anisotropic)
				r.make_anisotropic_aabb (ax, ay, az);
			else
				r.make_nonanisotropic_aabb();
		}

		//****** Analysis

		// slide-wide (max ROI area) x (number of ROIs)
		size_t maxArea = 0;
		size_t max_w = 0, max_h = 0, max_d = 0;
		for (const auto& pair : R)
		{
			const LR& r = pair.second;
			maxArea = (std::max)(maxArea, (size_t)r.aux_area);
			const AABB& bb = r.aabb;
			auto w = bb.get_width();
			auto h = bb.get_height();
			auto d = bb.get_z_depth();
			max_w = (std::max)(max_w, (size_t)w);
			max_h = (std::max)(max_h, (size_t)h);
			max_d = (std::max)(max_d, (size_t)d);
		}

		p.slide_w = fullW;
		p.slide_h = fullH;
		p.volume_d = fullD;

		record_scanned_intensity_range (p, slide_I_min, slide_I_max, allpix_I_min);

		p.max_roi_area = maxArea;
		p.n_rois = R.size();
		p.max_roi_w = max_w;
		p.max_roi_h = max_h;
		p.max_roi_d = max_d;

		return true;
	}

	std::pair <std::string, std::string> split_alnum(const std::string& annot)
	{
		std::string A = annot; // a string that we can edit
		std::string al;
		for (auto c : A)
		{
			if (!std::isdigit(c))
				al += c;
			else
			{
				A.erase(0, al.size());
				break;
			}
		}

		return { al, A };
	}

	bool scan_slide_props_montage (SlideProps & p, int dim, const AnisotropyOptions & aniso)
	{
		if (dim != 2)
			return false;

		gatherRoisMetrics_2_slideprops_2D_montage (aniso, p);

		return true;
	}

	//
	// prerequisite: initialized fields fname_int and  fname_seg
	//
	bool scan_slide_props (SlideProps & p, int dim, const AnisotropyOptions & aniso, bool use_physical_spacing, const FpImageOptions & fpo, bool need_annot)
	{
		RawImageLoader ilo;
		if (! ilo.open(p.fname_int, p.fname_seg))
		{
			std::cerr << "error opening an ImageLoader for " << p.fname_int << " | " << p.fname_seg << "\n";
			return false;
		}

		bool ok = dim==2 ? gatherRoisMetrics_2_slideprops_2D(ilo, aniso, p) : gatherRoisMetrics_2_slideprops_3D(ilo, aniso, use_physical_spacing, p);
		if (!ok)
		{
			std::cerr << "error gathering ROI metrics to slide/volume props \n";
			return false;
		}

		ilo.close();

		// the load-time intensity map, now that the slide's range is known
		record_intensity_domain_map (p, fpo);

		// annotations
		if (need_annot)
		{
			// throw away the directory part
			fs::path pth(p.fname_seg);
			auto purefn = pth.filename().string();

			// LHS part till the 1st dot is the annotation info that we need to parse
			std::vector<std::string>toks1;
			Nyxus::parse_delimited_string (purefn, ".", toks1);

			// the result tokens is the annotation info that we need
			p.annots.clear();
			Nyxus::parse_delimited_string (toks1[0], "_", p.annots);

			// prune blank annotations (usually caused by multiple separator e.g. '__' in 'slide_blah1_bla_2__something3.ome.tiff')
			p.annots.erase (
				std::remove_if (p.annots.begin(), p.annots.end(), [](const std::string & s) { return s.empty(); }),
				p.annots.end());
		}


		return true;
	}

	void record_scanned_intensity_range (SlideProps & p, double slide_I_min, double slide_I_max, double allpix_I_min)
	{
		// No finite sample reached the extrema, so neither was ever assigned and both still hold
		// the sentinel they started from -- the one way the maximum can sit below the minimum. A
		// flat zero range is what such a slide has, and it settles every consumer: the recorded
		// map comes out as the identity, the forward map lands on grey level 0 (which is where
		// the loaders put a non-finite sample), and the ROI range is divided by a zero rather
		// than by an infinity.
		if (slide_I_max < slide_I_min)
			slide_I_min = slide_I_max = 0.0;

		// The all-pixel minimum is measured off the whole buffer rather than off the mask, so it
		// survives an empty mask and is settled separately -- only a reduction that never ran can
		// leave it on its sentinel, and zero is the right offset for a slide with no negative in
		// it either way.
		if (allpix_I_min == (std::numeric_limits<double>::max)())
			allpix_I_min = 0.0;

		p.max_preroi_inten = slide_I_max;		// in case fp_phys_pivoxels==true, max/min _preroi_inten
		p.min_preroi_inten = slide_I_min;		// needs adjusting (grey-binning) before using in wsi scenarios (assigning ROI's min and max)
		p.min_allpix_inten = allpix_I_min;		// drives the load-time offset (see record_intensity_domain_map)
	}

	//
	// Records the map the tile loader will apply to this slide, so the intensity families can
	// report their location statistics in the slide's own domain instead of in grey levels.
	// ImageLoader::open() hands p.inten_offset to the loader, so the two sides cannot drift.
	//
	void record_intensity_domain_map (SlideProps & p, const FpImageOptions & fpo)
	{
		// Default: the loader stores this slide's values as they are.
		p.inten_scale = 1.0;
		p.inten_offset = 0.0;
		p.inten_top_grey = 0.0;
		p.inten_stored_shift = 0.0;
		p.inten_map = IntenMap::native;

		// A slide handed over in memory (montage / numpy input) never went through a tile
		// loader, so nothing mapped it.
		if (p.fname_int.empty())
			return;

		// A floating-point slide is hard-clamped to a range and quantized into
		// [0, target dynamic range], unless preserve_hu asks for the offset map instead.
		if (p.fp_phys_pivoxels && ! p.preserve_hu)
		{
			double fpmin = p.min_preroi_inten,
				fpmax = p.max_preroi_inten;
			if (! fpo.empty())
			{
				fpmin = fpo.min_intensity();
				fpmax = fpo.max_intensity();
			}
			double dr = fpo.target_dyn_range();
			if (dr > 0.0 && fpmax > fpmin)
			{
				p.inten_map = IntenMap::quantized;
				p.inten_scale = (fpmax - fpmin) / dr;
				p.inten_offset = fpmin;
				p.inten_top_grey = dr;		// where the loaders' upper clamp lands, for the forward map
				return;
			}

			// There is no range to quantize into: the slide is constant, or the requested dynamic
			// range is empty. The loader falls back on the offset map either way, so the offset has
			// to be recorded -- leaving it at 0 let the loader truncate a constant 0.5 slide to
			// grey level 0 and the identity inverse then reported 0 as the source value. Unlike the
			// integer branch below, the offset is the real minimum rather than its floor, so a
			// fractional constant survives the round trip exactly.
			p.inten_map = IntenMap::offset;
			p.inten_offset = fpmin;
			return;
		}

		// A slide of integer samples that a header rescale carries into physical units keeps the
		// stored integers as its grey levels and moves the rescale into the recorded inverse:
		// intensity = (intercept + slope*shift) + slope*u. Nothing is narrowed, so no resolution is
		// lost however small the slope -- a PET series with RescaleSlope 0.0005 keeps every one of
		// its stored levels, where rescaling before narrowing would squeeze them into a handful.
		// The shift follows the offset map's rule in stored units, so a slope-1 integer-intercept
		// CT stores exactly the grey levels the offset map would: with a negative physical value
		// anywhere the slide's minimum sits on grey level 0, and otherwise physical 0 stays there.
		// preserve_hu asks for 1 grey level == 1 intensity unit and takes the offset map below
		// instead; so does a slope the inverse cannot carry (zero, negative or non-finite).
		if (p.integer_rescale && ! p.preserve_hu
			&& std::isfinite (p.rescale_slope) && p.rescale_slope > 0.0 && std::isfinite (p.rescale_intercept))
		{
			double shift;
			if (p.min_allpix_inten < 0.0)
			{
				// the stored minimum itself, recovered exactly: stored samples are integers
				shift = std::round ((p.min_allpix_inten - p.rescale_intercept) / p.rescale_slope);
			}
			else
			{
				// The stored value that sits at physical 0, rounded down so no sample goes below it.
				// A quotient within rounding error of an integer is that integer, so an exact case
				// does not lose a level to the division.
				double q = -p.rescale_intercept / p.rescale_slope,
					r = std::round (q);
				shift = std::fabs (q - r) <= 1e-9 * (std::max)(1.0, std::fabs (q)) ? r : std::floor (q);
			}

			p.inten_map = IntenMap::stored;
			p.inten_scale = p.rescale_slope;
			p.inten_stored_shift = shift;
			p.inten_offset = p.rescale_intercept + p.rescale_slope * shift;
			return;
		}

		// Everything else keeps 1 grey level == 1 intensity unit and is shifted only when it has
		// to be: a slide holding a negative intensity anywhere would wrap it on the unsigned
		// cast, so the whole slide is offset by that floored minimum. A slide with no negative
		// pixel is stored as it is, which leaves every ordinary integer image untouched.
		p.inten_map = IntenMap::offset;
		if (p.min_allpix_inten < 0.0)
			p.inten_offset = std::floor (p.min_allpix_inten);
	}
}
