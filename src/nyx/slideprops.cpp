#include <limits>
#include <string>
#include <vector>
#include "globals.h"
#include "helpers/fsystem.h"
#include "helpers/timing.h"
#include "raw_image_loader.h"

namespace Nyxus
{
	bool gatherRoisMetrics_2_slideprops_2D_montage (
		// in
		const AnisotropyOptions& aniso,
		// out
		SlideProps& p)
	{
		// low-level slide properties (intensity and mask, if available)
		p.lolvl_slide_descr = "from montage";
		p.fp_phys_pivoxels = false;

		// time series
		p.inten_time = 0;
		p.mask_time = 0;

		// scan intensity slide's data

		bool wholeslide = false;

		double slide_I_max = (std::numeric_limits<double>::lowest)(),
			slide_I_min = (std::numeric_limits<double>::max)();

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

		p.max_preroi_inten = slide_I_max;		// in case fp_phys_pivoxels==true, max/min _preroi_inten 
		p.min_preroi_inten = slide_I_min;		// needs adjusting (grey-binning) before using in wsi scenarios (assigning ROI's min and max)

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

		// time series
		p.inten_time = ilo.get_inten_time();
		p.mask_time = ilo.get_mask_time();

		// scan intensity slide's data

		bool wholeslide = p.fname_seg.empty();

		double slide_I_max = (std::numeric_limits<double>::lowest)(),
			slide_I_min = (std::numeric_limits<double>::max)();

		std::unordered_set<int> U;	// unique ROI mask labels
		std::unordered_map <int, LR> R;	// ROI data

		int lvl = 0, // pyramid level
			lyr = 0; //	layer

		// Read the image/volume. The image loader is put in the open state in processDataset_XX_YY ()
		size_t nth = ilo.get_num_tiles_hor(),
			ntv = ilo.get_num_tiles_vert(),
			fw = ilo.get_tile_width(),
			th = ilo.get_tile_height(),
			tw = ilo.get_tile_width(),
			tileSize = ilo.get_tile_size(),
			fullwidth = ilo.get_full_width(),
			fullheight = ilo.get_full_height();

		// iterate abstract tiles (in a tiled slide /e.g. tiled tiff/ they correspond to physical tiles, in a nontiled slide /e.g. scanline tiff or strip tiff/ they correspond to )
		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
			for (unsigned int col = 0; col < ntv; col++)
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

				// Get ahold of tile's pixel buffer
				auto tidx = row * nth + col;

				// Iterate pixels
				for (size_t i = 0; i < tileSize; i++)
				{
					// Mask
					uint32_t msk = 1; // wholeslide by default
					if (!wholeslide)
						msk = ilo.get_cur_tile_seg_pixel(i);

					// Skip non-mask pixels					
					if (!msk)
						continue;

					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// dynamic range within- and off-ROI
					double dxequiv_I = ilo.get_cur_tile_dpequiv_pixel(i);
					slide_I_max = (std::max)(slide_I_max, dxequiv_I);
					slide_I_min = (std::min)(slide_I_min, dxequiv_I);

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

		p.max_preroi_inten = slide_I_max;		// in case fp_phys_pivoxels==true, max/min _preroi_inten 
		p.min_preroi_inten = slide_I_min;		// needs adjusting (grey-binning) before using in wsi scenarios (assigning ROI's min and max)

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
		// out
		SlideProps& p)
	{
		// low-level slide properties (intensity and mask, if available)
		p.lolvl_slide_descr = ilo.get_slide_descr();
		p.fp_phys_pivoxels = ilo.get_fp_phys_pixvoxels();

		// time series
		p.inten_time = ilo.get_inten_time();
		p.mask_time = ilo.get_mask_time();

		// scan intensity slide's data

		bool wholeslide = p.fname_seg.empty();

		double slide_I_max = (std::numeric_limits<double>::lowest)(),
			slide_I_min = (std::numeric_limits<double>::max)();

		std::unordered_set<int> U;	// unique ROI mask labels
		std::unordered_map <int, LR> R;	// ROI data

		// Read the volume. The image loader is in the open state by previously called processDataset_XX_YY ()
		size_t fullW = ilo.get_full_width(),
			fullH = ilo.get_full_height(),
			fullD = ilo.get_full_depth(),
			sliceSize = fullW * fullH,
			nVox = sliceSize * fullD;

		// in the 3D case tiling is a formality, so fetch the only tile in the file
		if (!ilo.load_tile(0, 0))
		{
#ifdef WITH_PYTHON_H
			throw "Error fetching tile";
#endif	
			std::cerr << "Error fetching tile\n";
			return false;
		}

		// iterate abstract tiles (in a tiled slide /e.g. tiled tiff/ they correspond to physical tiles, in a nontiled slide /e.g. scanline tiff or strip tiff/ they correspond to )
		int cnt = 1;

		// iterate voxels
		for (size_t i = 0; i < nVox; i++)
		{
			// Mask
			uint32_t msk = 1; // wholeslide by default
			if (!wholeslide)
				msk = ilo.get_cur_tile_seg_pixel(i);

			// Skip non-mask pixels					
			if (!msk)
				continue;

			int z = i / sliceSize,
				y = (i - z * sliceSize) / fullW,
				x = (i - z * sliceSize) % fullW;

			// Skip tile buffer pixels beyond the image's bounds
			if (x >= fullW || y >= fullH || z >= fullD)
				continue;

			// dynamic range within- and off-ROI
			double dxequiv_I = ilo.get_cur_tile_dpequiv_pixel(i);
			slide_I_max = (std::max)(slide_I_max, dxequiv_I);
			slide_I_min = (std::min)(slide_I_min, dxequiv_I);

			// Update pixel's ROI metrics
			//		- the following block mocks feed_pixel_2_metrics (x, y, dataI[i], msk, tidx)
			if (U.find(msk) == U.end())
			{
				// Remember this label
				U.insert(msk);

				// Initialize the ROI label record
				LR r(msk);

				//		- mocking init_label_record_3 (newData, theSegFname, theIntFname, x, y, label, intensity, tile_index)
				// Initialize basic counters
				r.aux_area = 1;
				r.aux_min = r.aux_max = 0; //we don't have uint-cast intensities at this moment
				r.init_aabb_3D(x, y, z);

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
				r.update_aabb_3D(x, y, z);
			}

#ifdef WITH_PYTHON_H
			// keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
#endif

		} //- all voxels

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
				r.make_anisotropic_aabb(aniso.get_aniso_x(), aniso.get_aniso_y(), aniso.get_aniso_z());
			}
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

		p.max_preroi_inten = slide_I_max;		// in case fp_phys_pivoxels==true, max/min _preroi_inten 
		p.min_preroi_inten = slide_I_min;		// needs adjusting (grey-binning) before using in wsi scenarios (assigning ROI's min and max)

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
	bool scan_slide_props (SlideProps & p, int dim, const AnisotropyOptions & aniso, bool need_annot)
	{
		RawImageLoader ilo;
		if (! ilo.open(p.fname_int, p.fname_seg))
		{
			std::cerr << "error opening an ImageLoader for " << p.fname_int << " | " << p.fname_seg << "\n";
			return false;
		}

		bool ok = dim==2 ? gatherRoisMetrics_2_slideprops_2D(ilo, aniso, p) : gatherRoisMetrics_2_slideprops_3D(ilo, aniso, p);
		if (!ok)
		{
			std::cerr << "error gathering ROI metrics to slide/volume props \n";
			return false;
		}

		ilo.close();

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
}