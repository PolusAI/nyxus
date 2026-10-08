#include <fstream>
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <array>
#include <regex>

#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
#endif

#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"
#include "features/focus_score.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/gldm.h"
#include "features/gldzm.h"
#include "features/glrlm.h"
#include "features/glszm.h"
#include "features/intensity.h"
#include "features/intensity_histogram.h"
#include "features/ngldm.h"
#include "features/ngtdm.h"
#include "features/power_spectrum.h"
#include "features/radial_distribution.h"
#include "features/saturation.h"
#include "features/sharpness.h"
#include "features/zernike.h"

namespace Nyxus
{

	#define disable_DUMP_ALL_ROI
	#ifdef DUMP_ALL_ROI
	void dump_all_roi()
	{
		std::string fpath = theEnvironment.output_dir + "/all_roi.txt";
		std::cout << "Dumping all the ROIs to " << fpath << " ...\n";

		std::ofstream f(fpath);

		for (auto lab : uniqueLabels)
		{
			auto& r = roiData[lab];
			std::cout << "Dumping ROI " << lab << "\n";

			r.aux_image_matrix.print(f);

			f << "ROI " << lab << ": \n"
				<< "xmin = " << r.aabb.get_xmin() << "; \n"
				<< "width=" << r.aabb.get_width() << "; \n"
				<< "ymin=" << r.aabb.get_ymin() << "; \n"
				<< "height=" << r.aabb.get_height() << "; \n"
				<< "area=" << r.aux_area << "; \n";

			// C++ constant:
			f << "// C:\n"
				<< "struct NyxusPixel {\n"
				<< "\tsize_t x, y; \n"
				<< "\tunsigned int intensity; \n"
				<< "}; \n"
				<< "NyxusPixel testData[] = {\n";
			for (auto i=0; i<r.raw_pixels.size(); i++)
			{
				auto& px = r.raw_pixels[i];
				f << "\t{" << px.x-r.aabb.get_xmin() << ", " << px.y- r.aabb.get_ymin() << ", " << px.inten << "}, ";
				if (i > 0 && i % 4 == 0)
					f << "\n";
			}
			f << "}; \n";

			// Matlab constant:
			f << "// MATLAB:\n"
				<< "%==== begin \n";
			f << "pixelCloud = [ \n";
			for (auto i = 0; i < r.raw_pixels.size(); i++)
			{
				auto& px = r.raw_pixels[i];
				f << px.inten << "; % [" << i << "] \n";
			}
			f << "]; \n";

			f << "testData = zeros(" << r.aabb.get_height() << "," << r.aabb.get_width() << ");\n";
			for (auto i = 0; i < r.raw_pixels.size(); i++)
			{
				auto& px = r.raw_pixels[i];
				f << "testData(" << (px.y - r.aabb.get_ymin() + 1) << "," << (px.x - r.aabb.get_xmin() + 1) << ")=" << px.inten << "; ";	// +1 due to 1-based nature of Matlab
				if (i > 0 && i % 4 == 0)
					f << "\n";
			}
			f << "\n";
			f << "testVecZ = reshape(testData, 1, []); \n";
			f << "testVecNZ = nonzeros(testData); \n";
			f << "[mean(testVecNZ) mean(testVecZ) mean2(testData)] \n";
			f << "%==== end \n";
		}

		f.flush();
	}
	#endif

	bool scanTrivialRois (
		const std::vector<int>& batch_labels, 
		const std::string& intens_fpath, 
		const std::string& label_fpath, 
		Environment & env,
		ImageLoader & ldr)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort (whiteList.begin(), whiteList.end());

		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs
		size_t ntHor = ldr.get_num_tiles_hor(),
			ntVert = ldr.get_num_tiles_vert(),
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		size_t cnt = 1;
		for (size_t row = 0; row < ntVert; row++)
			for (size_t col = 0; col < ntHor; col++)
			{
				// Fetch the tile 
				bool ok = ldr.load_tile(row, col);
				if (!ok)
				{
					std::stringstream ss;
					ss << "Error fetching tile row=" << row << " col=" << col;
					#ifdef WITH_PYTHON_H
						throw ss.str();
					#endif	
					std::cerr << ss.str() << "\n";
					return false;
				}

				// Get ahold of tile's pixel buffer
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = ldr.get_seg_tile_sptr();
				bool wholeslide = env.singleROI;

				// Iterate pixels
				for (unsigned long i = 0; i < tileSize; i++)
				{
					// mask label if not in the wholeslide mode
					PixIntens label = 1;
					if (!wholeslide)
						label = (*spL)[i];

					// Merge all foreground labels into a single ROI if requested (must
					// happen before the white-list test: only the merged label 1 is pending)
					if (env.mergeLabels && label)
						label = 1;

					// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
					if (! env.singleROI && ! std::binary_search(whiteList.begin(), whiteList.end(), label))
						continue;

					auto inten = dataI[i];
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Cache this pixel 
					LR& r = env.roiData [label];
					feed_pixel_2_cache_LR (x, y, dataI[i], r);
				}

				VERBOSLVL2 (env.get_verbosity_level(),
					// Show stayalive progress info
					if (cnt++ % 4 == 0)
					{
						static int prevIntPc = 0;
						float pc = Nyxus::round2(100. * float(row * ntHor + col) / float(ntHor * ntVert));
						if (int(pc) != prevIntPc)
						{
							std::cout << "\t" << "scan trivial " << int(pc) << " % \n";
							prevIntPc = int(pc);
						}
					}
				);
			}

		return true;
	}

	bool scanTrivialRois_anisotropic (
		const std::vector<int>& batch_labels,
		const std::string& intens_fpath,
		const std::string& label_fpath,
		Environment& env,
		ImageLoader& ldr,
		double sf_x, 
		double sf_y)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		int lvl = 0,	// pyramid level
			lyr = 0;	//	layer

		// physical slide properties
		size_t ntHor = ldr.get_num_tiles_hor(),
			ntVert = ldr.get_num_tiles_vert(),
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		// virtual slide properties
		size_t vh = (size_t) (double(fullheight) * sf_y),
			vw = (size_t) (double(fullwidth) * sf_x);

		// current tile to skip tile reloads
		size_t curt_x = 999, curt_y = 999;

		for (size_t vr = 0; vr < vh; vr++)
		{
			for (size_t vc = 0; vc < vw; vc++)
			{
				// A virtual pixel's tile is found through its PHYSICAL position, not through a virtual tile
				// width: (tile width * factor) truncates, so virtual_extent / truncated_tile_width can exceed
				// the tile count the slide actually has -- 2048 px of 1024-px tiles at 0.7 gives a virtual
				// width of 1433 over a virtual tile of 716, and the last column asks for tile 2 of 2. Going
				// through the physical column cannot overrun, and needs no bound of its own: the loop puts
				// vc below (size_t)(fullwidth * sf_x), which is at most fullwidth * sf_x, so vc / sf_x is
				// below the slide's width, its tile index is below the grid's, and the within-tile offset
				// is exact rather than accumulated. The row follows the same argument.
				const size_t ph_col = (size_t) (double(vc) / sf_x),
					ph_row = (size_t) (double(vr) / sf_y);
				const size_t tidx_x = ph_col / tw,
					tidx_y = ph_row / th;

				// load it
				if (tidx_y != curt_y || tidx_x != curt_x)
				{
					bool ok = ldr.load_tile(tidx_y, tidx_x);
					if (!ok)
					{
						std::string s = "Error fetching tile row=" + std::to_string(tidx_y) + " col=" + std::to_string(tidx_x);
#ifdef WITH_PYTHON_H
						throw s;
#endif	
						std::cerr << s << "\n";
						return false;
					}

					// cache tile position to avoid reloading
					curt_y = tidx_y;
					curt_x = tidx_x;
				}

				// the physical pixel's offset inside the tile buffer, from the loader that filled it
				const size_t i = ldr.get_within_tile_idx (ph_row, ph_col);

				// read buffered physical pixel 
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = ldr.get_seg_tile_sptr();
				bool wholeslide = env.singleROI;

				PixIntens label = 1;
				if (!wholeslide)
					label = (*spL)[i];

				// not a ROI ?
				if (!label)
					continue;

				// Merge all foreground labels into a single ROI if requested
				if (env.mergeLabels)
					label = 1;

				// skip this ROI if the label isn't in the to-do list 'whiteList' that's only possible in multi-ROI mode
				if (wholeslide==false && !std::binary_search(whiteList.begin(), whiteList.end(), label))
					continue;

				auto inten = dataI[i];

				// cache this pixel 
				// (ROI 'label' is known to the cache by means of gatherRoisMetrics() called previously.)
				LR& r = env.roiData [label];
				feed_pixel_2_cache_LR (vc, vr, inten, r);
			}
		}

		return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi'
	//
	bool scan_trivial_wholeslide (
		LR & vroi,
		const std::string& intens_fpath,
		ImageLoader& ldr)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs
		size_t ntHor = ldr.get_num_tiles_hor(),	// tiles across a row
			ntVert = ldr.get_num_tiles_vert(),	// tiles down a column
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		int cnt = 1;
		for (unsigned int row = 0; row < ntVert; row++)
			for (unsigned int col = 0; col < ntHor; col++)
			{
				// Fetch the tile 
				bool ok = ldr.load_tile(row, col);
				if (!ok)
				{
					std::stringstream ss;
					ss << "Error fetching tile row=" << row << " col=" << col;
#ifdef WITH_PYTHON_H
					throw ss.str();
#endif	
					std::cerr << ss.str() << "\n";
					return false;
				}

				// Get ahold of tile's pixel buffer
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();

				// Iterate pixels
				for (unsigned long i = 0; i < tileSize; i++)
				{
					auto inten = dataI[i];
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight)
						continue;

					// Cache this pixel 
					feed_pixel_2_cache_LR (x, y, dataI[i], vroi);
				}
			}

			return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi' 
	// performing anisotropy correction
	//
	bool scan_trivial_wholeslide_anisotropic (
		LR& vroi,
		const std::string& intens_fpath,
		ImageLoader& ldr,
		double aniso_x,
		double aniso_y)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// physical slide properties
		size_t ntHor = ldr.get_num_tiles_hor(),	// tiles across a row
			ntVert = ldr.get_num_tiles_vert(),	// tiles down a column
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		// virtual slide properties
		size_t vh = (size_t)(double(fullheight) * aniso_y),
			vw = (size_t)(double(fullwidth) * aniso_x);

		// current tile to skip tile reloads
		size_t curt_x = 999, curt_y = 999;

		for (size_t vr = 0; vr < vh; vr++)
		{
			for (size_t vc = 0; vc < vw; vc++)
			{
				// A virtual pixel's tile is found through its PHYSICAL position, not through a virtual tile
				// width: (tile width * factor) truncates, so virtual_extent / truncated_tile_width can exceed
				// the tile count the slide actually has -- 2048 px of 1024-px tiles at 0.7 gives a virtual
				// width of 1433 over a virtual tile of 716, and the last column asks for tile 2 of 2. Going
				// through the physical column cannot overrun, and needs no bound of its own: the loop puts
				// vc below (size_t)(fullwidth * aniso_x), which is at most fullwidth * aniso_x, so
				// vc / aniso_x is below the slide's width, its tile index is below the grid's, and the
				// within-tile offset is exact rather than accumulated. The row follows the same argument.
				const size_t ph_col = (size_t) (double(vc) / aniso_x),
					ph_row = (size_t) (double(vr) / aniso_y);
				const size_t tidx_x = ph_col / tw,
					tidx_y = ph_row / th;

				// load it
				if (tidx_y != curt_y || tidx_x != curt_x)
				{
					bool ok = ldr.load_tile(tidx_y, tidx_x);
					if (!ok)
					{
						std::string s = "Error fetching tile row=" + std::to_string(tidx_y) + " col=" + std::to_string(tidx_x);
#ifdef WITH_PYTHON_H
						throw s;
#endif	
						std::cerr << s << "\n";
						return false;
					}

					// cache tile position to avoid reloading
					curt_y = tidx_y;
					curt_x = tidx_x;
				}

				// the physical pixel's offset inside the tile buffer, from the loader that filled it
				const size_t i = ldr.get_within_tile_idx (ph_row, ph_col);

				// read buffered physical pixel
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();

				// Cache this pixel
				feed_pixel_2_cache_LR (vc, vr, dataI[i], vroi);
			}
		}

		return true;
	}

	void allocateTrivialRoisBuffers (const std::vector<int>& Pending, Roidata& roiData, CpusideCache& cache)
	{
		// Calculate the total memory demand (in # of items) of all segments' image matrices
		cache.imageMatrixBufferLen = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];

			size_t w = r.aabb.get_width(), 
				h = r.aabb.get_height(), 
				imatrSize = w * h;
			cache.imageMatrixBufferLen += imatrSize;

			cache.largest_roi_imatr_buf_len = cache.largest_roi_imatr_buf_len == 0 ? imatrSize : std::max (cache.largest_roi_imatr_buf_len, imatrSize);
		}

		// Lagest ROI
		cache.largest_roi_imatr_buf_len = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];
			cache.largest_roi_imatr_buf_len = cache.largest_roi_imatr_buf_len ? std::max(cache.largest_roi_imatr_buf_len, r.raw_pixels.size()) : r.raw_pixels.size();
		}

		// Allocate image matrices and remember each ROI's image matrix offset in 'ImageMatrixBuffer'
		size_t baseIdx = 0;
		for (auto lab : Pending)
		{
			LR& r = roiData[lab];

			// matrix data
			size_t imatrSize = r.aabb.get_width() * r.aabb.get_height();
			r.aux_image_matrix.allocate (r.aabb.get_width(), r.aabb.get_height());
			baseIdx += imatrSize;	

			// Calculate the image matrix or cube 
			r.aux_image_matrix.calculate_from_pixelcloud (r.raw_pixels, r.aabb);
		}
	}

	void freeTrivialRoisBuffers (const std::vector<int>& roi_labels, Roidata& roiData)
	{
		// Dispose memory of ROIs having their feature calculation finished 
		// in order to give memory ROIs of the next ROI batch. 
		// (Vector 'Pending' is the set of ROIs of the finished batch.)
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			std::vector<Pixel2>().swap(r.raw_pixels);
			std::vector<PixIntens>().swap(r.aux_image_matrix._pix_plane);
			std::vector<Pixel2>().swap(r.convHull_CH);	// convex hull is not a large object but there's no point to keep it beyond the batch, unlike contour
		}
	}

	void freeTrivialRoisBuffers_3D (const std::vector<int>& roi_labels, Roidata& roiData)
	{
		// Dispose memory of ROIs having their feature calculation finished 
		// in order to give memory ROIs of the next ROI batch. 
		// (Vector 'Pending' is the set of ROIs of the finished batch.)
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			std::vector<Pixel3>().swap(r.raw_pixels_3D);

			//
			// Deallocate image matrices, cubes, and convex shells here (in the future).
			//
		}

		// Dispose the buffer of batches' ROIs' image matrices. (We allocate the 
		// image matrix buffer externally to minimize host-GPU transfers.)
//		delete ImageMatrixBuffer;
	}

	bool is_grid_method_2d (FeatureMethod* f)
	{
		return dynamic_cast<PixelIntensityFeatures*>(f) != nullptr
			|| dynamic_cast<IntensityHistogramFeatures*>(f) != nullptr
			|| dynamic_cast<GLCMFeature*>(f) != nullptr
			|| dynamic_cast<GLRLMFeature*>(f) != nullptr
			|| dynamic_cast<GLDZMFeature*>(f) != nullptr
			|| dynamic_cast<GLSZMFeature*>(f) != nullptr
			|| dynamic_cast<GLDMFeature*>(f) != nullptr
			|| dynamic_cast<NGLDMfeature*>(f) != nullptr
			|| dynamic_cast<NGTDMFeature*>(f) != nullptr
			|| dynamic_cast<GaborFeature*>(f) != nullptr
			|| dynamic_cast<FocusScoreFeature*>(f) != nullptr
			|| dynamic_cast<PowerSpectrumFeature*>(f) != nullptr
			|| dynamic_cast<SaturationFeature*>(f) != nullptr
			|| dynamic_cast<SharpnessFeature*>(f) != nullptr;
	}

	void split_2d_selection (const FeatureSet& requested, FeatureSet& on_grid, FeatureSet& geometric)
	{
		geometric = requested;
		for (auto F : { PixelIntensityFeatures::featureset, IntensityHistogramFeatures::featureset,
			GLCMFeature::featureset, GLRLMFeature::featureset, GLDZMFeature::featureset, GLSZMFeature::featureset,
			GLDMFeature::featureset, NGLDMfeature::featureset, NGTDMFeature::featureset, GaborFeature::featureset })
			geometric.enableFeatures (F, false);
		for (auto F : { FocusScoreFeature::featureset, PowerSpectrumFeature::featureset, SaturationFeature::featureset,
			SharpnessFeature::featureset })
			geometric.enableFeatures (F, false);
		// PixelIntensityFeatures also provides HISTOGRAM, which its featureset leaves out so that
		// *ALL_INTENSITY* does not request it
		geometric.enableFeatures ({ Feature2D::HISTOGRAM }, false);

		on_grid = requested;
		on_grid.subtract (geometric);
	}

	size_t trivial_footprint_2d (const Environment& env, const LR& r, size_t n_rois)
	{
		size_t fp = r.get_ram_footprint_estimate (n_rois);
		if (env.anisoOptions.customized())
		{
			const double area = env.anisoOptions.get_aniso_x() * env.anisoOptions.get_aniso_y();
			if (area > 1.0)
				fp = (size_t) (double(fp) * area);
		}
		return fp;
	}

	bool adopt_resampled_cloud_2d (LR& r)
	{
		// A ROI thinner than the factor's step maps to no virtual pixel at all: every virtual
		// coordinate that would carry it truncates back to a physical one outside it
		if (r.raw_pixels.empty())
			return false;

		r.aabb.init_x (r.raw_pixels[0].x);
		r.aabb.init_y (r.raw_pixels[0].y);
		for (const Pixel2& p : r.raw_pixels)
		{
			r.aabb.update_x (p.x);
			r.aabb.update_y (p.y);
		}
		r.aux_area = (unsigned int) r.raw_pixels.size();
		return true;
	}

	void report_unmeasurable_geometry_2d (LR& r, const FeatureSet& geometric, double ax, double ay)
	{
		std::cerr << "Warning: ROI " << r.label << " maps to no pixel at anisotropy " << ax << "," << ay
			<< ", so its geometric features are reported as not available\n";

		// as many values as the writers read for each feature
		const double nan = std::numeric_limits<double>::quiet_NaN();
		for (int f = (int) Feature2D::_FIRST_; f < (int) Feature2D::_COUNT_; f++)
		{
			if (! geometric.isEnabled ((Feature2D) f))
				continue;
			size_t n = 1;
			switch ((Feature2D) f)
			{
			case Feature2D::ZERNIKE2D: n = ZernikeFeature::NUM_FEATURE_VALS; break;
			case Feature2D::FRAC_AT_D: n = RadialDistributionFeature::num_features_FracAtD; break;
			case Feature2D::MEAN_FRAC: n = RadialDistributionFeature::num_features_MeanFrac; break;
			case Feature2D::RADIAL_CV: n = RadialDistributionFeature::num_features_RadialCV; break;
			default: break;
			}
			r.fvals[f].assign (n, nan);
		}
	}

	// Reduces the families 'selection' enables over the ROIs' cached pixels, then frees them. The
	// user's selection is put back on the way out, a throw included.
	static void reduce_trivial_batch_selection (Environment& env, std::vector<int>& Pending, const FeatureSet& selection)
	{
		struct Restore
		{
			Environment& env;
			FeatureSet requested;
			~Restore() { env.theFeatureSet = requested; }
		} restore { env, env.theFeatureSet };

		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tallocating ROI buffers\n");
		allocateTrivialRoisBuffers (Pending, env.roiData, env.hostCache);

#ifdef DUMP_ALL_ROI
		dump_all_roi();
#endif

		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\treducing ROIs\n");
		env.theFeatureSet = selection;
		reduce_trivial_rois_manual (Pending, env);

		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\tfreeing ROI buffers\n");
		freeTrivialRoisBuffers (Pending, env.roiData);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()
	}

	// Scans and reduces one batch of trivial ROIs. An anisotropic batch is scanned twice: the
	// families defined on the image grid reduce the pixels as acquired, then the geometric families
	// reduce the cloud resampled by the spacing. Each pass computes only its own families, so
	// together they fill every requested feature once. After the second pass each ROI's box and
	// pixel count describe the resampled cloud, as its contour does, which is the geometry the
	// neighbor pass after the batches measures. A ROI the resampling leaves with no pixel has its
	// geometric features reported as not available and is added to 'unmeasurable', which the
	// neighbor pass leaves out.
	static bool scan_reduce_trivial_batch (Environment& env, std::vector<int>& Pending, const std::string& intens_fpath, const std::string& label_fpath,
		std::vector<int>& unmeasurable)
	{
		if (! env.anisoOptions.customized())
		{
			if (! scanTrivialRois (Pending, intens_fpath, label_fpath, env, env.theImLoader))
				return false;
			reduce_trivial_batch_selection (env, Pending, env.theFeatureSet);
			return true;
		}

		FeatureSet on_grid, geometric;
		split_2d_selection (env.theFeatureSet, on_grid, geometric);

		if (on_grid.numOfEnabled (2))
		{
			if (! scanTrivialRois (Pending, intens_fpath, label_fpath, env, env.theImLoader))
				return false;
			reduce_trivial_batch_selection (env, Pending, on_grid);
		}
		if (! geometric.numOfEnabled (2))
			return true;

		const double ax = env.anisoOptions.get_aniso_x(),
			ay = env.anisoOptions.get_aniso_y();
		if (! scanTrivialRois_anisotropic (Pending, intens_fpath, label_fpath, env, env.theImLoader, ax, ay))
			return false;
		std::vector<int> measurable;
		for (auto lab : Pending)
		{
			LR& r = env.roiData[lab];
			if (adopt_resampled_cloud_2d (r))
				measurable.push_back (lab);
			else
			{
				report_unmeasurable_geometry_2d (r, geometric, ax, ay);
				unmeasurable.push_back (lab);
			}
		}
		if (! measurable.empty())
			reduce_trivial_batch_selection (env, measurable, geometric);
		return true;
	}

	bool processTrivialRois (Environment & env, const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit)
	{
		std::vector<int> Pending, unmeasurable;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = env.roiData[lab];

			size_t itemFootprint = trivial_footprint_2d (env, r, trivRoiLabels.size());

			// Check if we are good to accumulate this ROI in the current batch or should close the batch and reduce it
			if (batchDemand + itemFootprint < memory_limit)
			{
				// There is room in the ROI batch. Insert another ROI in it
				Pending.push_back(lab);
				batchDemand += itemFootprint;
			}
			else
			{
				// The ROI batch is full. Let's process it 
				std::sort(Pending.begin(), Pending.end());
				VERBOSLVL2 (env.get_verbosity_level(), std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of total " << env.uniqueLabels.size() << " ROIs\n");
				VERBOSLVL2(env.get_verbosity_level(),
					if (Pending.size() == 1)
						std::cout << ">>> (single ROI label " << Pending[0] << ")\n";
					else
						std::cout << ">>> (ROI labels " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				);

				if (! scan_reduce_trivial_batch (env, Pending, intens_fpath, label_fpath, unmeasurable))
					return false;

				// Reset the RAM footprint accumulator
				batchDemand = 0;

				// Clear the freshly processed ROIs from pending list 
				Pending.clear();

				// Start a new pending set by adding the batch-overflowing ROI 
				Pending.push_back(lab);

				// Advance the batch counter
				roiBatchNo++;
			}

			// Allow keyboard interrupt
			#ifdef WITH_PYTHON_H
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
			#endif
		}

		// Process what's remaining pending
		if (Pending.size() > 0)
		{
			// Scan pixels of pending trivial ROIs 
			std::sort (Pending.begin(), Pending.end());
			VERBOSLVL2 (env.get_verbosity_level(), std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of " << env.uniqueLabels.size() << " all ROIs\n");
			VERBOSLVL2 (env.get_verbosity_level(),
				if (Pending.size() == 1)
					std::cout << ">>> (single ROI " << Pending[0] << ")\n";
				else
					std::cout << ">>> (ROIs " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				);

			if (! scan_reduce_trivial_batch (env, Pending, intens_fpath, label_fpath, unmeasurable))
				return false;

			#ifdef WITH_PYTHON_H
			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
			#endif
		}

		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\treducing neighbor features and their depends for all ROIs\n");
		// A ROI with no resampled geometry has no contour for the neighbor pass to measure, so the
		// pass runs without it and its geometric features stay not available
		if (unmeasurable.empty())
			reduce_neighbors_and_dependencies_manual (env);
		else
		{
			const auto all = env.uniqueLabels;
			for (auto lab : unmeasurable)
				env.uniqueLabels.erase (lab);
			reduce_neighbors_and_dependencies_manual (env);
			env.uniqueLabels = all;
		}

		return true;
	}

#ifdef WITH_PYTHON_H

	bool scanTrivialRoisInMemory (
		const std::vector<int>& batch_labels, 
		const py::array_t<unsigned int, 
		py::array::c_style | py::array::forcecast>& intens_images, 
		const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images, 
		int pair_idx,
		Environment & env)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		auto rI = intens_images.unchecked<3>();
		auto rL = label_images.unchecked<3>();
		size_t width = rI.shape(2);
		size_t height = rI.shape(1);

		int cnt = 1;
		for (size_t col = 0; col < width; col++)
			for (size_t row = 0; row < height; row++)
			{
				// Skip non-mask pixels
				auto label = rL (pair_idx, row, col);
				if (!label)
					continue;

				// Merge all foreground labels into a single ROI if requested (before the
				// white-list test, which only contains the merged label 1)
				if (env.mergeLabels)
					label = 1;

				// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
				if (! env.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
					continue;

				auto inten = rI (pair_idx, row, col);

				// Collapse all the labels to one if single-ROI mde is requested
				if (env.singleROI)
					label = 1;

				// Cache this pixel 
				LR& r = env.roiData [label];
				feed_pixel_2_cache_LR (col, row, inten, r);
			}

		return true;
	}

	bool processTrivialRoisInMemory (Environment& env, const std::vector<int>& trivRoiLabels, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intens, const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label, int pair_idx, size_t memory_limit)
	{
		VERBOSLVL4(env.get_verbosity_level(), std::cout << "processTrivialRoisInMemory (pair_idx=" << pair_idx << ") \n");

		std::vector<int> Pending;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = env.roiData[lab];

			size_t itemFootprint = r.get_ram_footprint_estimate(env.uniqueLabels.size());

			// Check if we are good to accumulate this ROI in the current batch or should close the batch and reduce it
			if (batchDemand + itemFootprint < memory_limit)
			{
				Pending.push_back(lab);
				batchDemand += itemFootprint;
			}
			else
			{
				// Scan pixels of pending trivial ROIs 
				std::sort(Pending.begin(), Pending.end());
				scanTrivialRoisInMemory (Pending, intens, label, pair_idx, env);

				// Allocate memory
				allocateTrivialRoisBuffers (Pending, env.roiData, env.hostCache);

				// reduce_trivial_rois(Pending);	
				reduce_trivial_rois_manual (Pending, env);

				// Free memory
				freeTrivialRoisBuffers (Pending, env.roiData);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()

				// Reset the RAM footprint accumulator
				batchDemand = 0;

				// Clear the freshly processed ROIs from pending list 
				Pending.clear();

				// Start a new pending set by adding the stopper ROI 
				Pending.push_back(lab);

				// Advance the batch counter
				roiBatchNo++;
			}

			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();

		}

		// Process what's remaining pending
		if (Pending.size() > 0)
		{
			// Scan pixels of pending trivial ROIs 
			std::sort(Pending.begin(), Pending.end());
			scanTrivialRoisInMemory(Pending, intens, label, pair_idx, env);

			// Allocate memory
			allocateTrivialRoisBuffers (Pending, env.roiData, env.hostCache);

			// Dump ROIs for use in unit testing
#ifdef DUMP_ALL_ROI
			dump_all_roi();
#endif

			// Reduce them
			reduce_trivial_rois_manual (Pending, env);

			// Free memory
			freeTrivialRoisBuffers (Pending, env.roiData);

			// Allow keyboard interrupt
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
		}

		reduce_neighbors_and_dependencies_manual (env);

		return true;

	}

#endif // python api

}
