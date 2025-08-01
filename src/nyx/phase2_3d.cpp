#include <fstream>
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

namespace Nyxus
{

	// Objects that are used by allocateTrivialRoisBuffers() and then by the GPU platform code 
	// to transfer image matrices of all the image's ROIs
	extern PixIntens* ImageMatrixBuffer;
	extern size_t imageMatrixBufferLen;
	extern size_t largest_roi_imatr_buf_len;

	bool scanTrivialRois_3D (const std::vector<int>& batch_labels, const std::string& intens_fpath, const std::string& label_fpath)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Scan this Z intensity-mask pair 
		SlideProps p;
		p.fname_int = intens_fpath;
		p.fname_seg = label_fpath;
		if (!theImLoader.open(p))
		{
			std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
			return false;
		}

		// Cache the file names to be picked up by labels to know their file origin
		theIntFname = intens_fpath;
		theSegFname = label_fpath;

		size_t nth = theImLoader.get_num_tiles_hor(),
			ntv = theImLoader.get_num_tiles_vert(),
			fw = theImLoader.get_tile_width(),
			th = theImLoader.get_tile_height(),
			tw = theImLoader.get_tile_width(),
			tileSize = theImLoader.get_tile_size(),
			fullwidth = theImLoader.get_full_width(),
			fullheight = theImLoader.get_full_height(),
			fullD = theImLoader.get_full_depth(),
			sliceSize = fullwidth * fullheight,
			nVox = sliceSize * fullD;	

		int cnt = 1;
		for (unsigned int row = 0; row < nth; row++)
		{
			for (unsigned int col = 0; col < ntv; col++)
			{
				// Fetch the tile 
				bool ok = theImLoader.load_tile(row, col);
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
				auto dataI = theImLoader.get_int_tile_buffer(),
					dataL = theImLoader.get_seg_tile_buffer();

				// Iterate voxels
				for (size_t i = 0; i < nVox; i++)
				{
					// Skip non-mask pixels
					auto label = dataL[i];
					if (!label)
						continue;

					// Skip this ROI if the label isn't in the pending set of a multi-ROI mode
					if (!theEnvironment.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
						continue;


					int z = i / sliceSize,
						y = (i - z * sliceSize) / tw,
						x = (i - z * sliceSize) % tw;

					// Skip tile buffer pixels beyond the image's bounds
					if (x >= fullwidth || y >= fullheight || z >= fullD)
						continue;

					// Collapse all the labels to one if single-ROI mde is requested
					if (theEnvironment.singleROI)
						label = 1;

					// Cache this pixel 
					LR& r = roiData[label];
					feed_pixel_2_cache_3D_LR (x, y, z, dataI[i], r);
				}

#ifdef WITH_PYTHON_H
				if (PyErr_CheckSignals() != 0)
					throw pybind11::error_already_set();
#endif

				// Show stayalive progress info
				VERBOSLVL2(
					if (cnt++ % 4 == 0)
						std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n";
				);
			} // tile cols
		} // tile rows

		// Close the image pair
		theImLoader.close();

		// Dump ROI pixel clouds to the output directory
		VERBOSLVL5(dump_roi_pixels(batch_labels, label_fpath))

		return true;
	}

	bool scanTrivialRois_3D_anisotropic(
		const std::vector<int>& batch_labels,
		const std::string& intens_fpath,
		const std::string& label_fpath,
		double aniso_x,
		double aniso_y,
		double aniso_z)
	{
		// Sort the batch's labels to enable binary searching in it
		std::vector<int> whiteList = batch_labels;
		std::sort(whiteList.begin(), whiteList.end());

		int lvl = 0,	// pyramid level
			lyr = 0;	//	layer

		// Scan this Z intensity-mask pair 
		SlideProps p;
		p.fname_int = intens_fpath;
		p.fname_seg = label_fpath;
		if (!theImLoader.open(p))
		{
			std::cerr << "Error opening a file pair with ImageLoader. Terminating\n";
			return false;
		}

		// Cache the file names to be picked up by labels to know their file origin
		theIntFname = intens_fpath;
		theSegFname = label_fpath;

		size_t nth = theImLoader.get_num_tiles_hor(),
			ntv = theImLoader.get_num_tiles_vert(),
			fw = theImLoader.get_tile_width(),
			th = theImLoader.get_tile_height(),
			tw = theImLoader.get_tile_width(),
			tileSize = theImLoader.get_tile_size(),
			fullW = theImLoader.get_full_width(),
			fullH = theImLoader.get_full_height(),
			fullD = theImLoader.get_full_depth();

		size_t vD = (size_t)(double(fullD) * aniso_z);	// virtual depth

		for (size_t vz = 0; vz < vD; vz++)
		{
			size_t z = size_t(double(vz) / aniso_z);	// physical z

			// virtual slide properties
			size_t vh = (size_t)(double(fullH) * aniso_y),
				vw = (size_t)(double(fullW) * aniso_x),
				vth = (size_t)(double(th) * aniso_y),
				vtw = (size_t)(double(tw) * aniso_x);

			// current tile to skip tile reloads
			size_t curt_x = 999, curt_y = 999;

			for (size_t vr = 0; vr < vh; vr++)
			{
				for (size_t vc = 0; vc < vw; vc++)
				{
					// tile position
					size_t tidx_y = size_t(vr / vth),
						tidx_x = size_t(vc / vtw);

					// load it
					if (tidx_y != curt_y || tidx_x != curt_x)
					{
						bool ok = theImLoader.load_tile(tidx_y, tidx_x);
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

					// within-tile virtual pixel position
					size_t vx = vc - tidx_x * vtw,
						vy = vr - tidx_y * vth;

					// within-tile physical pixel position
					size_t ph_x = size_t(double(vx) / aniso_x),
						ph_y = size_t(double(vy) / aniso_y),
						i = ph_y * tw + ph_x;

					// read buffered physical pixel 
					auto dataI = theImLoader.get_int_tile_buffer(),
						dataL = theImLoader.get_seg_tile_buffer();

					// skip non-mask pixels
					auto label = dataL[i];
					if (!label)
						continue;

					// skip this ROI if the label isn't in the pending set of a multi-ROI mode
					if (!theEnvironment.singleROI && !std::binary_search(whiteList.begin(), whiteList.end(), label))
						continue;

					// skip tile buffer pixels beyond the image's bounds
					if (vc >= fullW || vr >= fullH)
						continue;

					// collapse all the labels to one if single-ROI mde is requested
					if (theEnvironment.singleROI)
						label = 1;

					// cache this voxel 
					auto inten = dataI[i];
					LR& r = roiData[label];
					feed_pixel_2_cache_3D_LR (vc, vr, vz, dataI[i], r);
				}
			}

			// Close the image pair
			theImLoader.close();
		}

		// Dump ROI pixel clouds to the output directory
		VERBOSLVL5(dump_roi_pixels(batch_labels, label_fpath))

			return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi'
	//
	bool scan_trivial_wholevolume (
		LR& vroi,
		const std::string& intens_fpath,
		ImageLoader& ilo)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs

		size_t fullwidth = ilo.get_full_width(),
			fullheight = ilo.get_full_height(),
			fullD = ilo.get_full_depth(),
			sliceSize = fullwidth * fullheight,
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

		// Get ahold of tile's pixel buffer
		const std::vector<uint32_t>& dataI = ilo.get_int_tile_buffer();

		// iterate abstract tiles (in a tiled slide /e.g. tiled tiff/ they correspond to physical tiles, in a nontiled slide /e.g. scanline tiff or strip tiff/ they correspond to )
		int cnt = 1;

		// iterate voxels
		for (size_t i = 0; i < nVox; i++)
		{
			int z = i / sliceSize,
				y = (i - z * sliceSize) / fullwidth,
				x = (i - z * sliceSize) % fullwidth;

			// Skip tile buffer pixels beyond the image's bounds
			if (x >= fullwidth || y >= fullheight || z >= fullD)
				continue;

			// dynamic range within- and off-ROI
			auto inten = dataI[i];

			// Cache this pixel 
			feed_pixel_2_cache_3D_LR (x, y, z, inten, vroi);

		} //- all voxels

		return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi'
	//
	bool scan_trivial_wholevolume_anisotropic (
		LR& vroi,
		const std::string& intens_fpath,
		ImageLoader& ilo,
		double aniso_x,
		double aniso_y,
		double aniso_z)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// Read the tiffs

		size_t fullW = ilo.get_full_width(),
			fullH = ilo.get_full_height(),
			fullD = ilo.get_full_depth(),
			sliceSize = fullW * fullH;
		//xxxxxxxxxxxxxxxxxxx ,
		//	sliceSize = fullW * fullH,
		//	nVox = sliceSize * fullD;

		size_t vh = (size_t) (double(fullH) * aniso_y),
			vw = (size_t) (double(fullW) * aniso_x),
			vd = (size_t) (double(fullD) * aniso_z);

		// in the 3D case tiling is a formality, so fetch the only tile in the file
		if (! ilo.load_tile(0, 0))
		{
#ifdef WITH_PYTHON_H
			throw "Error loading volume data";
#endif	
			std::cerr << "Error loading volume data\n";
			return false;
		}

		// Get ahold of tile's pixel buffer
		const std::vector<uint32_t>& dataI = ilo.get_int_tile_buffer();

		// iterate virtual voxels
		size_t vSliceSize = vh * vw, 
			nVox = vh * vw * vd;
		for (size_t i = 0; i < nVox; i++)
		{
			// virtual voxel position
			int z = i / vSliceSize,
				y = (i - z * vSliceSize) / vw,
				x = (i - z * vSliceSize) % vw;

			// physical voxel position
			size_t ph_x = (size_t) (double(x) / aniso_x),
				ph_y = (size_t) (double(y) / aniso_y),
				ph_z = (size_t) (double(z) / aniso_z);
				i = ph_z * sliceSize + ph_y * fullH + ph_x;

			// Cache this pixel 
			feed_pixel_2_cache_3D_LR (x, y, z, dataI[i], vroi);

		}

		return true;
	}

	//
	// Reads pixels of whole slide 'intens_fpath' into virtual ROI 'vroi' 
	// performing anisotropy correction
	//
	bool scan_trivial_wholevolume_anisotropic__OLD(
		LR& vroi,
		const std::string& intens_fpath,
		ImageLoader& ldr,
		double aniso_x,
		double aniso_y)
	{
		int lvl = 0,	// Pyramid level
			lyr = 0;	//	Layer

		// physical slide properties
		size_t nth = ldr.get_num_tiles_hor(),
			ntv = ldr.get_num_tiles_vert(),
			fw = ldr.get_tile_width(),
			th = ldr.get_tile_height(),
			tw = ldr.get_tile_width(),
			tileSize = ldr.get_tile_size(),
			fullwidth = ldr.get_full_width(),
			fullheight = ldr.get_full_height();

		// virtual slide properties
		size_t vh = (size_t)(double(fullheight) * aniso_y),
			vw = (size_t)(double(fullwidth) * aniso_x),
			vth = (size_t)(double(th) * aniso_y),
			vtw = (size_t)(double(tw) * aniso_x);

		// current tile to skip tile reloads
		size_t curt_x = 999, curt_y = 999;

		for (size_t vr = 0; vr < vh; vr++)
		{
			for (size_t vc = 0; vc < vw; vc++)
			{
				// tile position for virtual pixel (vc, vr)
				size_t tidx_y = size_t(vr / vth),
					tidx_x = size_t(vc / vtw);

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

				// within-tile virtual pixel position
				size_t vx = vc - tidx_x * vtw,
					vy = vr - tidx_y * vth;

				// within-tile physical pixel position
				size_t ph_x = size_t(double(vx) / aniso_x),
					ph_y = size_t(double(vy) / aniso_y),
					i = ph_y * tw + ph_x;

				// read buffered physical pixel 
				const std::vector<uint32_t>& dataI = ldr.get_int_tile_buffer();
				const std::shared_ptr<std::vector<uint32_t>>& spL = ldr.get_seg_tile_sptr();
				bool wholeslide = spL == nullptr; // alternatively, theEnvironment.singleROI

				// Cache this pixel 
				feed_pixel_2_cache_LR(vc, vr, dataI[i], vroi);
			}
		}

		return true;
	}


	bool processTrivialRois_3D(const std::vector<int>& trivRoiLabels, const std::string& intens_fpath, const std::string& label_fpath, size_t memory_limit)
	{
		std::vector<int> Pending;
		size_t batchDemand = 0;
		int roiBatchNo = 1;

		for (auto lab : trivRoiLabels)
		{
			LR& r = roiData[lab];

			size_t itemFootprint = r.get_ram_footprint_estimate();

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
				VERBOSLVL2(std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of total " << uniqueLabels.size() << " ROIs\n";)
					VERBOSLVL2(
						if (Pending.size() == 1)
							std::cout << ">>> (single ROI label " << Pending[0] << ")\n";
						else
							std::cout << ">>> (ROI labels " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
				)

					if (theEnvironment.anisoOptions.customized() == false)
					{
						scanTrivialRois_3D(Pending, intens_fpath, label_fpath);
					}
					else
					{
						double	ax = theEnvironment.anisoOptions.get_aniso_x(),
							ay = theEnvironment.anisoOptions.get_aniso_y(),
							az = theEnvironment.anisoOptions.get_aniso_z();
						scanTrivialRois_3D_anisotropic(Pending, intens_fpath, label_fpath, ax, ay, az);
					}

				// Allocate memory
				VERBOSLVL2(std::cout << "\tallocating ROI buffers\n";)
					allocateTrivialRoisBuffers_3D(Pending);

				// Reduce them
				VERBOSLVL2(std::cout << "\treducing ROIs\n";)
					// reduce_trivial_rois(Pending);	
					reduce_trivial_rois_manual(Pending);

				// Free memory
				VERBOSLVL2(std::cout << "\tfreeing ROI buffers\n";)
					freeTrivialRoisBuffers_3D(Pending);	// frees what's allocated by feed_pixel_2_cache() and allocateTrivialRoisBuffers()

					// Reset the RAM footprint accumulator
				batchDemand = 0;

				// Clear the freshly processed ROIs from pending list 
				Pending.clear();

				// Start a new pending set by adding the stopper ROI 
				Pending.push_back(lab);

				// Advance the batch counter
				roiBatchNo++;
			}

			// Allow heyboard interrupt.
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
			// Read raw pixels of pending trivial ROIs 
			std::sort(Pending.begin(), Pending.end());
			VERBOSLVL2(std::cout << ">>> Scanning batch #" << roiBatchNo << " of " << Pending.size() << " pending ROIs of " << uniqueLabels.size() << " all ROIs\n";)
				VERBOSLVL2(
					if (Pending.size() == 1)
						std::cout << ">>> (single ROI " << Pending[0] << ")\n";
					else
						std::cout << ">>> (ROIs " << Pending[0] << " ... " << Pending[Pending.size() - 1] << ")\n";
			)
				if (theEnvironment.anisoOptions.customized() == false)
				{
					scanTrivialRois_3D(Pending, intens_fpath, label_fpath);
				}
				else
				{
					double	ax = theEnvironment.anisoOptions.get_aniso_x(),
						ay = theEnvironment.anisoOptions.get_aniso_y(),
						az = theEnvironment.anisoOptions.get_aniso_z();
					scanTrivialRois_3D_anisotropic(Pending, intens_fpath, label_fpath, ax, ay, az);
				}

			// Allocate memory
			VERBOSLVL2(std::cout << "\tallocating ROI buffers\n";)
				allocateTrivialRoisBuffers_3D(Pending);

			// Dump ROIs for use in unit testing
#ifdef DUMP_ALL_ROI
			dump_all_roi();
#endif

			// Reduce them
			VERBOSLVL2(std::cout << "\treducing ROIs\n";)
				//reduce_trivial_rois(Pending);	
				reduce_trivial_rois_manual(Pending);

			// Free memory
			VERBOSLVL2(std::cout << "\tfreeing ROI buffers\n";)
				freeTrivialRoisBuffers_3D(Pending);

#ifdef WITH_PYTHON_H
			// Allow heyboard interrupt.
			if (PyErr_CheckSignals() != 0)
			{
				sureprint("\nAborting per user input\n");
				throw pybind11::error_already_set();
			}
#endif
		}

		VERBOSLVL2(std::cout << "\treducing neighbor features and their depends for all ROIs\n")
			reduce_neighbors_and_dependencies_manual();

		return true;
	}

	void allocateTrivialRoisBuffers_3D(const std::vector<int>& roi_labels)
	{
		// Calculate the total memory demand (in # of items) of all segments' image matrices
		imageMatrixBufferLen = 0;
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			size_t w = r.aabb.get_width(),
				h = r.aabb.get_height(),
				d = r.aabb.get_z_depth(),
				cubeSize = w * h * d;
			imageMatrixBufferLen += cubeSize;

			largest_roi_imatr_buf_len = largest_roi_imatr_buf_len == 0 ? cubeSize : std::max(largest_roi_imatr_buf_len, cubeSize);
		}

		//
		// Preallocate image matrices and cubes here (in the future).
		//
		for (auto lab : roi_labels)
		{
			LR& r = roiData[lab];
			r.aux_image_cube.calculate_from_pixelcloud(r.raw_pixels_3D, r.aabb);
		}
	}

}