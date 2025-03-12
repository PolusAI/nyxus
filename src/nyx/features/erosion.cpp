#include <algorithm>
#include "erosion.h"

using namespace Nyxus;

namespace NyxusGpu 
{
	bool ErosionFeature_calculate_via_gpu(size_t roi_index, size_t roi_w, size_t roi_h, int max_n_erosions, int & fval);
}

ErosionPixelsFeature::ErosionPixelsFeature() : FeatureMethod("ErosionPixelsFeature")
{
	provide_features({ Feature2D::EROSIONS_2_VANISH, Feature2D::EROSIONS_2_VANISH_COMPLEMENT });
	add_dependencies({ Feature2D::CONVEX_HULL_AREA });	// Feature EROSIONS_2_VANISH_COMPLEMENT requires the convex hull
}

void ErosionPixelsFeature::calculate(LR& r)
{
	// Build the mask image matrix 'I2'
	auto width = r.aabb.get_width(),
		height = r.aabb.get_height(),
		minx = r.aabb.get_xmin(),
		miny = r.aabb.get_ymin();
	
	SimpleMatrix<PixIntens> I2((int)width, (int)height);

	for (auto px : r.raw_pixels)
	{
		auto x = px.x - minx,
			y = px.y - miny;
		//<--shape erosion, not greyscale!-- 
		//		I2.xy(x, y) = px.inten + 1;	// '+1' does the job: wherever intensity pixels are defined (via raw_pixels), in the MIM we have at least 1 even if the intensity is 0
		I2.xy(x, y) = 1;
	}

	// structural element's half-width and height
	auto halfHeight = (int)floor(SE_R / 2);
	auto halfWidth = (int)floor(SE_C / 2);

	// Initialize output image
	std::vector<PixIntens> Nv;
	Nv.reserve(SE_R*SE_C/2);	// Reserving the nnz(struc elem matrix), roughly equal to the 50% of the SE matrix size
	
	// Blank auxiliary image that we'll need in the loop
	SimpleMatrix<PixIntens> I1;

	for (numErosions = 0; numErosions < SANITY_MAX_NUM_EROSIONS; numErosions++)
	{
		// Copy the matrix from previous iteration
		I1 = I2;

		// Perform an erosion operation and count the number of surviving non-blank pixels
		int numNon0 = 0;

		// --Perform local min operation, which is morphological erosion
		for (int col = (halfWidth + 1); col < (width - halfWidth); col++)
			for (int row = (halfHeight + 1); row < (height - halfHeight); row++)
			{
				// Get the 3x3 (or in general, NxN) neighborhood
				int row1 = row - halfHeight;
				int row2 = row + halfHeight;
				int col1 = col - halfWidth;
				int col2 = col + halfWidth;

				bool all0 = true;
				int N[SE_R][SE_C];
				for (int r = row1; r <= row2; r++)
					for (int c = col1; c <= col2; c++)
					{
						auto pi = I1.xy(c,r);
						N[r - row1][c - col1] = pi;

						if (pi)
							all0 = false;
					}

				// Skip finding minimum if we have all-zeros
				if (all0)
				{
					I2.xy(col, row) = 0;
					continue;
				}

				// Apply the structuring element
				Nv.clear();
				for (int r = 0; r < SE_R; r++)
					for (int c = 0; c < SE_C; c++)
					{
						int s = strucElem[r][c];
						if (s)
							Nv.push_back(N[r][c]);
					}

				PixIntens minPixel = *std::min_element(Nv.begin(), Nv.end());
				I2.xy(col, row) = minPixel;

				// Count non-0 pixels
				if (minPixel > 0)
					numNon0++;
			}

		// Any remaining nonzero pixels?
		if (numNon0 == 0)
			break;
	}
}

void ErosionPixelsFeature::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {} // Not supporting online for erosions

void ErosionPixelsFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	// Build the mask image matrix 'I2'
	auto width = r.aabb.get_width(),
		height = r.aabb.get_height(),
		minx = r.aabb.get_xmin(),
		miny = r.aabb.get_ymin();

	WriteImageMatrix_nontriv I2("I2", r.label);
	I2.allocate_from_cloud (r.raw_pixels_NT, r.aabb, true);

	auto halfHeight = (int)floor(SE_R / 2);
	auto halfWidth = (int)floor(SE_C / 2);

	// Initialize output image
	std::vector<PixIntens> Nv;
	Nv.reserve(SE_R * SE_C / 2);	// Reserving the nnz(struc elem matrix), roughly equal to the 50% of the SE matrix size

	// Blank auxiliary image that we'll need to implement a chain of erosions
	WriteImageMatrix_nontriv I1("I1", r.label);
	I1.allocate(width, height, 0);

	for (numErosions = 0; numErosions < SANITY_MAX_NUM_EROSIONS; numErosions++)
	{
		// Copy the matrix from previous iteration
		I1.copy(I2);

		// Perform an erosion operation and count the number of surviving non-blank pixels 
		int numNon0 = 0;

		// --Perform local min operation, which is morphological erosion
		for (int col = (halfWidth + 1); col < (width - halfWidth); col++)
			for (int row = (halfHeight + 1); row < (height - halfHeight); row++)
			{
				// Get the 3x3 (or in general, NxN) neighborhood
				int row1 = row - halfHeight;
				int row2 = row + halfHeight;
				int col1 = col - halfWidth;
				int col2 = col + halfWidth;

				bool all0 = true;
				int N[SE_R][SE_C];
				for (int r = row1; r <= row2; r++)
					for (int c = col1; c <= col2; c++)
					{
						auto pi = I1[r * width + c];
						N[r - row1][c - col1] = pi;

						if (pi)
							all0 = false;
					}

				// Skip finding minimum if we have all-zeros
				if (all0)
				{
					I2.set_at(row * width + col, 0);
					continue;
				}

				// Apply the structuring element
				Nv.clear();
				for (int r = 0; r < SE_R; r++)
					for (int c = 0; c < SE_C; c++)
					{
						int s = strucElem[r][c];
						if (s)
							Nv.push_back(N[r][c]);
					}

				PixIntens minPixel = *std::min_element(Nv.begin(), Nv.end());
				I2.set_at(row * width + col, minPixel);

				// Count non-0 pixels
				if (minPixel > 0)
					numNon0++;
			}

		// Any remaining nonzero pixels?
		if (numNon0 == 0)
			break;
	}
}

void ErosionPixelsFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::EROSIONS_2_VANISH][0] = numErosions;
}

void ErosionPixelsFeature::extract (LR& r)
{		
	ErosionPixelsFeature epix;
	epix.calculate(r);
	epix.save_value(r.fvals);
}

void ErosionPixelsFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		// Check if data is good
		if (r.aux_min == r.aux_max)
			continue;

		// Calculate feature
		extract (r);
	}
}

#ifdef USE_GPU

void ErosionPixelsFeature::gpu_process_all_rois (
	const std::vector<int>& L, 
	std::unordered_map <int, LR>& RoiData,
	size_t batch_offset,
	size_t batch_len)
{
	for (size_t i = 0; i < batch_len; i++)
	{
		size_t far_i = i + batch_offset;
		auto lab = L[far_i];
		LR& r = RoiData[lab];

		ErosionPixelsFeature f;
		f.calculate_via_gpu (r, i, 1000/*SANITY_MAX_NUM_EROSIONS*/, f.numErosions);
		f.save_value(r.fvals);
	}
}

void ErosionPixelsFeature::calculate_via_gpu (LR& r, size_t roiidx, int max_n_erosions, int & fval)
{
	bool ok = NyxusGpu::ErosionFeature_calculate_via_gpu (roiidx, r.aabb.get_width(), r.aabb.get_height(), max_n_erosions, fval);
	if (!ok)
		std::cerr << "ErosionFeature: error calculating on GPU\n";
}


#endif // USE_GPU