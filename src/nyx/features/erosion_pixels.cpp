#include <algorithm>
#include "erosion.h"

ErosionPixelsFeature::ErosionPixelsFeature() : FeatureMethod("ErosionPixelsFeature")
{
	provide_features({ EROSIONS_2_VANISH, EROSIONS_2_VANISH_COMPLEMENT });
	add_dependencies({ PERIMETER });
}

void ErosionPixelsFeature::calculate(LR& r)
{
	auto& I = r.aux_image_matrix;

	//[rows, columns, numberOfColorChannels] = size(grayImage);
	int rows = I.height,
		cols = I.width;

	//% Define structuring element.
	//se = logical([0 1 0; 1 1 1; 0 1 0]);
	//[p, q] = size(se);

	auto halfHeight = (int)floor(SE_R / 2);
	auto halfWidth = (int)floor(SE_C / 2);

	// Initialize output image
	ImageMatrix I1, I2(I);

	// Discretize
	#if 0
		writeablePixels I2wp = I2.WriteablePixels();
		for (size_t i = 0; i < I2wp.size(); i++)
		{
			PixIntens& pi = I2wp[i];
			if (pi != 0)
				pi = 1;
		}
	#endif

	std::vector<PixIntens> Nv;
	Nv.reserve(SE_R*SE_C/2);	// Reserving the nnz(struc elem matrix), roughly equal to the 50% of the SE matrix size
	
	numErosions = 0;
	for (; numErosions < SANITY_MAX_NUM_EROSIONS; numErosions++)
	{
		// Copy the matrix from previous iteration
		I1 = I2;
		writeablePixels I2p = I2.WriteablePixels();
		pixData I1p = I1.ReadablePixels();

		int numNon0 = 0;

		// Perform local min operation, which is morphological erosion.
		for (int col = (halfWidth + 1); col < (cols - halfWidth); col++)
			for (int row = (halfHeight + 1); row < (rows - halfHeight); row++)
			{
				//% Get the 3x3 (or in general, NxN) neighborhood
				int row1 = row - halfHeight;
				int row2 = row + halfHeight;
				int col1 = col - halfWidth;
				int col2 = col + halfWidth;

				bool all0 = true;
				int N[SE_R][SE_C];
				for (int r = row1; r <= row2; r++)
					for (int c = col1; c <= col2; c++)
					{
						auto pi = I1p.yx(r, c);
						N[r - row1][c - col1] = pi;

						if (pi)
							all0 = false;
					}

				// Skip finding minimum if we have all-zeros
				if (all0)
				{
					I2p.yx(row, col) = 0;
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
				I2p.yx(row, col) = minPixel;

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
	//
	// ImageLoader& imlo, ReadImageMatrix_nontriv& I
	// 
	// WriteImageMatrix_nontriv& W
	//
	//

	//-- auto& I = r.aux_image_matrix;
	ReadImageMatrix_nontriv I(r.aabb);

	//[rows, columns, numberOfColorChannels] = size(grayImage);
	int rows = r.aabb.get_height(),
		cols = r.aabb.get_width();

	//% Define structuring element.
	//se = logical([0 1 0; 1 1 1; 0 1 0]);
	//[p, q] = size(se);

	auto halfHeight = (int) floor(SE_R / 2);
	auto halfWidth = (int) floor(SE_C / 2);

	//% Initialize output image
	//localMinImage = zeros(size(grayImage), class(grayImage));
	//-- ImageMatrix I1, I2(I);
	WriteImageMatrix_nontriv I1 ("ErosionPixelsFeature::osized_calculate_I1", r.label), 
		I2 ("ErosionPixelsFeature::osized_calculate_I2", r.label);
	I2.init_with_cloud(r.osized_pixel_cloud, r.aabb);

	numErosions = 0;
	for (; numErosions < SANITY_MAX_NUM_EROSIONS; numErosions++)
	{
		// Copy the matrix from previous iteration
		//-- I1 = I2;
		I1.copy (I2);

		//-- writeablePixels I2p = I2.WriteablePixels();
		//-- pixData I1p = I1.ReadablePixels();

		int numNon0 = 0;

		// Perform local min operation, which is morphological erosion.
		for (int col = (halfWidth + 1); col < (cols - halfWidth); col++)
			for (int row = (halfHeight + 1); row < (rows - halfHeight); row++)
			{
				//% Get the 3x3 neighborhood
				int row1 = row - halfHeight;
				int row2 = row + halfHeight;
				int col1 = col - halfWidth;
				int col2 = col + halfWidth;

				//thisNeighborhood = grayImage (row1:row2, col1 : col2);
				bool all0 = true;
				int N[SE_R][SE_C];
				for (int r = row1; r <= row2; r++)
					for (int c = col1; c <= col2; c++)
					{
						N[r - row1][c - col1] = I1.get_at (r, c);

						if (N[r - row1][c - col1])
							all0 = false;
					}

				// Skip finding minimum if we have all-zeros
				if (all0)
				{
					I2.set_at (row, col, 0);
					continue;
				}

				// Apply the structuring element
				//pixelsInSE = thisNeighborhood(se);
				std::vector<PixIntens> Nv;
				for (int r = 0; r < SE_R; r++)
					for (int c = 0; c < SE_C; c++)
					{
						int s = strucElem[r][c];
						if (s)
							Nv.push_back(N[r][c]);
					}

				//localMinImage(row, col) = min(pixelsInSE);
				PixIntens minPixel = *std::min_element(Nv.begin(), Nv.end());
				I2.set_at (row, col, minPixel);

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
	fvals[EROSIONS_2_VANISH][0] = numErosions;
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
		if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
			continue;

		// Calculate feature
		ErosionPixelsFeature epix;
		epix.calculate(r);
		epix.save_value(r.fvals);
	}
}

