#include <algorithm>
#include "erosion.h"

int ErosionPixels::calc_feature (ImageMatrix & I)
{
	//[rows, columns, numberOfColorChannels] = size(grayImage);
	int rows = I.height,
		cols = I.width;

	//% Define structuring element.
	//se = logical([0 1 0; 1 1 1; 0 1 0]);
	//[p, q] = size(se);

	auto halfHeight = (int) floor(SE_R / 2);
	auto halfWidth = (int) floor(SE_C / 2);
	
	//% Initialize output image
	//localMinImage = zeros(size(grayImage), class(grayImage));
	ImageMatrix I1, I2(I);

	int numErosions = 0;	// The feature value
	for (; numErosions < SANITY_MAX_NUM_EROSIONS; numErosions++)
	{
		// Copy the matrix from previous iteration
		I1 = I2;
		writeablePixels I2p = I2.WriteablePixels();
		pixData I1p = I1.ReadablePixels();

		int numNon0 = 0;

		//% Perform local min operation, which is morphological erosion.
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
						N[r-row1][c-col1] = I1p(r, c);

						if (N[r-row1][c-col1])
							all0 = false;
					}

				// Skip finding minimum if we have all-zeros
				if (all0)
				{
					I2p(row, col) = 0;
					continue;
				}

				//% Apply the structuring element
				//pixelsInSE = thisNeighborhood(se);
				std::vector<PixIntens> Nv;
				for (int r = 0; r < SE_R; r++)
					for (int c = 0; c < SE_C; c++)
					{
						int s = strucElem[r][c];
						if (s)
							Nv.push_back (N[r][c]);
					}

				//localMinImage(row, col) = min(pixelsInSE);
				PixIntens minPixel = *std::min_element(Nv.begin(), Nv.end());
				I2p(row, col) = minPixel;

				// Count non-0 pixels
				if (minPixel > 0)
					numNon0++;
			}

		// Any remaining nonzero pixels?
		if (numNon0 == 0)
			break;
	}

	return numErosions;
}

void ErosionPixels::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Skip calculation in case of bad data
		if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
			continue;

		// Calculate feature
		//---	ImageMatrix im(r.raw_pixels, r.aabb);
		ErosionPixels epix;
		r.fvals[EROSIONS_2_VANISH][0] = epix.calc_feature(r.aux_image_matrix);
	}
}

