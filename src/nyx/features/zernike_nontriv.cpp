
/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*                              mb_Znl.c                                    */
/*                                                                          */
/*                                                                          */
/*                           Michael Boland                                 */
/*                            09 Dec 1998                                   */
/*                                                                          */
/*  Revisions:                                                              */
/*  9-1-04 Tom Macura <tmacura@nih.gov> modified to make the code ANSI C    */
/*         and work with included complex arithmetic library from           */
/*         Numerical Recepies in C instead of using the system's C++ STL    */
/*         Libraries.                                                       */
/*                                                                          */
/*  1-29-06 Lior Shamir <shamirl (-at-) mail.nih.gov> modified "factorial"  */
/*  to a loop, replaced input structure with ImageMatrix class.             */
/*  2011-04-25 Ilya Goldberg. Optimization due to this function accounting  */
/*    for 35% of the total wndchrm run-time.  Now ~4x faster.               */
/*  2012-12-13 Ilya Goldberg. Added 10x faster mb_zernike2D_2               */
/*    the feature values this algorithm produces are not the same as before */
/*    however, the weights assigned to these features in classification     */
/*    are as good or better than mb_zernike2D                               */
/****************************************************************************/

#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <complex>
#include <cmath>
#include <cfloat> // Has definition of DBL_EPSILON
#include <assert.h>
#include <stdio.h>
#include "specfunc.h" 

#include "image_matrix.h" 

#include <unordered_map>
#include "../roi_cache.h"
#include "zernike.h"

void ZernikeFeature::osized_calculate(LR& r, ImageLoader& imloader)
{
	OOR_ReadMatrix I (imloader, r.aabb);

	zernike2D_nontriv (I, ZernikeFeature::ZERNIKE2D_ORDER);

	// Fix calculated feature values due to all-0 intensity labels to avoid NANs in the output
	if (r.has_bad_data())
	{
		for (int i = 0; i < coeffs.size(); i++)
			coeffs[i] = 0.0;
	}
}

void ZernikeFeature::zernike2D_nontriv (OOR_ReadMatrix& I, int order)
{
	coeffs.resize (ZernikeFeature::NUM_FEATURE_VALS, 0);

	// Calculate features
	long output_size;   // output size is normally 72 (NUM_FEATURE_VALS)
	mb_zernike2D_nontriv (I, order, 0/*rad*/, coeffs.data(), &output_size);
}

/*
  Algorithms for fast computation of Zernike moments and their numerical stability
  Chandan Singh and Ekta Walia, Image and Vision Computing 29 (2011) 251–259

  Implemented from pseudo-code by Ilya Goldberg 2011-04-27
  This code is 10x faster than the previous code, and 50x faster than previous unoptimized code.
  200 x 200 image 2 GHz CoreDuo (ms):
  3813 previous unoptimized
   877 previous optimized
	75 this implementation
  The values of the returned features are different, but their Fischer scores are slightly
  better on average than the previous version, and they produce better classification in problems
  where zernike features are useful.
*/


void ZernikeFeature::mb_zernike2D_nontriv (OOR_ReadMatrix& I, double order, double rad, double* zvalues, long* output_size)
{
	int cols = I.get_width();
	int rows = I.get_height();

	int L, N, D;

	// N is the smaller of Im.width and Im.height
	N = cols < rows ? cols : rows;
	//--alternatively-- N = Im.width > Im.height ? Im.width : Im.height; //MM: This change is needed for bounding box implementations to ensure disk is covering the entire area of the Image

	if (order > 0)
		L = (int)order;
	else
		L = 15;
	assert(L < MAX_L);

	if (!(rad > 0.0))
		rad = N;
	D = (int)(rad * 2);

	static double H1[MAX_L][MAX_L];
	static double H2[MAX_L][MAX_L];
	static double H3[MAX_L][MAX_L];
	static char init = 1;

	double COST[MAX_L], SINT[MAX_L], R[MAX_L];
	double Rn, Rnm, Rnm2, Rnnm2, Rnmp2, Rnmp4;

	double a, b, x, y, area, r, r2, f, const_t;
	int n, m, i, j;

	double AR[MAX_L][MAX_L], AI[MAX_L][MAX_L];

	double sum = 0;

	// compute x/0, y/0 and 0/0 moments to center the unit circle on the centroid
	double moment10 = 0.0, moment00 = 0.0, moment01 = 0.0;
	double intensity;
	for (i = 0; i < cols; i++)
		for (j = 0; j < rows; j++)
		{
			if (std::isnan(I.get_at(j, i)))
				continue; //MM
			intensity = I.get_at(j, i);
			sum += intensity;
			moment10 += (i + 1) * intensity;
			moment00 += intensity;
			moment01 += (j + 1) * intensity;
		}
	double m10_m00 = moment10 / moment00;
	double m01_m00 = moment01 / moment00;

	// Pre-initialization of statics
	if (init)
	{
		for (n = 0; n < MAX_L; n++)
		{
			for (m = 0; m <= n; m++)
			{
				if (n != m)
				{
					H3[n][m] = -(double)(4.0 * (m + 2.0) * (m + 1.0)) / (double)((n + m + 2.0) * (n - m));
					H2[n][m] = ((double)(H3[n][m] * (n + m + 4.0) * (n - m - 2.0)) / (double)(4.0 * (m + 3.0))) + (m + 2.0);
					H1[n][m] = ((double)((m + 4.0) * (m + 3.0)) / 2.0) - ((m + 4.0) * H2[n][m]) + ((double)(H3[n][m] * (n + m + 6.0) * (n - m - 4.0)) / 8.0);
				}
			}
		}
		init = 0;
	}

	// Zero-out the Zernike moment accumulators
	for (n = 0; n <= L; n++)
	{
		for (m = 0; m <= n; m++)
		{
			AR[n][m] = AI[n][m] = 0.0;
		}
	}

	area = M_PI * rad * rad;
	for (i = 0; i < cols; i++)
	{
		// In the paper, the center of the unit circle was the center of the image
		//	x = (double)(2*i+1-N)/(double)D;
		x = (i + 1 - m10_m00) / rad;
		for (j = 0; j < rows; j++)
		{
			if (std::isnan(I.get_at(j, i)))
				continue; //MM

		// In the paper, the center of the unit circle was the center of the image
		//	y = (double)(2*j+1-N)/(double)D;
			y = (j + 1 - m01_m00) / rad;
			r2 = x * x + y * y;
			r = sqrt(r2);
			if (r < DBL_EPSILON || r > 1.0)
				continue;
			/*compute all powers of r and save in a table */
			R[0] = 1;
			for (n = 1; n <= L; n++)
				R[n] = r * R[n - 1];
			/* compute COST SINT and save in tables */
			a = COST[0] = x / r;
			b = SINT[0] = y / r;
			for (m = 1; m <= L; m++)
			{
				COST[m] = a * COST[m - 1] - b * SINT[m - 1];
				SINT[m] = a * SINT[m - 1] + b * COST[m - 1];
			}

			// compute contribution to Zernike moments for all 
			// orders and repetitions by the pixel at (i,j)
			// In the paper, the intensity was the raw image intensity
			f = I.get_at(j, i) / sum;

			Rnmp2 = Rnm2 = 0;
			for (n = 0; n <= L; n++)
			{
				// In the paper, this was divided by the area in pixels
				// seemed that pi was supposed to be the area of a unit circle.
				const_t = (n + 1) * f / M_PI;
				Rn = R[n];
				if (n >= 2)
					Rnm2 = R[n - 2];
				for (m = n; m >= 0; m -= 2)
				{
					if (m == n)
					{
						Rnm = Rn;
						Rnmp4 = Rn;
					}
					else
						if (m == n - 2)
						{
							Rnnm2 = n * Rn - (n - 1) * Rnm2;
							Rnm = Rnnm2;
							Rnmp2 = Rnnm2;
						}
						else
						{
							Rnm = H1[n][m] * Rnmp4 + (H2[n][m] + (H3[n][m] / r2)) * Rnmp2;
							Rnmp4 = Rnmp2;
							Rnmp2 = Rnm;
						}
					AR[n][m] += const_t * Rnm * COST[m];
					AI[n][m] -= const_t * Rnm * SINT[m];
				}
			}
		}
	}

	int numZ = 0;
	for (n = 0; n <= L; n++)
	{
		for (m = 0; m <= n; m++)
		{
			if ((n - m) % 2 == 0)
			{
				AR[n][m] *= AR[n][m];
				AI[n][m] *= AI[n][m];
				zvalues[numZ] = fabs(sqrt(AR[n][m] + AI[n][m]));
				numZ++;
			}
		}
	}
	*output_size = numZ;
}



