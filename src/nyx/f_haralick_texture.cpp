#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "sensemaker.h"
#include "image_matrix.h"
#include "environment.h"

#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif

typedef struct {
	double ASM;          /*  (1) Angular Second Moment */
	double contrast;     /*  (2) Contrast */
	double correlation;  /*  (3) Correlation */
	double variance;     /*  (4) Variance */
	double IDM;		    /*  (5) Inverse Diffenence Moment */
	double sum_avg;	    /*  (6) Sum Average */
	double sum_var;	    /*  (7) Sum Variance */
	double sum_entropy;	/*  (8) Sum Entropy */
	double entropy;	    /*  (9) Entropy */
	double diff_var;	    /* (10) Difference Variance */
	double diff_entropy;	/* (11) Diffenence Entropy */
	double meas_corr1;	/* (12) Measure of Correlation 1 */
	double meas_corr2;	/* (13) Measure of Correlation 2 */
	double max_corr_coef; /* (14) Maximal Correlation Coefficient */
} TEXTURE;

double f14_maxcorr(double** P, int Ng);

// allocate_matrix() is depends for CoOcMat_Angle_KKK()
/* Allocates a double matrix with range [nrl..nrh][ncl..nch] */
double** allocate_matrix(int nrl, int nrh, int ncl, int nch)
{
	int i;
	double** m;

	/* allocate pointers to rows */
	m = (double**)malloc((unsigned)(nrh - nrl + 1) * sizeof(double*));
	if (!m) fprintf(stderr, "memory allocation failure (allocate_matrix 1) "), exit(1);
	m -= ncl;

	/* allocate rows and set pointers to them */
	for (i = nrl; i <= nrh; i++) {
		m[i] = (double*)malloc((unsigned)(nch - ncl + 1) * sizeof(double));
		if (!m[i]) fprintf(stderr, "memory allocation failure (allocate_matrix 2) "), exit(2);
		m[i] -= ncl;
	}

	/* return pointer to array of pointers to rows */
	return m;
}

// Depends:
using u_int8_t = unsigned char;
//

/* Compute gray-tone spatial dependence matrix */
void CoOcMat_Angle_0 (SimpleMatrix<double> & matrix, int distance, u_int8_t** grays,
	int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; /* normalizing factor */

	//A		double** matrix = allocate_matrix(0, tone_count, 0, tone_count);

	/* zero out matrix */
	/*/
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix(itone,jtone) = 0;	//A		matrix[itone][jtone] = 0;
	*/
	matrix.fill(0.0);

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			if (grays[row][col] == 0)
				continue;

			/* find x tone */
			if (col + d < cols && grays[row][col + d]) {
				x = tone_LUT[grays[row][col]];
				y = tone_LUT[grays[row][col + d]];
				matrix(x,y)++;		//A		matrix[x][y]++;
				matrix(y,x)++;		//A		matrix[y][x]++;
				count += 2;
			}
		}

	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0)   /* protect from error */
				matrix(itone,jtone) = 0;	//A		matrix[itone][jtone] = 0;
			else 
				matrix(itone,jtone) /= count;	//A		matrix[itone][jtone] /= count;

	//A		return matrix;
}

void CoOcMat_Angle_90 (SimpleMatrix<double>& matrix, int distance, u_int8_t** grays,
	int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; /* normalizing factor */

	//A		double** matrix = allocate_matrix(0, tone_count, 0, tone_count);

	/* zero out matrix */
	/*
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix(itone, jtone) = 0;	//A		matrix[itone][jtone] = 0;
			*/
	matrix.fill(0.0);

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			if (grays[row][col] == 0)
				continue;

			/* find x tone */
			if (row + d < rows && grays[row + d][col]) {
				x = tone_LUT[grays[row][col]];
				y = tone_LUT[grays[row + d][col]];
				matrix(x,y)++;		//A		matrix[x][y]++;
				matrix(y,x)++;		//A		matrix[y][x]++;
				count += 2;
			}
		}

	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0) 
				matrix(itone, jtone) = 0;	//A		matrix[itone][jtone] = 0;
			else 
				matrix(itone, jtone) /= count;	//A		matrix[itone][jtone] /= count;

	//A		return matrix;
}

void CoOcMat_Angle_45 (SimpleMatrix<double>& matrix, int distance, u_int8_t** grays,
	int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; /* normalizing factor */

	//A		double** matrix = allocate_matrix(0, tone_count, 0, tone_count);

	/* zero out matrix */
	/*
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix(itone, jtone) = 0;	//A		matrix[itone][jtone] = 0;
	*/
	matrix.fill(0.0);

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			if (grays[row][col] == 0)
				continue;

			/* find x tone */
			if (row + d < rows && col - d >= 0 && grays[row + d][col - d]) {
				x = tone_LUT[grays[row][col]];
				y = tone_LUT[grays[row + d][col - d]];
				matrix(x,y)++;		//A		matrix[x][y]++;
				matrix(y,x)++;		//A		matrix[y][x]++;
				count += 2;
			}
		}

	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0) 
				matrix(itone, jtone) = 0;	//A		matrix[itone][jtone] = 0;       /* protect from error */
			else 
				matrix(itone, jtone) /= count;	//A		matrix[itone][jtone] /= count;

	//A		return matrix;
}

void CoOcMat_Angle_135 (SimpleMatrix<double>& matrix, int distance, u_int8_t** grays,
	int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; /* normalizing factor */

	//A		double** matrix = allocate_matrix(0, tone_count, 0, tone_count);

	/* zero out matrix */
	/*
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix(itone, jtone) = 0;	//A		matrix[itone][jtone] = 0;
	*/
	matrix.fill(0.0);

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			if (grays[row][col] == 0)
				continue;

			/* find x tone */
			if (row + d < rows && col + d < cols && grays[row + d][col + d]) {
				x = tone_LUT[grays[row][col]];
				y = tone_LUT[grays[row + d][col + d]];
				matrix(x,y)++;	//A		matrix[x][y]++;
				matrix(y,x)++;	//A		matrix[y][x]++;
				count += 2;
			}
		}

	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0) 
				matrix(itone, jtone) = 0;	//A		matrix[itone][jtone] = 0;   /* protect from error */
			else 
				matrix(itone, jtone) /= count;	//A		matrix[itone][jtone] /= count;

	//A		return matrix;
}

/* Angular Second Moment
*
* The angular second-moment feature (ASM) f1 is a measure of homogeneity
* of the image. In a homogeneous image, there are very few dominant
* gray-tone transitions. Hence the P matrix for such an image will have
* fewer entries of large magnitude.
*/
double f1_asm (const SimpleMatrix<double>& P, int Ng) 
{
	int i, j;
	double sum = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			sum += P(i,j) * P(i,j);

	return sum;
}

/* Contrast
*
* The contrast feature is a difference moment of the P matrix and is a
* measure of the contrast or the amount of local variations present in an
* image.
*/
double f2_contrast(SimpleMatrix<double>& P, int Ng) 
{
	//=== W-C:
	//int i, j, n;
	//double sum = 0, bigsum = 0;
	//
	//for (n = 0; n < Ng; ++n) 
	//{
	//	for (i = 0; i < Ng; ++i)
	//		for (j = 0; j < Ng; ++j) 
	//		{
	//			if ((i - j) == n || (j - i) == n)
	//				sum += P(i,j);
	//		}
	//	bigsum += n * n * sum;
	//	sum = 0;
	//}
	//
	//return bigsum;
	//

	double sum = 0;
	for (int i = 0; i < Ng; i++)
		for (int j = 0; j < Ng; j++)
			sum += P(i,j) * (j-i) * (i-j);
	return sum;
}

double* allocate_vector(int nl, int nh) {
	double* v;

	v = (double*)calloc(1, (unsigned)(nh - nl + 1) * sizeof(double));
	if (!v) fprintf(stderr, "memory allocation failure (allocate_vector) "), exit(1);

	return v - nl;
}

/* Correlation
*
* This correlation feature is a measure of gray-tone linear-dependencies
* in the image.
*/
double f3_corr(SimpleMatrix<double>& P, int Ng)
{
	int i, j;
	double sum_sqrx = 0, tmp, * px;
	double meanx = 0, meany = 0, stddevx, stddevy;

	px = allocate_vector(0, Ng);
	for (i = 0; i < Ng; ++i)
		px[i] = 0;

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			px[i] += P(i,j);


	/* Now calculate the means and standard deviations of px and py */
	/*- fix supplied by J. Michael Christensen, 21 Jun 1991 */
	/*- further modified by James Darrell McCauley, 16 Aug 1991
	*     after realizing that meanx=meany and stddevx=stddevy
	*/
	for (i = 0; i < Ng; ++i) {
		meanx += px[i] * i;
		sum_sqrx += px[i] * i * i;
	}

	/* M. Boland meanx = meanx/(sqrt(Ng)); */
	meany = meanx;
	stddevx = sqrt(sum_sqrx - (meanx * meanx));
	stddevy = stddevx;

	/* Finally, the correlation ... */
	for (tmp = 0, i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			tmp += i * j * P(i,j);

	free(px);
	if (stddevx * stddevy == 0) return(1);  /* protect from error */
	else return (tmp - meanx * meany) / (stddevx * stddevy);
}

/* Sum of Squares: Variance */
double f4_var(SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double mean = 0, var = 0;

	/*- Corrected by James Darrell McCauley, 16 Aug 1991
	*  calculates the mean intensity level instead of the mean of
	*  cooccurrence matrix elements
	*/
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			mean += i * P(i,j);

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/*  M. Boland - var += (i + 1 - mean) * (i + 1 - mean) * P[i][j]; */
			var += (i - mean) * (i - mean) * P(i,j);

	return var;
}

/* Inverse Difference Moment */
double f5_idm(SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double idm = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			idm += P(i,j) / (1 + (i - j) * (i - j));

	return idm;
}

/* Sum Average */
double f6_savg(SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double savg = 0;
	double* Pxpy = allocate_vector(0, 2 * Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/* M. Boland Pxpy[i + j + 2] += P[i][j]; */
			/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
			Pxpy[i + j] += P(i,j);

	/* M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		savg += i * Pxpy[i];

	free(Pxpy);
	return savg;
}

/* Sum Variance */
double f7_svar(SimpleMatrix<double>& P, int Ng, double S) {
	int i, j;
	double var = 0;
	double* Pxpy = allocate_vector(0, 2 * Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/* M. Boland Pxpy[i + j + 2] += P[i][j]; */
			/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
			Pxpy[i + j] += P(i,j);

	/*  M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		var += (i - S) * (i - S) * Pxpy[i];

	free(Pxpy);
	return var;
}

#define EPSILON 0.000000001

/* Sum Entropy */
double f8_sentropy(SimpleMatrix<double>& P, int Ng)
{
	int i, j;
	double sentropy = 0;
	double* Pxpy = allocate_vector(0, 2 * Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			Pxpy[i + j + 2] += P(i,j);

	for (i = 2; i <= 2 * Ng; ++i)
		/*  M. Boland  sentropy -= Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sentropy -= Pxpy[i] * log10(Pxpy[i] + EPSILON) / log10(2.0);

	free(Pxpy);
	return sentropy;
}

/* Entropy */
double f9_entropy(SimpleMatrix<double>& P, int Ng) 
{
	int i, j;
	double entropy = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/*      entropy += P[i][j] * log10 (P[i][j] + EPSILON); */
			entropy += P(i,j) * log10(P(i,j) + EPSILON) / log10(2.0);

	return -entropy;
}

/* Difference Variance */
double f10_dvar(SimpleMatrix<double>& P, int Ng)
{
	int i, j;
	double sum = 0, sum_sqr = 0, var = 0;
	double* Pxpy = allocate_vector(0, 2 * Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			Pxpy[abs(i - j)] += P(i,j);

	/* Now calculate the variance of Pxpy (Px-y) */
	for (i = 0; i < Ng; ++i) {
		sum += i * Pxpy[i];
		sum_sqr += i * i * Pxpy[i];
		/* M. Boland sum += Pxpy[i];
		sum_sqr += Pxpy[i] * Pxpy[i];*/
	}

	/*tmp = Ng * Ng ;  M. Boland - wrong anyway, should be Ng */
	/*var = ((tmp * sum_sqr) - (sum * sum)) / (tmp * tmp); */

	var = sum_sqr - sum * sum;

	free(Pxpy);
	return var;
}

/* Difference Entropy */
double f11_dentropy(SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double sum = 0;
	double* Pxpy = allocate_vector(0, 2 * Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			Pxpy[abs(i - j)] += P(i,j);

	for (i = 0; i < Ng; ++i)
		/*    sum += Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sum += Pxpy[i] * log10(Pxpy[i] + EPSILON) / log10(2.0);

	free(Pxpy);
	return -sum;
}

/* Information Measures of Correlation */
double f12_icorr(SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double* px, * py;
	double hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;

	px = allocate_vector(0, Ng);
	py = allocate_vector(0, Ng);
	/* All /log10(2.0) added by M. Boland */

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			px[i] += P(i,j);
			py[j] += P(i,j);
		}
	}

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j) {
			hxy1 -= P(i,j) * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy2 -= px[i] * py[j] * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy -= P(i,j) * log10(P(i,j) + EPSILON) / log10(2.0);
		}

	/* Calculate entropies of px and py - is this right? */
	for (i = 0; i < Ng; ++i) {
		hx -= px[i] * log10(px[i] + EPSILON) / log10(2.0);
		hy -= py[i] * log10(py[i] + EPSILON) / log10(2.0);
	}

	free(px);
	free(py);
	if ((hx > hy ? hx : hy) == 0) return(1);
	else
		return ((hxy - hxy1) / (hx > hy ? hx : hy));
}

/* Information Measures of Correlation */
double f13_icorr(SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double* px, * py;
	double hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;

	px = allocate_vector(0, Ng);
	py = allocate_vector(0, Ng);

	/* All /log10(2.0) added by M. Boland */

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			px[i] += P(i,j);
			py[j] += P(i,j);
		}
	}

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j) 
		{
			hxy1 -= P(i,j) * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy2 -= px[i] * py[j] * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy -= P(i,j) * log10(P(i,j) + EPSILON) / log10(2.0);
		}

	/* Calculate entropies of px and py */
	for (i = 0; i < Ng; ++i) {
		hx -= px[i] * log10(px[i] + EPSILON) / log10(2.0);
		hy -= py[i] * log10(py[i] + EPSILON) / log10(2.0);
	}

	free(px);
	free(py);
	return (sqrt(fabs(1 - exp(-2.0 * (hxy2 - hxy)))));
}

/* free matrix */
void free_matrix(double** matrix, int nrh)
{
	int col_index;
	for (col_index = 0; col_index <= nrh; col_index++)
		free(matrix[col_index]);
	free(matrix);
}

// Depends:
#define PGM_MAXMAXVAL 255
// 
void Extract_Texture_Features(int distance, int angle,
	u_int8_t** grays, unsigned int nrows, unsigned int ncols, 
	// output
	TEXTURE & Texture)
{
	int tone_LUT[PGM_MAXMAXVAL + 1]; /* LUT mapping gray tone(0-255) to matrix indicies */
	int tone_count = 0; /* number of tones actually in the img. atleast 1 less than 255 */
	int itone;
	int row, col, rows = nrows, cols = ncols;
	//A		double** P_matrix;
	double sum_entropy;

	//xxx	TEXTURE* Texture;
	//xxx	Texture = (TEXTURE*)calloc(1, sizeof(TEXTURE));
	//xxx	if (!Texture) {
	//xxx		printf("\nERROR in TEXTURE structure allocate\n");
	//xxx		exit(1);
	//xxx	}

	/* Determine the number of different gray tones (not maxval) */
	for (row = PGM_MAXMAXVAL; row >= 0; --row)
		tone_LUT[row] = -1;
	for (row = rows - 1; row >= 0; --row)
		for (col = 0; col < cols; ++col)
			tone_LUT[grays[row][col]] = grays[row][col];

	for (row = PGM_MAXMAXVAL, tone_count = 0; row >= 0; --row)
		if (tone_LUT[row] != -1)
			tone_count++;

	/* Use the number of different tones to build LUT */
	for (row = 0, itone = 0; row <= PGM_MAXMAXVAL; row++)
		if (tone_LUT[row] != -1)
			tone_LUT[row] = itone++;

	/* compute gray-tone spatial dependence matrix */
	SimpleMatrix<double> P_matrix (tone_count, tone_count);

	if (angle == 0)
		//A		P_matrix = 
		CoOcMat_Angle_0 (P_matrix, distance, grays, rows, cols, tone_LUT, tone_count);
	else if (angle == 45)
		//A		P_matrix = 
		CoOcMat_Angle_45 (P_matrix, distance, grays, rows, cols, tone_LUT, tone_count);
	else if (angle == 90)
		//A		P_matrix = 
		CoOcMat_Angle_90 (P_matrix, distance, grays, rows, cols, tone_LUT, tone_count);
	else if (angle == 135)
		//A		P_matrix = 
		CoOcMat_Angle_135 (P_matrix, distance, grays, rows, cols, tone_LUT, tone_count);
	else {
		fprintf(stderr, "Cannot created co-occurence matrix for angle %d. Unsupported angle.\n", angle);
		return; //xxx NULL;
	}



	/* compute the statistics for the spatial dependence matrix */
	Texture.ASM = f1_asm (P_matrix, tone_count);	// 2.6%
	Texture.contrast = f2_contrast (P_matrix, tone_count);	// heavy!	-> 13.42% after fix 4%
	Texture.correlation = f3_corr (P_matrix, tone_count);
	Texture.variance = f4_var (P_matrix, tone_count);
	Texture.IDM = f5_idm (P_matrix, tone_count);
	Texture.sum_avg = f6_savg (P_matrix, tone_count);	// allocation!

	/* T.J.M watch below the cast from float to double */
	sum_entropy = f8_sentropy (P_matrix, tone_count);	// allocation! but fast
	Texture.sum_entropy = sum_entropy;	// allocation! +1%
	Texture.sum_var = f7_svar (P_matrix, tone_count, sum_entropy);
	Texture.entropy = f9_entropy (P_matrix, tone_count);	// +3 ! heavy!
	Texture.diff_var = f10_dvar (P_matrix, tone_count);	// allocation	-> 6.7%
	Texture.diff_entropy = f11_dentropy (P_matrix, tone_count);	// allocation	-> 7.1%
	Texture.meas_corr1 = f12_icorr (P_matrix, tone_count);	// allocation -> 9.2%
	Texture.meas_corr2 = f13_icorr (P_matrix, tone_count);	// allocation -> 13.2%
	Texture.max_corr_coef = 0.0; // f14_maxcorr(P_matrix, tone_count);

	//A		free_matrix (P_matrix, tone_count);
	//xxx	return (Texture);

	#if 0//?
	#endif//?
}

void haralick2D_imp (
	const ImageMatrix& Im, 
	double distance, 
	// out
	std::vector<double>& Texture_Feature_Angles, 
	std::vector<double>& Texture_AngularSecondMoments,
	std::vector<double>& Texture_Contrast,
	std::vector<double>& Texture_Correlation,
	std::vector<double>& Texture_Variance,
	std::vector<double>& Texture_InverseDifferenceMoment,
	std::vector<double>& Texture_SumAverage,
	std::vector<double>& Texture_SumVariance,
	std::vector<double>& Texture_SumEntropy,
	std::vector<double>& Texture_Entropy,
	std::vector<double>& Texture_DifferenceVariance,
	std::vector<double>& Texture_DifferenceEntropy,
	std::vector<double>& Texture_InfoMeas1,
	std::vector<double>& Texture_InfoMeas2)
{
	unsigned char** p_gray;
	TEXTURE TF = {}; //xxx TEXTURE* features;
	int angle;
	double min_value, max_value;
	double scale255;

	readOnlyPixels pix_plane = Im.ReadablePixels();

	if (distance <= 0) distance = 1;

	p_gray = new unsigned char* [Im.height];
	for (auto y = 0; y < Im.height; y++)
		p_gray[y] = new unsigned char[Im.width];

	// to keep this method from modifying the const Im, we use GetStats on a local Moments2 object
	Moments2 local_stats;
	Im.GetStats(local_stats);
	min_value = local_stats.min__();
	max_value = local_stats.max__();

	scale255 = (255.0 / (max_value - min_value));
	for (auto y = 0; y < Im.height; y++)
		for (auto x = 0; x < Im.width; x++)
			p_gray[y][x] = (unsigned char)((pix_plane(y, x) - min_value) * scale255);

	Texture_Feature_Angles.clear();	// Actual angles used, just for verification
	Texture_AngularSecondMoments.clear(); 
	Texture_Contrast.clear();
	Texture_Correlation.clear();
	Texture_Variance.clear();
	Texture_InverseDifferenceMoment.clear();
	Texture_SumAverage.clear();
	Texture_SumVariance.clear();
	Texture_SumEntropy.clear();
	Texture_Entropy.clear();
	Texture_DifferenceVariance.clear();
	Texture_DifferenceEntropy.clear();
	Texture_InfoMeas1.clear();
	Texture_InfoMeas2.clear();

	// Conflict. Make standard angles at leat appear in the output
	//for (auto angle : theEnvironment.rotAngles) 
	theEnvironment.rotAngles.clear();
	theEnvironment.rotAngles.push_back(0);
	theEnvironment.rotAngles.push_back(45);
	theEnvironment.rotAngles.push_back(90);
	theEnvironment.rotAngles.push_back(135);



	for (angle = 0; angle <= 135; angle = angle + 45) 
	{
		Extract_Texture_Features((int)distance, angle, p_gray, Im.height, Im.width, TF); //xxx features = Extract_Texture_Features ((int)distance, angle, p_gray, Im.height, Im.width);

		Texture_Feature_Angles.push_back(angle);

		Texture_AngularSecondMoments.push_back (TF.ASM);
		Texture_Contrast.push_back (TF.contrast);
		Texture_Correlation.push_back (TF.correlation);
		Texture_Variance.push_back (TF.variance);
		Texture_InverseDifferenceMoment.push_back (TF.IDM);
		Texture_SumAverage.push_back (TF.sum_avg);
		Texture_SumVariance.push_back (TF.sum_var);
		Texture_SumEntropy.push_back (TF.sum_entropy);
		Texture_Entropy.push_back (TF.entropy);
		Texture_DifferenceVariance.push_back (TF.diff_var);
		Texture_DifferenceEntropy.push_back (TF.diff_entropy);
		Texture_InfoMeas1.push_back (TF.meas_corr1);
		Texture_InfoMeas2.push_back (TF.meas_corr2);

		//xxx free(features);
	}

	for (auto y = 0; y < Im.height; y++)
		delete[] p_gray[y];
	delete[] p_gray;
}

void haralick2D(
	// in
	std::vector <Pixel2>& nonzero_intensity_pixels,
	AABB & aabb,
	double distance,
	// out	
	std::vector<double>& Texture_Feature_Angles, 
	std::vector<double>& Texture_AngularSecondMoments, 
	std::vector<double>& Texture_Contrast,
	std::vector<double>& Texture_Correlation,
	std::vector<double>& Texture_Variance, 
	std::vector<double>& Texture_InverseDifferenceMoment,
	std::vector<double>& Texture_SumAverage,
	std::vector<double>& Texture_SumVariance,
	std::vector<double>& Texture_SumEntropy,
	std::vector<double>& Texture_Entropy,
	std::vector<double>& Texture_DifferenceVariance,
	std::vector<double>& Texture_DifferenceEntropy,
	std::vector<double>& Texture_InfoMeas1,
	std::vector<double>& Texture_InfoMeas2)
{
	// Create a temp image matrix from label's pixels
	ImageMatrix im (nonzero_intensity_pixels, aabb);

	// Call the Wndchrm's implementation
	haralick2D_imp (im, distance, 
		Texture_Feature_Angles, 
		Texture_AngularSecondMoments, 
		Texture_Contrast,
		Texture_Correlation,
		Texture_Variance, 
		Texture_InverseDifferenceMoment,
		Texture_SumAverage,
		Texture_SumVariance,
		Texture_SumEntropy,
		Texture_Entropy,
		Texture_DifferenceVariance,
		Texture_DifferenceEntropy,
		Texture_InfoMeas1,
		Texture_InfoMeas2);
}