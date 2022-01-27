#include "glcm.h"

#define PGM_MAXMAXVAL 255

GLCMFeature::GLCMFeature() : FeatureMethod("GLCMFeature")
{
	provide_features({
		GLCM_ANGULAR2NDMOMENT,
		GLCM_CONTRAST,
		GLCM_CORRELATION,
		GLCM_VARIANCE,
		GLCM_INVERSEDIFFERENCEMOMENT,
		GLCM_SUMAVERAGE,
		GLCM_SUMVARIANCE,
		GLCM_SUMENTROPY,
		GLCM_ENTROPY,
		GLCM_DIFFERENCEVARIANCE,
		GLCM_DIFFERENCEENTROPY,
		GLCM_INFOMEAS1,
		GLCM_INFOMEAS2
			});
}

void GLCMFeature::calculate(LR& r)
{
	SimpleMatrix<uint8_t> G;
	calculate_graytones(G, r.aux_min, r.aux_max, r.aux_image_matrix);

	int Angles[] = { 0, 45, 90, 135 },
		n = sizeof(Angles) / sizeof(Angles[0]);
	for (int i = 0; i < n; i++)
		Extract_Texture_Features (distance_parameter, Angles[i], G);
}

void GLCMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void GLCMFeature::osized_calculate(LR& r, ImageLoader& imloader)
{}

void GLCMFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	get_AngularSecondMoments(fvals[GLCM_ANGULAR2NDMOMENT]);
	get_Contrast(fvals[GLCM_CONTRAST]);
	get_Correlation(fvals[GLCM_CORRELATION]);
	get_Variance(fvals[GLCM_VARIANCE]);
	get_InverseDifferenceMoment(fvals[GLCM_INVERSEDIFFERENCEMOMENT]);
	get_SumAverage(fvals[GLCM_SUMAVERAGE]);
	get_SumVariance(fvals[GLCM_SUMVARIANCE]);
	get_SumEntropy(fvals[GLCM_SUMENTROPY]);
	get_Entropy(fvals[GLCM_ENTROPY]);
	get_DifferenceVariance(fvals[GLCM_DIFFERENCEVARIANCE]);
	get_DifferenceEntropy(fvals[GLCM_DIFFERENCEENTROPY]);
	get_InfoMeas1(fvals[GLCM_INFOMEAS1]);
	get_InfoMeas2(fvals[GLCM_INFOMEAS2]);
}

void GLCMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
		{
			// Dfault values for all 4 standard angles
			r.fvals[GLCM_ANGULAR2NDMOMENT].resize(4, 0);
			r.fvals[GLCM_CONTRAST].resize(4, 0);
			r.fvals[GLCM_CORRELATION].resize(4, 0);
			r.fvals[GLCM_VARIANCE].resize(4, 0);
			r.fvals[GLCM_INVERSEDIFFERENCEMOMENT].resize(4, 0);
			r.fvals[GLCM_SUMAVERAGE].resize(4, 0);
			r.fvals[GLCM_SUMVARIANCE].resize(4, 0);
			r.fvals[GLCM_SUMENTROPY].resize(4, 0);
			r.fvals[GLCM_ENTROPY].resize(4, 0);
			r.fvals[GLCM_DIFFERENCEVARIANCE].resize(4, 0);
			r.fvals[GLCM_DIFFERENCEENTROPY].resize(4, 0);
			r.fvals[GLCM_INFOMEAS1].resize(4, 0);
			r.fvals[GLCM_INFOMEAS2].resize(4, 0);
			continue;
		}

		//=== GLCM version 2
		// Skip calculation in case of bad data
		int minI = (int)r.fvals[MIN][0],
			maxI = (int)r.fvals[MAX][0];
		if (minI == maxI)
		{
			// Dfault values for all 4 standard angles
			r.fvals[GLCM_ANGULAR2NDMOMENT].resize(4, 0);
			r.fvals[GLCM_CONTRAST].resize(4, 0);
			r.fvals[GLCM_CORRELATION].resize(4, 0);
			r.fvals[GLCM_VARIANCE].resize(4, 0);
			r.fvals[GLCM_INVERSEDIFFERENCEMOMENT].resize(4, 0);
			r.fvals[GLCM_SUMAVERAGE].resize(4, 0);
			r.fvals[GLCM_SUMVARIANCE].resize(4, 0);
			r.fvals[GLCM_SUMENTROPY].resize(4, 0);
			r.fvals[GLCM_ENTROPY].resize(4, 0);
			r.fvals[GLCM_DIFFERENCEVARIANCE].resize(4, 0);
			r.fvals[GLCM_DIFFERENCEENTROPY].resize(4, 0);
			r.fvals[GLCM_INFOMEAS1].resize(4, 0);
			r.fvals[GLCM_INFOMEAS2].resize(4, 0);
			continue;
		}

		GLCMFeature f; 
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void GLCMFeature::calculate_graytones (SimpleMatrix<uint8_t>& G, int minI, int maxI, const ImageMatrix& Im)
{
	const pixData& I = Im.ReadablePixels();
	G.allocate(I.width(), I.height(), 0);
	double scale255 = 255.0 / double(maxI-minI);
	for (auto y = 0; y < I.height(); y++)
		for (auto x = 0; x < I.width(); x++)
			G(x,y) = (uint8_t)((I(y, x) - minI) * scale255);
}

void GLCMFeature::Extract_Texture_Features (
	int distance, 
	int angle, 
	const SimpleMatrix<uint8_t>& grays)	// 'grays' is 0-255 grays 
{
	int nrows = grays.height();
	int ncols = grays.width();

	int tone_LUT [PGM_MAXMAXVAL + 1]; /* LUT mapping gray tone(0-255) to matrix indicies */
	int tone_count = 0; /* number of tones actually in the img. atleast 1 less than 255 */

	/* Determine the number of different gray tones (not maxval) */
	for (int row = PGM_MAXMAXVAL; row >= 0; --row)
		tone_LUT[row] = -1;
	for (int row = nrows - 1; row >= 0; --row)
		for (int col = 0; col < ncols; ++col)
			tone_LUT[grays(col,row)] = grays(col,row);

	for (int row = PGM_MAXMAXVAL; row >= 0; --row)
		if (tone_LUT[row] != -1)
			tone_count++;

	/* Use the number of different tones to build LUT */
	for (int row = 0, itone = 0; row <= PGM_MAXMAXVAL; row++)
		if (tone_LUT[row] != -1)
			tone_LUT[row] = itone++;

	// Allocate Px and Py vectors
	std::vector<double> Px(tone_count * 2), Py(tone_count);

	/* compute gray-tone spatial dependence matrix */
	SimpleMatrix<double> P_matrix (tone_count, tone_count);

	if (angle == 0)
		CoOcMat_Angle_0 (P_matrix, distance, grays, tone_LUT, tone_count);
	else if (angle == 45)
		CoOcMat_Angle_45 (P_matrix, distance, grays, tone_LUT, tone_count);
	else if (angle == 90)
		CoOcMat_Angle_90 (P_matrix, distance, grays, tone_LUT, tone_count);
	else if (angle == 135)
		CoOcMat_Angle_135 (P_matrix, distance, grays, tone_LUT, tone_count);
	else {
		fprintf(stderr, "Cannot create co-occurence matrix for angle %d. Unsupported angle.\n", angle);
		return; 
	}

	/* compute the statistics for the spatial dependence matrix */
	fvals_ASM.push_back (f1_asm (P_matrix, tone_count));	// 2.6%
	fvals_contrast.push_back (f2_contrast (P_matrix, tone_count));	// heavy!	-> 13.42% after fix 4%
	fvals_correlation.push_back (f3_corr (P_matrix, tone_count, Px));
	fvals_variance.push_back (f4_var (P_matrix, tone_count));
	fvals_IDM.push_back (f5_idm (P_matrix, tone_count));
	fvals_sum_avg.push_back (f6_savg (P_matrix, tone_count, Px));	// allocation!
	double se = f8_sentropy(P_matrix, tone_count, Px);
	fvals_sum_entropy.push_back (se);	// allocation! but fast +1%
	fvals_sum_var.push_back (f7_svar (P_matrix, tone_count, se, Px));
	fvals_entropy.push_back (f9_entropy (P_matrix, tone_count));	// +3 ! heavy!
	fvals_diff_var.push_back (f10_dvar (P_matrix, tone_count, Px));	// allocation	-> 6.7%
	fvals_diff_entropy.push_back (f11_dentropy (P_matrix, tone_count, Px));	// allocation	-> 7.1%
	fvals_meas_corr1.push_back (f12_icorr (P_matrix, tone_count, Px, Py));	// allocation -> 9.2%
	fvals_meas_corr2.push_back (f13_icorr (P_matrix, tone_count, Px, Py));	// allocation -> 13.2%
	fvals_max_corr_coef.push_back (0.0); // f14_maxcorr(P_matrix, tone_count);

}

/* Compute gray-tone spatial dependence matrix */
void GLCMFeature::CoOcMat_Angle_0 (
	// out
	SimpleMatrix<double>& matrix, 
	// in
	int distance, 
	const SimpleMatrix<uint8_t>& grays, 
	const int* tone_LUT, 
	int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; /* normalizing factor */

	/* zero out matrix */
	matrix.fill (0.0);

	int rows = grays.height(),
		cols = grays.width();	
	
	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) 
		{
			/* only non-zero values count*/
			if (grays(col,row) == 0)
				continue;

			/* find x tone */
			if (col + d < cols && grays(col+d, row)) 
			{
				x = tone_LUT[grays(col, row)];
				y = tone_LUT[grays(col + d, row)];
				matrix(x, y)++;		
				matrix(y, x)++;		
				count += 2;
			}
		}

	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0)   /* protect from error */
				matrix(itone, jtone) = 0;	
			else
				matrix(itone, jtone) /= count;	
}

void GLCMFeature::CoOcMat_Angle_90(
	// out
	SimpleMatrix<double>& matrix,
	// in
	int distance,
	const SimpleMatrix<uint8_t>& grays,
	const int* tone_LUT,
	int tone_count)
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

	int rows = grays.height(),
		cols = grays.width();

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			if (grays(col, row) == 0)
				continue;

			/* find x tone */
			if (row + d < rows && grays(col, row + d)) {
				x = tone_LUT[grays(col,row)];
				y = tone_LUT[grays(col,row + d)];
				matrix(x, y)++;		//A		matrix[x][y]++;
				matrix(y, x)++;		//A		matrix[y][x]++;
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

void GLCMFeature::CoOcMat_Angle_45(
	// out
	SimpleMatrix<double>& matrix,
	// in
	int distance,
	const SimpleMatrix<uint8_t>& grays,
	const int* tone_LUT,
	int tone_count)
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

	int rows = grays.height(),
		cols = grays.width();

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			if (grays(col,row) == 0)
				continue;

			/* find x tone */
			if (row + d < rows && col - d >= 0 && grays(col - d, row + d)) {
				x = tone_LUT[grays(col - d, row)];
				y = tone_LUT[grays(col - d, row + d)];
				matrix(x, y)++;		//A		matrix[x][y]++;
				matrix(y, x)++;		//A		matrix[y][x]++;
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

void GLCMFeature::CoOcMat_Angle_135(
	// out
	SimpleMatrix<double>& matrix,
	// in
	int distance,
	const SimpleMatrix<uint8_t>& grays,
	const int* tone_LUT,
	int tone_count)
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

	int rows = grays.height(),
		cols = grays.width();

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			if (grays(col,row) == 0)
				continue;

			/* find x tone */
			if (row + d < rows && col + d < cols && grays(col + d, row + d)) {
				x = tone_LUT[grays(col, row)];
				y = tone_LUT[grays(col + d, row + d)];
				matrix(x, y)++;	//A		matrix[x][y]++;
				matrix(y, x)++;	//A		matrix[y][x]++;
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

void GLCMFeature::get_AngularSecondMoments(AngledFeatures& af)
{
	af.assign (fvals_ASM.begin(), fvals_ASM.end());
}

void GLCMFeature::get_Contrast(AngledFeatures& af)
{
	af.assign (fvals_contrast.begin(), fvals_contrast.end());
}

void GLCMFeature::get_Correlation(AngledFeatures& af)
{
	af.assign (fvals_correlation.begin(), fvals_correlation.end());
}

void GLCMFeature::get_Variance(AngledFeatures& af)
{
	af.assign (fvals_variance.begin(), fvals_variance.end());
}

void GLCMFeature::get_InverseDifferenceMoment(AngledFeatures& af)
{
	af.assign (fvals_IDM.begin(), fvals_IDM.end());
}

void GLCMFeature::get_SumAverage(AngledFeatures& af)
{
	af.assign (fvals_sum_avg.begin(), fvals_sum_avg.end());
}

void GLCMFeature::get_SumVariance(AngledFeatures& af)
{
	af.assign (fvals_sum_var.begin(), fvals_sum_var.end());
}

void GLCMFeature::get_SumEntropy(AngledFeatures& af)
{
	af.assign (fvals_sum_entropy.begin(), fvals_sum_entropy.end());
}

void GLCMFeature::get_Entropy(AngledFeatures& af)
{
	af.assign (fvals_entropy.begin(), fvals_entropy.end());
}

void GLCMFeature::get_DifferenceVariance(AngledFeatures& af)
{
	af.assign (fvals_diff_var.begin(), fvals_diff_var.end());
}

void GLCMFeature::get_DifferenceEntropy(AngledFeatures& af)
{
	af.assign (fvals_diff_entropy.begin(), fvals_diff_entropy.end());
}

void GLCMFeature::get_InfoMeas1(AngledFeatures& af)
{
	af.assign (fvals_meas_corr1.begin(), fvals_meas_corr1.end());
}

void GLCMFeature::get_InfoMeas2(AngledFeatures& af)
{
	af.assign (fvals_meas_corr2.begin(), fvals_meas_corr2.end());
}

/* Angular Second Moment
*
* The angular second-moment feature (ASM) f1 is a measure of homogeneity
* of the image. In a homogeneous image, there are very few dominant
* gray-tone transitions. Hence the P matrix for such an image will have
* fewer entries of large magnitude.
*/
double GLCMFeature::f1_asm (const SimpleMatrix<double>& P, int Ng)
{
	int i, j;
	double sum = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			sum += P(i, j) * P(i, j);

	return sum;
}

/* Contrast
*
* The contrast feature is a difference moment of the P matrix and is a
* measure of the contrast or the amount of local variations present in an
* image.
*/
double GLCMFeature::f2_contrast (const SimpleMatrix<double>& P, int Ng)
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
			sum += P(i, j) * (j - i) * (i - j);
	return sum;
}

/*---
double* GLCM_features::allocate_vector(int nl, int nh) {
	double* v;

	v = (double*)calloc(1, (unsigned)(nh - nl + 1) * sizeof(double));
	if (!v) fprintf(stderr, "memory allocation failure (allocate_vector) "), exit(1);

	return v - nl;
}
*/

/* Correlation
*
* This correlation feature is a measure of gray-tone linear-dependencies
* in the image.
*/
double GLCMFeature::f3_corr (const SimpleMatrix<double>& P, int Ng, std::vector<double> & px)
{
	int i, j;
	double sum_sqrx = 0, tmp;
	double meanx = 0, meany = 0, stddevx, stddevy;

	//---	double *px = allocate_vector(0, Ng);
	//		for (i = 0; i < Ng; ++i)
	//			px[i] = 0;
	std::fill(px.begin(), px.end(), 0.0);

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			px[i] += P(i, j);


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
			tmp += i * j * P(i, j);

	//---	free(px);

	if (stddevx * stddevy == 0) return(1);  /* protect from error */
	else return (tmp - meanx * meany) / (stddevx * stddevy);
}

/* Sum of Squares: Variance */
double GLCMFeature::f4_var(const SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double mean = 0, var = 0;

	/*- Corrected by James Darrell McCauley, 16 Aug 1991
	*  calculates the mean intensity level instead of the mean of
	*  cooccurrence matrix elements
	*/
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			mean += i * P(i, j);

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/*  M. Boland - var += (i + 1 - mean) * (i + 1 - mean) * P[i][j]; */
			var += (i - mean) * (i - mean) * P(i, j);

	return var;
}

/* Inverse Difference Moment */
double GLCMFeature::f5_idm(const SimpleMatrix<double>& P, int Ng) {
	int i, j;
	double idm = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			idm += P(i, j) / (1 + (i - j) * (i - j));

	return idm;
}

/* Sum Average */
double GLCMFeature::f6_savg(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy) {
	int i, j;
	double savg = 0;
	//---	double* Pxpy = allocate_vector(0, 2 * Ng);
	//		for (i = 0; i <= 2 * Ng; ++i)
	//			Pxpy[i] = 0;
	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/* M. Boland Pxpy[i + j + 2] += P[i][j]; */
			/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
			Pxpy[i + j] += P(i, j);

	/* M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		savg += i * Pxpy[i];

	//---	free(Pxpy);

	return savg;
}

/* Sum Variance */
double GLCMFeature::f7_svar(const SimpleMatrix<double>& P, int Ng, double S, std::vector<double>& Pxpy) {
	int i, j;
	double var = 0;
	//---	double* Pxpy = allocate_vector(0, 2 * Ng);
	//		for (i = 0; i <= 2 * Ng; ++i)
	//			Pxpy[i] = 0;
	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/* M. Boland Pxpy[i + j + 2] += P[i][j]; */
			/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
			Pxpy[i + j] += P(i, j);

	/*  M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		var += (i - S) * (i - S) * Pxpy[i];

	//---	free(Pxpy);

	return var;
}

#define EPSILON 0.000000001

/* Sum Entropy */
double GLCMFeature::f8_sentropy(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy)
{
	int i, j;
	double sentropy = 0;
	//---	double* Pxpy = allocate_vector(0, 2 * Ng);
	//		for (i = 0; i <= 2 * Ng; ++i)
	//			Pxpy[i] = 0;
	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (i = 0; i < Ng-1; ++i)
		for (j = 0; j < Ng-1; ++j)
			Pxpy[i + j + 2] += P(i, j);

	for (i = 2; i < 2 * Ng; ++i)
		/*  M. Boland  sentropy -= Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sentropy -= Pxpy[i] * log10(Pxpy[i] + EPSILON) / log10(2.0);

	//---	free(Pxpy);

	return sentropy;
}

/* Entropy */
double GLCMFeature::f9_entropy(const SimpleMatrix<double>& P, int Ng)
{
	int i, j;
	double entropy = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/*      entropy += P[i][j] * log10 (P[i][j] + EPSILON); */
			entropy += P(i, j) * log10(P(i, j) + EPSILON) / log10(2.0);

	return -entropy;
}

/* Difference Variance */
double GLCMFeature::f10_dvar(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy)
{
	int i, j;
	double sum = 0, sum_sqr = 0, var = 0;
	//---	double* Pxpy = allocate_vector(0, 2 * Ng);
	//		for (i = 0; i <= 2 * Ng; ++i)
	//			Pxpy[i] = 0;
	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			Pxpy[abs(i - j)] += P(i, j);

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

	//---	free(Pxpy);

	return var;
}

/* Difference Entropy */
double GLCMFeature::f11_dentropy(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy) {
	int i, j;
	double sum = 0;
	//---	double* Pxpy = allocate_vector(0, 2 * Ng);
	//		for (i = 0; i <= 2 * Ng; ++i)
	//			Pxpy[i] = 0;
	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			Pxpy[abs(i - j)] += P(i, j);

	for (i = 0; i < Ng; ++i)
		/*    sum += Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sum += Pxpy[i] * log10(Pxpy[i] + EPSILON) / log10(2.0);

	//---	free(Pxpy);

	return -sum;
}

/* Information Measures of Correlation */
double GLCMFeature::f12_icorr(const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py) {
	int i, j;
	double hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;

	//---	double *px = allocate_vector(0, Ng);
	//---	double *py = allocate_vector(0, Ng);
	/* All /log10(2.0) added by M. Boland */

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			px[i] += P(i, j);
			py[j] += P(i, j);
		}
	}

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j) {
			hxy1 -= P(i, j) * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy2 -= px[i] * py[j] * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy -= P(i, j) * log10(P(i, j) + EPSILON) / log10(2.0);
		}

	/* Calculate entropies of px and py - is this right? */
	for (i = 0; i < Ng; ++i) {
		hx -= px[i] * log10(px[i] + EPSILON) / log10(2.0);
		hy -= py[i] * log10(py[i] + EPSILON) / log10(2.0);
	}

	//---	free(px);
	//---	free(py);

	if ((hx > hy ? hx : hy) == 0) return(1);
	else
		return ((hxy - hxy1) / (hx > hy ? hx : hy));
}

/* Information Measures of Correlation */
double GLCMFeature::f13_icorr(const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py) {
	int i, j;
	double hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;

	//---	double *px = allocate_vector(0, Ng);
	//---	double *py = allocate_vector(0, Ng);

	/* All /log10(2.0) added by M. Boland */

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			px[i] += P(i, j);
			py[j] += P(i, j);
		}
	}

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
		{
			hxy1 -= P(i, j) * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy2 -= px[i] * py[j] * log10(px[i] * py[j] + EPSILON) / log10(2.0);
			hxy -= P(i, j) * log10(P(i, j) + EPSILON) / log10(2.0);
		}

	/* Calculate entropies of px and py */
	for (i = 0; i < Ng; ++i) {
		hx -= px[i] * log10(px[i] + EPSILON) / log10(2.0);
		hy -= py[i] * log10(py[i] + EPSILON) / log10(2.0);
	}

	//---	free(px);
	//---	free(py);

	return (sqrt(fabs(1 - exp(-2.0 * (hxy2 - hxy)))));
}




