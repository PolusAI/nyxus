#include "glcm.h"
#include "../helpers/helpers.h"
#include "../environment.h"

int GLCMFeature::offset = 1;
int GLCMFeature::n_levels = 8;
std::vector<int> GLCMFeature::angles = { 0, 45, 90, 135 };

GLCMFeature::GLCMFeature() : FeatureMethod("GLCMFeature")
{
	provide_features({
		GLCM_ANGULAR2NDMOMENT,
		GLCM_CONTRAST,
		GLCM_CORRELATION,
		GLCM_DIFFERENCEAVERAGE,	
		GLCM_DIFFERENCEENTROPY,
		GLCM_DIFFERENCEVARIANCE,
		GLCM_ENERGY, 
		GLCM_ENTROPY,
		GLCM_HOMOGENEITY,	
		GLCM_INFOMEAS1,
		GLCM_INFOMEAS2,
		GLCM_INVERSEDIFFERENCEMOMENT,
		GLCM_SUMAVERAGE,
		GLCM_SUMENTROPY,
		GLCM_SUMVARIANCE,
		GLCM_VARIANCE
		});
}

void GLCMFeature::calculate(LR& r)
{
	for (auto a: angles)
		Extract_Texture_Features2 (a, r.aux_image_matrix, r.aux_min, r.aux_max); 
}

void GLCMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}		// Not supporting the online mode for this feature method

void GLCMFeature::copyfvals (AngledFeatures& dst, const AngledFeatures& src)
{
	dst.assign(src.begin(), src.end());
}

void GLCMFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	copyfvals (fvals[GLCM_ANGULAR2NDMOMENT], fvals_ASM);
	copyfvals (fvals[GLCM_CONTRAST], fvals_contrast);
	copyfvals (fvals[GLCM_CORRELATION], fvals_correlation);
	copyfvals (fvals[GLCM_DIFFERENCEAVERAGE], fvals_diff_avg);
	copyfvals (fvals[GLCM_DIFFERENCEVARIANCE], fvals_diff_var);
	copyfvals (fvals[GLCM_DIFFERENCEENTROPY], fvals_diff_entropy);
	copyfvals (fvals[GLCM_ENERGY], fvals_energy);
	copyfvals (fvals[GLCM_ENTROPY], fvals_entropy);
	copyfvals (fvals[GLCM_HOMOGENEITY], fvals_homo);
	copyfvals (fvals[GLCM_INFOMEAS1], fvals_meas_corr1);
	copyfvals (fvals[GLCM_INFOMEAS2], fvals_meas_corr2);
	copyfvals (fvals[GLCM_INVERSEDIFFERENCEMOMENT], fvals_IDM);
	copyfvals (fvals[GLCM_SUMAVERAGE], fvals_sum_avg);
	copyfvals (fvals[GLCM_SUMVARIANCE], fvals_sum_var);
	copyfvals (fvals[GLCM_SUMENTROPY], fvals_sum_entropy);
	copyfvals (fvals[GLCM_VARIANCE], fvals_variance);
}

void GLCMFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Skip calculation in case of bad data
		auto minI = Nyxus::to_grayscale(r.aux_min, r.aux_min, r.aux_max-r.aux_min, theEnvironment.get_coarse_gray_depth()),
			maxI = Nyxus::to_grayscale(r.aux_max, r.aux_min, r.aux_max - r.aux_min, theEnvironment.get_coarse_gray_depth());
		if (minI == maxI)
		{
			auto n = angles.size();
			// Zero out each angled feature value 
			r.fvals [GLCM_ANGULAR2NDMOMENT].resize (n, 0);
			r.fvals [GLCM_CONTRAST].resize (n, 0);
			r.fvals [GLCM_CORRELATION].resize (n, 0);
			r.fvals [GLCM_VARIANCE].resize (n, 0);
			r.fvals [GLCM_INVERSEDIFFERENCEMOMENT].resize (n, 0);
			r.fvals [GLCM_SUMAVERAGE].resize (n, 0);
			r.fvals [GLCM_SUMVARIANCE].resize (n, 0);
			r.fvals [GLCM_SUMENTROPY].resize(n, 0);
			r.fvals [GLCM_ENTROPY].resize(n, 0);
			r.fvals [GLCM_DIFFERENCEVARIANCE].resize(n, 0);
			r.fvals [GLCM_DIFFERENCEENTROPY].resize(n, 0);
			r.fvals [GLCM_INFOMEAS1].resize(n, 0);
			r.fvals [GLCM_INFOMEAS2].resize(n, 0);
			continue;
		}

		GLCMFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void GLCMFeature::Extract_Texture_Features2 (int angle, const ImageMatrix & grays, PixIntens min_val, PixIntens max_val)
{
	int nrows = grays.height;
	int ncols = grays.width;

	// Allocate Px and Py vectors
	std::vector<double> Px (n_levels * 2), 
		Py (n_levels);

	// Compute the gray-tone spatial dependence matrix 
	int dx, dy;
	switch (angle)
	{
		case 0:
			dx = offset;
			dy = 0;
			break;
		case 45:
			dx = offset;
			dy = offset;
			break;
		case 90:
			dx = 0;
			dy = offset;
			break;
		case 135:
			dx = -offset;
			dy = offset;
			break;
		default:
			std::cerr << "Cannot create co-occurence matrix for angle " << angle << ": unsupported angle\n";
			return;
	}
	calculateCoocMatAtAngle (P_matrix, dx, dy, grays, min_val, max_val, false);
	calculatePxpmy ();

	// Compute Haralick statistics 
	double f;
	f = theFeatureSet.isEnabled(GLCM_ANGULAR2NDMOMENT) ? f_asm(P_matrix, n_levels) : 0.0;
	fvals_ASM.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_CONTRAST) ? f_contrast(P_matrix, n_levels) : 0.0;
	fvals_contrast.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_CORRELATION) ? f_corr(P_matrix, n_levels, Px) : 0.0;
	fvals_correlation.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_ENERGY) ? f_energy (P_matrix, n_levels, Px) : 0.0;
	fvals_energy.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_HOMOGENEITY) ? f_homogeneity (P_matrix, n_levels, Px) : 0.0;
	fvals_homo.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_VARIANCE) ? f_var (P_matrix, n_levels) : 0.0;
	fvals_variance.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_INVERSEDIFFERENCEMOMENT) ? f_idm (P_matrix, n_levels) : 0.0;
	fvals_IDM.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_SUMAVERAGE) ? f_savg (P_matrix, n_levels, Px) : 0.0;
	fvals_sum_avg.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_SUMENTROPY) ? f_sentropy (P_matrix, n_levels, Px) : 0.0;
	fvals_sum_entropy.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_SUMVARIANCE) ? f_svar (P_matrix, n_levels, f, Px) : 0.0;
	fvals_sum_var.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_ENTROPY) ? f_entropy (P_matrix, n_levels) : 0.0;
	fvals_entropy.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_DIFFERENCEVARIANCE) ? f_dvar (P_matrix, n_levels, Px) : 0.0;
	fvals_diff_var.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_DIFFERENCEENTROPY) ? f_dentropy (P_matrix, n_levels, Px) : 0.0;
	fvals_diff_entropy.push_back (f);

	f = theFeatureSet.isEnabled (GLCM_DIFFERENCEAVERAGE) ? f_difference_avg (P_matrix, n_levels, Px) : 0.0;
	fvals_diff_avg.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_INFOMEAS1) ? f_info_meas_corr1 (P_matrix, n_levels, Px, Py) : 0.0;
	fvals_meas_corr1.push_back (f);

	f = theFeatureSet.isEnabled(GLCM_INFOMEAS2) ? f_info_meas_corr2 (P_matrix, n_levels, Px, Py) : 0.0;
	fvals_meas_corr2.push_back (f);

	fvals_max_corr_coef.push_back (0.0);
}

void GLCMFeature::calculateCoocMatAtAngle(
	// out
	SimpleMatrix<double>& matrix,
	// in
	int dx,
	int dy,
	const ImageMatrix& grays,
	PixIntens min_val,
	PixIntens max_val, 
	bool normalize)
{
	matrix.allocate(n_levels, n_levels); 
	matrix.fill(0.0);

	int d = GLCMFeature::offset;
	int count = 0;	// normalizing factor 

	int rows = grays.height,
		cols = grays.width;

	PixIntens range = max_val - min_val;

	const pixData& graysdata = grays.ReadablePixels();

	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++)
		{
			if (row + dy >= 0 && row + dy < rows && col + dx >= 0 && col + dx < cols && graysdata.yx(row + dy, col + dx))
			{
				// Raw intensities
				auto raw_lvl_y = graysdata.yx(row, col),
					raw_lvl_x = graysdata.yx(row + dy, col + dx);

				// Skip non-informative pixels
				if (raw_lvl_x == 0 || raw_lvl_y == 0)
					continue;

				// Cast intensities on the 1-n_levels scale
				int x = GLCMFeature::cast_to_range (raw_lvl_x, min_val, max_val, 1, GLCMFeature::n_levels) -1, 
					y = GLCMFeature::cast_to_range (raw_lvl_y, min_val, max_val, 1, GLCMFeature::n_levels) -1;

				// Increment the symmetric count
				count += 2;	
				matrix.xy(y,x)++;
				matrix.xy(x,y)++;

				#ifdef TEST_GLCM
				std::stringstream ss;
				ss << y << "," << x;
				print_doubles_matrix(matrix, "after " + ss.str(), "");
				#endif
			}
		}

	// Normalize the matrix
	if (normalize == false)
		return;

	double realCnt = count;
	for (int i = 0; i < GLCMFeature::n_levels; i++)
		for (int j = 0; j < GLCMFeature::n_levels; j++)
			matrix.xy(i, j) /= (realCnt + EPSILON);
}

void GLCMFeature::calculatePxpmy()
{
	Pxpy.resize (2 * n_levels - 1, 0.0);
	Pxmy.resize (n_levels, 0.0);

	for (int x = 0; x < n_levels; x++) 
		for (int y = 0; y < n_levels; y++) 
		{
			Pxpy[x + y] += P_matrix.xy(x,y);
			Pxmy[std::abs(x - y)] += P_matrix.xy(x,y); 
		}
}

/* Angular Second Moment
*
* The angular second-moment feature (ASM) f1 is a measure of homogeneity
* of the image. In a homogeneous image, there are very few dominant
* gray-tone transitions. Hence the P matrix for such an image will have
* fewer entries of large magnitude.
*/
double GLCMFeature::f_asm(const SimpleMatrix<double>& P, int Ng)
{
	int i, j;
	double sum = 0;

	for (j = 0; j < Ng; ++j)
		for (i = 0; i < Ng; ++i)
			sum += P.xy(i, j) * P.xy(i, j);

	return sum;
}

/* Contrast
*
* The contrast feature is a difference moment of the P matrix and is a
* measure of the contrast or the amount of local variations present in an
* image.
*/
double GLCMFeature::f_contrast(const SimpleMatrix<double>& P, int Ng)
{
	double sum = 0;

	for (int j = 0; j < Ng; j++)
		for (int i = 0; i < Ng; i++)
			sum += P.xy(i, j) * (j - i) * (i - j);

	return sum;
}

/* Correlation
*
* This correlation feature is a measure of gray-tone linear-dependencies
* in the image.
*/
double GLCMFeature::f_corr (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px)
{
	int i, j;
	double sum_sqrx = 0, tmp;
	double meanx = 0, meany = 0, stddevx, stddevy;

	std::fill (px.begin(), px.end(), 0.0);

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (int j = 0; j < Ng; j++)
		for (int i = 0; i < Ng; i++)
			px[i] += P.xy(i, j);


	/* Now calculate the means and standard deviations of px and py */
	/*- fix supplied by J. Michael Christensen, 21 Jun 1991 */
	/*- further modified by James Darrell McCauley, 16 Aug 1991
	*     after realizing that meanx=meany and stddevx=stddevy
	*/
	for (i = 0; i < Ng; ++i) 
	{
		meanx += px[i] * i;
		sum_sqrx += px[i] * i * i;
	}

	/* M. Boland meanx = meanx/(sqrt(Ng)); */
	meany = meanx;
	stddevx = sqrt(sum_sqrx - (meanx * meanx));
	stddevy = stddevx;

	/* Finally, the correlation ... */
	tmp = 0;
	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			tmp += i * j * P.xy(i, j);

	if (stddevx * stddevy == 0) 
		return(1);  // protect from error
	else 
		return (tmp - meanx * meany) / (stddevx * stddevy);
}

/* Sum of Squares: Variance */
double GLCMFeature::f_var(const SimpleMatrix<double>& P, int Ng) 
{
	int i, j;
	double mean = 0, var = 0;

	/*- Corrected by James Darrell McCauley, 16 Aug 1991
	*  calculates the mean intensity level instead of the mean of
	*  cooccurrence matrix elements
	*/
	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			mean += i * P.xy(i, j);

	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			/*  M. Boland - var += (i + 1 - mean) * (i + 1 - mean) * P[i][j]; */
			var += (i - mean) * (i - mean) * P.xy(i, j);

	return var;
}

/* Inverse Difference Moment */
double GLCMFeature::f_idm(const SimpleMatrix<double>& P, int Ng) 
{
	double idm = 0;

	for (int j = 0; j < Ng; j++)
		for (int i = 0; i < Ng; i++)
			idm += P.xy(i, j) / (1 + (i - j) * (i - j));

	return idm;
}

/* Sum Average */
double GLCMFeature::f_savg(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy) 
{
	int i, j;
	double savg = 0;

	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			/* M. Boland Pxpy[i + j + 2] += P[i][j]; */
			/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
			Pxpy[i + j] += P.xy(i, j);

	/* M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		savg += i * Pxpy[i];

	return savg;
}

/* Sum Variance */
double GLCMFeature::f_svar(const SimpleMatrix<double>& P, int Ng, double S, std::vector<double>& Pxpy) 
{
	int i, j;
	double var = 0;

	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			/* M. Boland Pxpy[i + j + 2] += P[i][j]; */
			/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
			Pxpy[i + j] += P.xy(i, j);

	/*  M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		var += (double(i) - S) * (double(i) - S) * Pxpy[i];

	return var;
}

/* Sum Entropy */
double GLCMFeature::f_sentropy(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy)
{
	int i, j;
	double sentropy = 0;

	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (j = 0; j < Ng - 1; ++j)
		for (i = 0; i < Ng - 1; ++i)
			Pxpy[i + j + 2] += P.xy(i, j);

	for (i = 2; i < 2 * Ng; ++i)
		/*  M. Boland  sentropy -= Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sentropy -= Pxpy[i] * fast_log10(Pxpy[i] + EPSILON) / LOG10_2;

	return sentropy;
}

/* Entropy */
double GLCMFeature::f_entropy(const SimpleMatrix<double>& P, int Ng)
{
	int i, j;
	double entropy = 0;

	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			entropy += P.xy(i,j) * fast_log10(P.xy(i,j) + EPSILON) / LOG10_2;	// Originally entropy += P[i][j] * log10 (P[i][j] + EPSILON)

	return -entropy;
}

/* Difference Variance */
double GLCMFeature::f_dvar(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy)
{
	int i, j;
	double sum = 0, sum_sqr = 0, var = 0;

	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			Pxpy[abs(i - j)] += P.xy(i, j);

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

	return var;
}

/* Difference Entropy */
double GLCMFeature::f_dentropy(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy) 
{
	int i, j;
	double sum = 0;

	std::fill(Pxpy.begin(), Pxpy.end(), 0.0);

	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			Pxpy[abs(i - j)] += P.xy(i, j);

	for (i = 0; i < Ng; ++i)
		/*    sum += Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sum += Pxpy[i] * fast_log10(Pxpy[i] + EPSILON) / LOG10_2;

	return -sum;
}

double GLCMFeature::f_difference_avg (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px)
{
	double diffAvg = 0.0;

	for (int x = 0; x < Pxmy.size(); x++)
		diffAvg += x * Pxmy[x];

	return diffAvg;
}

void GLCMFeature::calcH (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py)
{
	hx = hy = hxy = hxy1 = hxy2 = 0;

	/* All /log10(2.0) added by M. Boland */

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (int j = 0; j < Ng; j++)
		for (int i = 0; i < Ng; i++)
		{
			auto p = P.xy(i, j);
			px[i] += p;
			py[j] += p;
		}

	for (int j = 0; j < Ng; j++)
	{
		auto pyj = py[j];

		for (int i = 0; i < Ng; i++)
		{
			auto p = P.xy(i, j),
				pxi = px[i];
			auto log_pp = fast_log10(pxi * pyj + EPSILON) /*avoid /LOG10_2 */;
			hxy1 -= p * log_pp;
			hxy2 -= pxi * pyj * log_pp;
			hxy -= p * fast_log10(p + EPSILON) /*avoid /LOG10_2 */;
		}
		hxy1 /= LOG10_2;
		hxy2 /= LOG10_2;
		hxy /= LOG10_2;
	}

	/* Calculate entropies of px and py */
	for (int i = 0; i < Ng; ++i)
	{
		hx -= px[i] * fast_log10(px[i] + EPSILON) /*avoid /LOG10_2 */;
		hy -= py[i] * fast_log10(py[i] + EPSILON) /*avoid /LOG10_2 */;
	}

	hx /= LOG10_2;
	hy /= LOG10_2;
}

double GLCMFeature::f_info_meas_corr1 (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py)
{
	// Calculate the entropies if they aren't available yet
	if (hxy < 0)
		calcH(P, Ng, px, py);

	// Calculate the feature
	double maxHxy = std::max(hx, hy);
	if (maxHxy == 0)
		return(1);
	else
		return (hxy - hxy1) / maxHxy;
}

double GLCMFeature::f_info_meas_corr2 (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py) 
{
	// Calculate the entropies if they aren't available yet
	if (hxy < 0)
		calcH(P, Ng, px, py);

	// Calculate the feature
	return sqrt(fabs(1.0 - exp(-2.0 * (hxy2 - hxy))));
}

double GLCMFeature::f_energy (const SimpleMatrix<double>& P_matrix, int n_levels, std::vector<double>& px)
{
	double energy = 0.0;

	for (int x = 0; x < n_levels; x++) 
		for (int y = 0; y < n_levels; y++) 
		{
			auto p = P_matrix.xy(x,y);
			energy += p*p;
		}

	return energy;
}

double GLCMFeature::f_homogeneity (const SimpleMatrix<double>& P_matrix, int n_levels, std::vector<double>& px)
{
	double homogeneity = 0.0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
			homogeneity += P_matrix.xy(x,y) / (1.0 + (double)std::abs(x - y));

	return homogeneity;
}

