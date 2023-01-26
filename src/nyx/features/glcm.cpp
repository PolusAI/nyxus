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
	// Clear the feature values buffers
	clear_result_buffers();

	// Calculate features for all the directions
	for (auto a: angles)
		Extract_Texture_Features2 (a, r.aux_image_matrix, r.aux_min, r.aux_max); 
}

void GLCMFeature::clear_result_buffers()
{
	fvals_ASM.clear();
	fvals_contrast.clear();
	fvals_correlation.clear();
	fvals_energy.clear();
	fvals_homo.clear();
	fvals_variance.clear();
	fvals_IDM.clear();
	fvals_sum_avg.clear();
	fvals_sum_var.clear();
	fvals_sum_entropy.clear();
	fvals_entropy.clear();
	fvals_diff_avg.clear();
	fvals_diff_var.clear();
	fvals_diff_entropy.clear();
	fvals_meas_corr1.clear();
	fvals_meas_corr2.clear();
	fvals_max_corr_coef.clear();
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

	if (Environment::ibsi_compliance) {
		n_levels = *std::max_element(std::begin(grays.ReadablePixels()), std::end(grays.ReadablePixels()));
	}

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

	// Zero all feature values for blank ROI
	if (sum_p == 0) {

		double f = 0.0;

		fvals_ASM.push_back (f);
		fvals_contrast.push_back (f);
		fvals_correlation.push_back (f);
		fvals_energy.push_back (f);
		fvals_homo.push_back (f);
		fvals_variance.push_back (f);
		fvals_IDM.push_back (f);
		fvals_sum_avg.push_back (f);
		fvals_sum_entropy.push_back (f);
		fvals_sum_var.push_back (f);
		fvals_entropy.push_back (f);
		fvals_diff_var.push_back (f);
		fvals_diff_entropy.push_back (f);
		fvals_diff_avg.push_back(f);
		fvals_meas_corr1.push_back (f);
		fvals_meas_corr2.push_back (f);
		fvals_max_corr_coef.push_back (0.0);

		return;
	}

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
	std::fill(matrix.begin(), matrix.end(), 0.);

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

				int x = raw_lvl_x -1, 
					y = raw_lvl_y -1;

				// Cast intensities on the 1-n_levels scale
				if (Environment::ibsi_compliance == false)
				{
					x = GLCMFeature::cast_to_range (raw_lvl_x, min_val, max_val, 1, GLCMFeature::n_levels) -1, 
					y = GLCMFeature::cast_to_range (raw_lvl_y, min_val, max_val, 1, GLCMFeature::n_levels) -1;
				}

				// Increment the symmetric count
				count += 2;	
				matrix.xy(y,x)++;
				matrix.xy(x,y)++;
			}
		}

	// calculate sum of P for feature calculations
	sum_p = 0;
	for (int i = 0; i < n_levels; ++i)
		for (int j = 0; j < n_levels; ++j)
			sum_p += matrix.xy(i, j);

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

	std::fill(Pxpy.begin(), Pxpy.end(), 0.);
	std::fill(Pxmy.begin(), Pxmy.end(), 0.);

	for (int x = 0; x < n_levels; x++) 
		for (int y = 0; y < n_levels; y++) 
		{
			Pxpy[x + y] += P_matrix.xy(x,y);
			Pxmy[std::abs(x - y)] += P_matrix.xy(x,y)/sum_p; // normalize matrix from IBSI definition
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
			sum += (P.xy(i, j)/sum_p) * (P.xy(i, j)/sum_p);

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
			sum += P.xy(i, j)/sum_p * (i - j) * (i - j);

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
			px[i] += P.xy(i, j)/sum_p;


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
			tmp += i * j * (P.xy(i, j)/sum_p);

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

	for (int k = 0; k < n_levels; ++k) {
		idm += Pxmy[k] / (1 + (k * k));
	}

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
	for (int i = 2; i <= (2 * Ng); ++i)
	 	savg += i * Pxpy[i-2]/sum_p;

	return savg;
}

/* Sum Variance */
double GLCMFeature::f_svar(const SimpleMatrix<double>& P, int Ng, double S, std::vector<double>& Pxpy) 
{
	int i, j;
	double var = 0;

	std::vector<double> Px (n_levels * 2);

	double diffAvg = f_savg(P_matrix, n_levels, Px);

	std::vector<double> pxpy(2*n_levels, 0);

	for (int i = 0; i < n_levels; ++i) {
		for (int j = 0; j < n_levels; ++j) {
			pxpy[i+j] += P.xy(i,j)/sum_p;
		}
	}
	
	for(int k = 2; k <= 2 * n_levels; ++k) {
		var += (k-diffAvg) * (k-diffAvg) * pxpy[k-2];
	}

	return var;
}

/* Sum Entropy */
double GLCMFeature::f_sentropy(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy)
{
	int i, j;
	double sentropy = 0;

	std::vector<double> pxpy(2*n_levels, 0);

	for (int i = 0; i < n_levels; ++i) {
		for (int j = 0; j < n_levels; ++j) {
			pxpy[i+j] += P.xy(i,j)/sum_p;
		}
	}
	
	for(int k = 2; k <= 2 * n_levels; ++k) {

		if (Pxpy[k-2] == 0) continue;

		sentropy += pxpy[k-2] * fast_log10(pxpy[k-2] + EPSILON) / LOG10_2;
	}

	return -sentropy;
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
	std::vector<double> Px (n_levels * 2);
	double diffAvg = f_difference_avg(P_matrix, n_levels, Px);
	std::vector<double> var(Pxmy.size(), 0);

	for (int x = 0; x < Pxmy.size(); x++) {
		for (int k = 0; k < Pxmy.size(); k++) {
			var[k] += pow((k - diffAvg), 2) * Pxmy[k];
		}
	}

	double sum = 0;
	for (int x = 0; x < Pxmy.size(); x++)
		sum += var[x];
	
	return sum/Pxmy.size();
}

/* Difference Entropy */
double GLCMFeature::f_dentropy(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy) 
{
	std::vector<double> entropy(n_levels, 0);
	double sum = 0;

	for (int k = 0; k < n_levels; ++k) {
		if (Pxmy[k] == 0) continue; // avoid NaN from log2 (note that Pxmy will never be negative)
		sum += Pxmy[k] * fast_log10(Pxmy[k] + EPSILON) / LOG10_2;
	}

	return -sum;
}

double GLCMFeature::f_difference_avg (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px)
{
	std::vector<double> diffAvg(Pxmy.size(), 0.);

	for (int x = 0; x < Pxmy.size(); x++) {
		for (int k = 0; k < Pxmy.size(); k++) {
			diffAvg[k] += k * Pxmy[k];
		}
	}

	double sum = 0;
	for (int x = 0; x < Pxmy.size(); x++)
		sum += diffAvg[x];
	
	return sum/Pxmy.size();
}

void GLCMFeature::calcH (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py)
{
	hx = hy = hxy = hxy1 = hxy2 = 0;

	for (int j = 0; j < Ng; j++)
		for (int i = 0; i < Ng; i++)
		{
			auto p = P.xy(i, j)/sum_p;
			px[i] += p;
			py[j] += p;
		}

	for (int j = 0; j < Ng; j++)
	{
		auto pyj = py[j];

		for (int i = 0; i < Ng; i++)
		{
			auto p = P.xy(i, j)/sum_p,
				pxi = px[i];
			double log_pp;
			if (pxi == 0 || pyj == 0) {
				log_pp = 0;
			} else {
				log_pp = fast_log10(pxi * pyj + EPSILON) / LOG10_2 /*avoid /LOG10_2 */;
			}
			
			hxy1 += p * log_pp;
			hxy2 += pxi * pyj * log_pp;
			if (p > 0) 
				hxy += p * fast_log10(p + EPSILON) / LOG10_2 /*avoid /LOG10_2 */;
		}
	}

	/* Calculate entropies of px and py */
	for (int i = 0; i < Ng; ++i)
	{	
		if (px[i] > 0) 
			hx += px[i] * fast_log10(px[i] + EPSILON) / LOG10_2 /*avoid /LOG10_2 */;
		
		
		if(py[i] > 0)
			hy += py[i] * fast_log10(py[i] + EPSILON) / LOG10_2 /*avoid /LOG10_2 */;
	}
}

double GLCMFeature::f_info_meas_corr1 (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py)
{
	double HX = 0, HXY = 0, HXY1 = 0;

	std::fill(px.begin(), px.end(), 0);
	std::fill(py.begin(), py.end(), 0);

	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			px[i] += P.xy(i,j)/sum_p;
			py[j] += P.xy(i,j)/sum_p;
		}	
	}
	
	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			HXY += P.xy(i,j)/sum_p * fast_log10(P.xy(i,j)/sum_p + EPSILON) / LOG10_2;
			HXY1 += P.xy(i,j)/sum_p * fast_log10(px[i] * py[j] + EPSILON) / LOG10_2;
			
		}	
	}
	
	for (int i = 0; i < Ng; ++i) {
		HX += px[i] * fast_log10(px[i] + EPSILON) / LOG10_2;
		
	}

	return (HXY - HXY1) / HX;
}

double GLCMFeature::f_info_meas_corr2 (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py) 
{
	double HX = 0, HXY = 0, HXY2 = 0;

	std::fill(px.begin(), px.end(), 0);
	std::fill(py.begin(), py.end(), 0);

	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			px[i] += P.xy(i,j)/sum_p;
			py[j] += P.xy(i,j)/sum_p;
		}	
	}
	
	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			HXY += P.xy(i,j)/sum_p * fast_log10(P.xy(i,j)/sum_p + EPSILON) / LOG10_2;
			
			HXY2 += px[i] * py[j] * fast_log10(px[i] * py[j] + EPSILON) / LOG10_2;
		}	
	}
	
	return sqrt(fabs(1 - exp(-2 * (-HXY2 + HXY))));
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

