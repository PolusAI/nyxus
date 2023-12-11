#include <numeric>
#include "glcm.h"
#include "../helpers/helpers.h"
#include "../environment.h"

int GLCMFeature::offset = 1;
int GLCMFeature::n_levels = 8;
std::vector<int> GLCMFeature::angles = { 0, 45, 90, 135 };

GLCMFeature::GLCMFeature() : FeatureMethod("GLCMFeature")
{
	provide_features(GLCMFeature::featureset);
}

void GLCMFeature::calculate(LR& r)
{
	// Clear the feature values buffers
	clear_result_buffers();

	// Calculate features for all the directions
	for (auto a : angles)
		Extract_Texture_Features2(a, r.aux_image_matrix, r.aux_min, r.aux_max);
}

void GLCMFeature::clear_result_buffers()
{
	fvals_ASM.clear();
	fvals_acor.clear();
	fvals_cluprom.clear();
	fvals_clushade.clear();
	fvals_clutend.clear();
	fvals_contrast.clear();
	fvals_correlation.clear();
	fvals_diff_avg.clear();
	fvals_diff_var.clear();
	fvals_diff_entropy.clear();
	fvals_dis.clear();
	fvals_energy.clear();
	fvals_entropy.clear();
	fvals_homo.clear();
	fvals_hom2.clear();
	fvals_id.clear();
	fvals_idn.clear();
	fvals_IDM.clear();
	fvals_idmn.clear();
	fvals_meas_corr1.clear();
	fvals_meas_corr2.clear();
	fvals_iv.clear();
	fvals_jave.clear();
	fvals_je.clear();
	fvals_jmax.clear();
	fvals_jvar.clear();
	fvals_sum_avg.clear();
	fvals_sum_var.clear();
	fvals_sum_entropy.clear();
	fvals_variance.clear();
}

void GLCMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}		// Not supporting the online mode for this feature method

void GLCMFeature::copyfvals(AngledFeatures& dst, const AngledFeatures& src)
{
	dst.assign(src.begin(), src.end());
}

void GLCMFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	copyfvals(fvals[GLCM_ASM], fvals_ASM);

	copyfvals(fvals[GLCM_ACOR], fvals_acor);
	copyfvals(fvals[GLCM_ACOR], fvals_acor);

	copyfvals(fvals[GLCM_CLUPROM], fvals_cluprom);
	copyfvals(fvals[GLCM_CLUSHADE], fvals_clushade);
	copyfvals(fvals[GLCM_CLUTEND], fvals_clutend);
	copyfvals(fvals[GLCM_CONTRAST], fvals_contrast);
	copyfvals(fvals[GLCM_CORRELATION], fvals_correlation);
	copyfvals(fvals[GLCM_DIFAVE], fvals_diff_avg);
	copyfvals(fvals[GLCM_DIFVAR], fvals_diff_var);
	copyfvals(fvals[GLCM_DIFENTRO], fvals_diff_entropy);
	copyfvals(fvals[GLCM_DIS], fvals_dis);
	copyfvals(fvals[GLCM_ENERGY], fvals_energy);
	copyfvals(fvals[GLCM_ENTROPY], fvals_entropy);
	copyfvals(fvals[GLCM_HOM1], fvals_homo);
	copyfvals(fvals[GLCM_HOM2], fvals_hom2);
	copyfvals(fvals[GLCM_ID], fvals_id);
	copyfvals(fvals[GLCM_IDN], fvals_idn);
	copyfvals(fvals[GLCM_IDM], fvals_IDM);
	copyfvals(fvals[GLCM_IDMN], fvals_idmn);
	copyfvals(fvals[GLCM_INFOMEAS1], fvals_meas_corr1);
	copyfvals(fvals[GLCM_INFOMEAS2], fvals_meas_corr2);
	copyfvals(fvals[GLCM_IV], fvals_iv);
	copyfvals(fvals[GLCM_JAVE], fvals_jave);
	copyfvals(fvals[GLCM_JE], fvals_je);
	copyfvals(fvals[GLCM_JMAX], fvals_jmax);
	copyfvals(fvals[GLCM_JVAR], fvals_jvar);
	copyfvals(fvals[GLCM_SUMAVERAGE], fvals_sum_avg);
	copyfvals(fvals[GLCM_SUMVARIANCE], fvals_sum_var);
	copyfvals(fvals[GLCM_SUMENTROPY], fvals_sum_entropy);
	copyfvals(fvals[GLCM_VARIANCE], fvals_variance);
	fvals[GLCM_ASM_AVE][0] = calc_ave(fvals_ASM);
	fvals[GLCM_ACOR_AVE][0] = calc_ave(fvals_acor);
	fvals[GLCM_CLUPROM_AVE][0] = calc_ave(fvals_cluprom);
	fvals[GLCM_CLUSHADE_AVE][0] = calc_ave(fvals_clushade);
	fvals[GLCM_CLUTEND_AVE][0] = calc_ave(fvals_clutend);
	fvals[GLCM_CONTRAST_AVE][0] = calc_ave(fvals_contrast);
	fvals[GLCM_CORRELATION_AVE][0] = calc_ave(fvals_correlation);
	fvals[GLCM_DIFAVE_AVE][0] = calc_ave(fvals_diff_avg);
	fvals[GLCM_DIFVAR_AVE][0] = calc_ave(fvals_diff_var);
	fvals[GLCM_DIFENTRO_AVE][0] = calc_ave(fvals_diff_entropy);
	fvals[GLCM_DIS_AVE][0] = calc_ave(fvals_dis);
	fvals[GLCM_ENERGY_AVE][0] = calc_ave(fvals_energy);
	fvals[GLCM_ENTROPY_AVE][0] = calc_ave(fvals_entropy);
	fvals[GLCM_HOM1_AVE][0] = calc_ave(fvals_homo);
	fvals[GLCM_ID_AVE][0] = calc_ave(fvals_id);
	fvals[GLCM_IDN_AVE][0] = calc_ave(fvals_idn);
	fvals[GLCM_IDM_AVE][0] = calc_ave(fvals_IDM);
	fvals[GLCM_IDMN_AVE][0] = calc_ave(fvals_idmn);
	fvals[GLCM_IV_AVE][0] = calc_ave(fvals_iv);
	fvals[GLCM_JAVE_AVE][0] = calc_ave(fvals_jave);
	fvals[GLCM_JE_AVE][0] = calc_ave(fvals_je);
	fvals[GLCM_INFOMEAS1_AVE][0] = calc_ave(fvals_meas_corr1);
	fvals[GLCM_INFOMEAS2_AVE][0] = calc_ave(fvals_meas_corr2);
	fvals[GLCM_VARIANCE_AVE][0] = calc_ave(fvals_variance);
	fvals[GLCM_JMAX_AVE][0] = calc_ave(fvals_jmax);
	fvals[GLCM_JVAR_AVE][0] = calc_ave(fvals_jvar);
	fvals[GLCM_SUMAVERAGE_AVE][0] = calc_ave(fvals_sum_avg);
	fvals[GLCM_SUMVARIANCE_AVE][0] = calc_ave(fvals_sum_var);
	fvals[GLCM_SUMENTROPY_AVE][0] = calc_ave(fvals_sum_entropy);
	fvals[GLCM_VARIANCE_AVE][0] = calc_ave(fvals_variance);
}

void GLCMFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Skip calculation in case of bad data
		auto minI = Nyxus::to_grayscale(r.aux_min, r.aux_min, r.aux_max - r.aux_min, theEnvironment.get_coarse_gray_depth()),
			maxI = Nyxus::to_grayscale(r.aux_max, r.aux_min, r.aux_max - r.aux_min, theEnvironment.get_coarse_gray_depth());
		if (minI == maxI)
		{
			// Zero out each angled feature value 
			auto n = angles.size();
			r.fvals[GLCM_ASM].assign(n, 0);
			r.fvals[GLCM_ACOR].assign(n, 0);
			r.fvals[GLCM_CLUPROM].assign(n, 0);
			r.fvals[GLCM_CLUSHADE].assign(n, 0);
			r.fvals[GLCM_CLUTEND].assign(n, 0);
			r.fvals[GLCM_CONTRAST].assign(n, 0);
			r.fvals[GLCM_CONTRAST_AVE][0] = 0;
			r.fvals[GLCM_CORRELATION].assign(n, 0);
			r.fvals[GLCM_CORRELATION_AVE][0] = 0;
			r.fvals[GLCM_DIFAVE].assign(n, 0);
			r.fvals[GLCM_DIFAVE_AVE][0] = 0;
			r.fvals[GLCM_DIFENTRO].assign(n, 0);
			r.fvals[GLCM_DIFENTRO][0] = 0;
			r.fvals[GLCM_DIFVAR].assign(n, 0);
			r.fvals[GLCM_DIFVAR_AVE][0] = 0;
			r.fvals[GLCM_DIS].assign(n, 0);
			r.fvals[GLCM_ENERGY].assign(n, 0);
			r.fvals[GLCM_ENERGY_AVE][0] = 0;
			r.fvals[GLCM_ENTROPY].assign(n, 0);
			r.fvals[GLCM_ENTROPY_AVE][0] = 0;
			r.fvals[GLCM_HOM1].assign(n, 0);
			r.fvals[GLCM_HOM1_AVE][0] = 0;
			r.fvals[GLCM_HOM2].assign(n, 0);
			r.fvals[GLCM_IDMN].assign(n, 0);
			r.fvals[GLCM_ID].assign(n, 0);
			r.fvals[GLCM_IDN].assign(n, 0);
			r.fvals[GLCM_INFOMEAS1].assign(n, 0);
			r.fvals[GLCM_INFOMEAS2].assign(n, 0);
			r.fvals[GLCM_IDM].assign(n, 0);
			r.fvals[GLCM_IDM_AVE][0] = 0;
			r.fvals[GLCM_IV].assign(n, 0);
			r.fvals[GLCM_JAVE].assign(n, 0);
			r.fvals[GLCM_JE].assign(n, 0);
			r.fvals[GLCM_JMAX].assign(n, 0);
			r.fvals[GLCM_JVAR].assign(n, 0);
			r.fvals[GLCM_SUMAVERAGE].assign(n, 0);
			r.fvals[GLCM_SUMAVERAGE_AVE][0] = 0;
			r.fvals[GLCM_SUMENTROPY].assign(n, 0);
			r.fvals[GLCM_SUMENTROPY_AVE][0] = 0;
			r.fvals[GLCM_SUMVARIANCE].assign(n, 0);
			r.fvals[GLCM_SUMVARIANCE][0] = 0;
			r.fvals[GLCM_VARIANCE].assign(n, 0);
			r.fvals[GLCM_VARIANCE_AVE][0] = 0;
			// No need to calculate features for this ROI
			continue;
		}

		GLCMFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void GLCMFeature::Extract_Texture_Features2(int angle, const ImageMatrix& grays, PixIntens min_val, PixIntens max_val)
{
	int nrows = grays.height;
	int ncols = grays.width;

	if (Environment::ibsi_compliance) {
		n_levels = *std::max_element(std::begin(grays.ReadablePixels()), std::end(grays.ReadablePixels()));
	}

	// Allocate Px and Py vectors
	std::vector<double> Px(n_levels * 2),
		Py(n_levels);

	// Mean of marginal totals of GLCM
	double mean_x;

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

	calculateCoocMatAtAngle(P_matrix, dx, dy, grays, min_val, max_val, false);

	// In a special case of blank ROI's collocation matrix at given angle, assign each feature value '0'
	if (sum_p == 0)
	{
		fvals_ASM.push_back(0);
		fvals_acor.push_back(0);
		fvals_cluprom.push_back(0);
		fvals_clushade.push_back(0);
		fvals_clutend.push_back(0);
		fvals_contrast.push_back(0);
		fvals_correlation.push_back(0);
		fvals_diff_avg.push_back(0);
		fvals_diff_var.push_back(0);
		fvals_diff_entropy.push_back(0);
		fvals_dis.push_back(0);
		fvals_energy.push_back(0);
		fvals_entropy.push_back(0);
		fvals_homo.push_back(0);
		fvals_hom2.push_back(0);
		fvals_id.push_back(0);
		fvals_idn.push_back(0);
		fvals_IDM.push_back(0);
		fvals_idmn.push_back(0);
		fvals_meas_corr1.push_back(0);
		fvals_meas_corr2.push_back(0);
		fvals_iv.push_back(0);
		fvals_jave.push_back(0);
		fvals_je.push_back(0);
		fvals_jmax.push_back(0);
		fvals_jvar.push_back(0);
		fvals_sum_avg.push_back(0);
		fvals_sum_var.push_back(0);
		fvals_sum_entropy.push_back(0);
		fvals_variance.push_back(0);

		return;
	}

	// ROI's collocation matrix is not blank, we're good to calculate texture features
	calculatePxpmy();

	// Calculate by-row mean
	calculate_by_row_mean();

	// Compute Haralick statistics 
	double f;
	f = theFeatureSet.isEnabled(GLCM_ASM) ? f_asm(P_matrix, n_levels) : 0.0;
	fvals_ASM.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_CONTRAST) ? f_contrast(P_matrix, n_levels) : 0.0;
	fvals_contrast.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_CORRELATION) ? f_corr(P_matrix, n_levels, Px, mean_x) : 0.0;
	fvals_correlation.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_ENERGY) ? f_energy(P_matrix, n_levels, Px) : 0.0;
	fvals_energy.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_HOM1) ? f_homogeneity(P_matrix, n_levels, Px) : 0.0;
	fvals_homo.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_VARIANCE) ? f_var(P_matrix, n_levels) : 0.0;
	fvals_variance.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_IDM) ? f_idm(P_matrix, n_levels) : 0.0;
	fvals_IDM.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_SUMAVERAGE) ? f_savg(P_matrix, n_levels, Px) : 0.0;
	fvals_sum_avg.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_SUMENTROPY) ? f_sentropy(P_matrix, n_levels, Px) : 0.0;
	fvals_sum_entropy.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_SUMVARIANCE) ? f_svar(P_matrix, n_levels, f, Px) : 0.0;
	fvals_sum_var.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_ENTROPY) ? f_entropy(P_matrix, n_levels) : 0.0;
	fvals_entropy.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_DIFVAR) ? f_dvar(P_matrix, n_levels, Px) : 0.0;
	fvals_diff_var.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_DIFENTRO) ? f_dentropy(P_matrix, n_levels, Px) : 0.0;
	fvals_diff_entropy.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_DIFAVE) ? f_difference_avg(P_matrix, n_levels, Px) : 0.0;
	fvals_diff_avg.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_INFOMEAS1) ? f_info_meas_corr1(P_matrix, n_levels, Px, Py) : 0.0;
	fvals_meas_corr1.push_back(f);

	f = theFeatureSet.isEnabled(GLCM_INFOMEAS2) ? f_info_meas_corr2(P_matrix, n_levels, Px, Py) : 0.0;
	fvals_meas_corr2.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_ACOR) ? 0 : f_GLCM_ACOR(P_matrix, n_levels);
	fvals_acor.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_CLUPROM) ? 0 : f_GLCM_CLUPROM();
	fvals_cluprom.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_CLUSHADE) ? 0 : f_GLCM_CLUSHADE();
	fvals_clushade.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_CLUTEND) ? 0 : f_GLCM_CLUTEND();
	fvals_clutend.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_DIS) ? 0 : f_GLCM_DIS(P_matrix, n_levels);
	fvals_dis.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_HOM2) ? 0 : f_GLCM_HOM2(P_matrix, n_levels);
	fvals_hom2.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_IDMN) ? 0 : f_GLCM_IDMN(P_matrix, n_levels);
	fvals_idmn.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_ID) ? 0 : f_GLCM_ID(P_matrix, n_levels);
	fvals_id.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_IDN) ? 0 : f_GLCM_IDN(P_matrix, n_levels);
	fvals_idn.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_IV) ? 0 : f_GLCM_IV(P_matrix, n_levels);
	fvals_iv.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_JAVE) ? 0 : f_GLCM_JAVE(P_matrix, n_levels);
	fvals_jave.push_back(f);
	auto jave = f;

	f = !theFeatureSet.isEnabled(GLCM_JE) ? 0 : f_GLCM_JE(P_matrix, n_levels);
	fvals_je.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_JMAX) ? 0 : f_GLCM_JMAX(P_matrix, n_levels);
	fvals_jmax.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_JVAR) ? 0 : f_GLCM_JVAR(P_matrix, n_levels, jave);
	fvals_jvar.push_back(f);
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

				int x = raw_lvl_x - 1,
					y = raw_lvl_y - 1;

				// Cast intensities on the 1-n_levels scale
				if (Environment::ibsi_compliance == false)
				{
					x = GLCMFeature::cast_to_range(raw_lvl_x, min_val, max_val, 1, GLCMFeature::n_levels) - 1,
						y = GLCMFeature::cast_to_range(raw_lvl_y, min_val, max_val, 1, GLCMFeature::n_levels) - 1;
				}

				// Increment the symmetric count
				count += 2;
				matrix.xy(y, x)++;
				matrix.xy(x, y)++;
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
	Pxpy.resize(2 * n_levels - 1, 0.0);
	Pxmy.resize(n_levels, 0.0);

	std::fill(Pxpy.begin(), Pxpy.end(), 0.);
	std::fill(Pxmy.begin(), Pxmy.end(), 0.);

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
		{
			Pxpy[x + y] += P_matrix.xy(x, y);
			Pxmy[std::abs(x - y)] += P_matrix.xy(x, y) / sum_p; // normalize matrix from IBSI definition
		}
}

void GLCMFeature::calculate_by_row_mean()
{
	// px[i] is the (i-1)th entry in the marginal probability matrix obtained
	// by summing the rows of p[i][j]
	std::vector<double> px(n_levels);
	for (int j = 0; j < n_levels; j++)
		for (int i = 0; i < n_levels; i++)
			px[i] += P_matrix.xy(i, j) / sum_p;

	// Now calculate the means and standard deviations of px and py */
	// - fix supplied by J. Michael Christensen, 21 Jun 1991 */
	// - further modified by James Darrell McCauley, 16 Aug 1991
	// after realizing that meanx=meany and stddevx=stddevy
	by_row_mean = 0;
	for (int i = 0; i < n_levels; ++i)
		by_row_mean += px[i] * (i + 1);
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
			sum += (P.xy(i, j) / sum_p) * (P.xy(i, j) / sum_p);

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
			sum += P.xy(i, j) / sum_p * (i - j) * (i - j);

	return sum;
}

/* Correlation
*
* This correlation feature is a measure of gray-tone linear-dependencies
* in the image.
*
* Returns marginal totals 'px' and their mean 'meanx'
*/
double GLCMFeature::f_corr(const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, double& meanx)
{
	int i, j;
	double sum_sqrx = 0, tmp;
	meanx = 0;
	double meany = 0, stddevx, stddevy;

	std::fill(px.begin(), px.end(), 0.0);

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (int j = 0; j < Ng; j++)
		for (int i = 0; i < Ng; i++)
			px[i] += P.xy(i, j) / sum_p;


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
			tmp += i * j * (P.xy(i, j) / sum_p);

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
		savg += i * Pxpy[i - 2] / sum_p;

	return savg;
}

/* Sum Variance */
double GLCMFeature::f_svar(const SimpleMatrix<double>& P, int Ng, double S, std::vector<double>& Pxpy)
{
	double var = 0;

	std::vector<double> Px(n_levels * 2);

	double diffAvg = f_savg(P_matrix, n_levels, Px);

	std::vector<double> pxpy(2 * n_levels, 0);

	for (int i = 0; i < n_levels; ++i) {
		for (int j = 0; j < n_levels; ++j) {
			pxpy[i + j] += P.xy(i, j) / sum_p;
		}
	}

	for (int k = 2; k <= 2 * n_levels; ++k) {
		var += (k - diffAvg) * (k - diffAvg) * pxpy[k - 2];
	}

	return var;
}

/* Sum Entropy */
double GLCMFeature::f_sentropy(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy)
{
	double sentropy = 0;

	std::vector<double> pxpy(2 * n_levels, 0);

	for (int i = 0; i < n_levels; ++i) {
		for (int j = 0; j < n_levels; ++j) {
			pxpy[i + j] += P.xy(i, j) / sum_p;
		}
	}

	for (int k = 2; k <= 2 * n_levels; ++k) {

		if (Pxpy[k - 2] == 0) continue;
		sentropy += pxpy[k - 2] * fast_log10(pxpy[k - 2] + EPSILON) / LOG10_2;
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
			entropy += P.xy(i, j) * fast_log10(P.xy(i, j) + EPSILON) / LOG10_2;	// Originally entropy += P[i][j] * log10 (P[i][j] + EPSILON)

	return -entropy;
}

/* Difference Variance */
double GLCMFeature::f_dvar(const SimpleMatrix<double>& P, int Ng, std::vector<double>& Pxpy)
{
	std::vector<double> Px(n_levels * 2);
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

	return sum / Pxmy.size();
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

double GLCMFeature::f_difference_avg(const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px)
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

	return sum / Pxmy.size();
}

void GLCMFeature::calcH(const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py)
{
	hx = hy = hxy = hxy1 = hxy2 = 0;

	for (int j = 0; j < Ng; j++)
		for (int i = 0; i < Ng; i++)
		{
			auto p = P.xy(i, j) / sum_p;
			px[i] += p;
			py[j] += p;
		}

	for (int j = 0; j < Ng; j++)
	{
		auto pyj = py[j];

		for (int i = 0; i < Ng; i++)
		{
			auto p = P.xy(i, j) / sum_p,
				pxi = px[i];
			double log_pp;
			if (pxi == 0 || pyj == 0) {
				log_pp = 0;
			}
			else {
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


		if (py[i] > 0)
			hy += py[i] * fast_log10(py[i] + EPSILON) / LOG10_2 /*avoid /LOG10_2 */;
	}
}

double GLCMFeature::f_info_meas_corr1(const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py)
{
	double HX = 0, HXY = 0, HXY1 = 0;

	std::fill(px.begin(), px.end(), 0);
	std::fill(py.begin(), py.end(), 0);

	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			px[i] += P.xy(i, j) / sum_p;
			py[j] += P.xy(i, j) / sum_p;
		}
	}

	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			HXY += P.xy(i, j) / sum_p * fast_log10(P.xy(i, j) / sum_p + EPSILON) / LOG10_2;
			HXY1 += P.xy(i, j) / sum_p * fast_log10(px[i] * py[j] + EPSILON) / LOG10_2;

		}
	}

	for (int i = 0; i < Ng; ++i) {
		HX += px[i] * fast_log10(px[i] + EPSILON) / LOG10_2;

	}

	return (HXY - HXY1) / HX;
}

double GLCMFeature::f_info_meas_corr2(const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, std::vector<double>& py)
{
	double HX = 0, HXY = 0, HXY2 = 0;

	std::fill(px.begin(), px.end(), 0);
	std::fill(py.begin(), py.end(), 0);

	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			px[i] += P.xy(i, j) / sum_p;
			py[j] += P.xy(i, j) / sum_p;
		}
	}

	for (int i = 0; i < Ng; ++i) {
		for (int j = 0; j < Ng; ++j) {
			HXY += P.xy(i, j) / sum_p * fast_log10(P.xy(i, j) / sum_p + EPSILON) / LOG10_2;

			HXY2 += px[i] * py[j] * fast_log10(px[i] * py[j] + EPSILON) / LOG10_2;
		}
	}

	return sqrt(fabs(1 - exp(-2 * (-HXY2 + HXY))));
}

double GLCMFeature::f_energy(const SimpleMatrix<double>& P_matrix, int n_levels, std::vector<double>& px)
{
	double energy = 0.0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
		{
			auto p = P_matrix.xy(x, y);
			energy += p * p;
		}

	return energy;
}

double GLCMFeature::f_homogeneity(const SimpleMatrix<double>& P_matrix, int n_levels, std::vector<double>& px)
{
	double homogeneity = 0.0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
			homogeneity += P_matrix.xy(x, y) / (1.0 + (double)std::abs(x - y));

	return homogeneity;
}

double GLCMFeature::f_GLCM_ACOR(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// autocorrelation = \sum^{N_g}_{i=1} sum^{N_g}_{j=1} p(i,j) i j

	double f = 0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
			f += P_matrix.xy(x, y) / sum_p * double(x + 1) * double(y + 1);

	return f;
}

//
// Argument 'mean_x' is calculated by f_corr()
//
double GLCMFeature::f_GLCM_CLUPROM()
{
	// cluster prominence = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^4 p(i,j)

	double f = 0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
		{
			double m = double(x + 1) + double(y + 1) - by_row_mean * 2.0;
			f += m * m * m * m * P_matrix.xy(x, y) / sum_p;
		}

	return f;
}

double GLCMFeature::f_GLCM_CLUSHADE()
{
	// cluster shade = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^3 p(i,j)

	double f = 0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
		{
			double m = double(x + 1) + double(y + 1) - by_row_mean * 2.0;
			f += m * m * m * P_matrix.xy(x, y) / sum_p;
		}

	return f;
}

double GLCMFeature::f_GLCM_CLUTEND()
{
	double f = 0;

	if (theEnvironment.ibsi_compliance)
		// According to IBSI, feature "cluster tendency" is equivalent to "sum variance"
		f = f_svar(P_matrix, n_levels, -999.999, this->Pxpy);
	else
		// Calculate it the radiomics way: cluster tendency = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^2 p(i,j)
		for (int x = 0; x < n_levels; x++)
			for (int y = 0; y < n_levels; y++)
			{
				double m = double(x + 1) + double(y + 1) - by_row_mean * 2.0;
				f += m * m * P_matrix.xy(x, y) / sum_p;
			}

	return f;
}

double GLCMFeature::f_GLCM_DIS(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// dissimilarity = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} |i-j| p(i,j)

	double f = 0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
			f += std::fabs(double(x + 1) - double(y + 1)) * P_matrix.xy(x, y) / sum_p;

	return f;
}

double GLCMFeature::f_GLCM_HOM2(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// homogeneity2 = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} \frac {p(i,j)} {1+|i-j|^2}

	double hom2 = 0.0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
			hom2 += P_matrix.xy(x, y) / (1.0 + (double)std::abs(x - y) * (double)std::abs(x - y));

	return hom2;
}

double GLCMFeature::f_GLCM_IDMN(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// IDMN = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+\frac{k^2}{N_g^2}}

	double f = 0;

	double Ng2 = double(tone_count) * double(tone_count);
	for (int k = 0; k < tone_count; k++)
		f += Pxmy[k] / (1.0 + (double(k) * double(k)) / Ng2);

	return f;
}

double GLCMFeature::f_GLCM_ID(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// inverse difference = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+k}

	double f = 0;

	for (int k = 0; k < tone_count; k++)
		f += Pxmy[k] / (1.0 + double(k));

	return f;
}

double GLCMFeature::f_GLCM_IDN(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// inverse difference normalized = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+\frac{k}{N_g}}

	double f = 0;

	double Ng = (double)tone_count;
	for (int k = 0; k < tone_count; k++)
		f += Pxmy[k] / (1.0 + double(k) / Ng);

	return f;
}

double GLCMFeature::f_GLCM_IV(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// inverse variance = \sum^{N_g-1}_{k=1} \frac {p_{x-y}(k)} {k^2}

	double f = 0;

	double Ng = (double)tone_count;
	for (int k = 1; k < tone_count; k++)
		f += Pxmy[k] / (double(k) * double(k));

	return f;
}

double GLCMFeature::f_GLCM_JAVE(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// joint average = \mu_x = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} p(i,j) i

	double f = 0;

	for (int x = 0; x < tone_count; x++)
		for (int y = 0; y < tone_count; y++)
			f += P_matrix.xy(x, y) / sum_p * double(x + 1);

	return f;
}

double GLCMFeature::f_GLCM_JE(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// jointentropy = - \sum^{N_g}_{i=1} sum^{N_g}_{j=1} p(i,j) \log_2 ( p(i,j) + \epsilon )

	double f = 0.0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
		{
			double p = P_matrix.xy(x, y) / sum_p;
			f += p * fast_log10(p + EPSILON) / LOG10_2;
		}

	return -f;
}

double GLCMFeature::f_GLCM_JMAX(const SimpleMatrix<double>& P_matrix, int tone_count)
{
	// maximum probability = \max p(i,j)

	double max_p = -1;	// never-probability

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
		{
			double p = P_matrix.xy(x, y) / sum_p;
			max_p = std::max(max_p, p);
		}

	return max_p;
}

double GLCMFeature::f_GLCM_JVAR(const SimpleMatrix<double>& P_matrix, int tone_count, double joint_ave)
{
	// joint variance = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i-\mu_x) ^2 p(i,j)
	//		where \mu_x is the value of joint average feature (IBSI: "Fcm.joint.avg"), 
	//		\mu_x = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} i p(i,j)

	double f = 0;
	for (int x = 0; x < n_levels; x++)
	{
		double d = double(x + 1) - joint_ave,
			d2 = d * d;
		for (int y = 0; y < n_levels; y++)
			f += d2 * P_matrix.xy(x, y) / sum_p;
	}
	return f;
}

// 'afv' is angled feature values
double GLCMFeature::calc_ave(const std::vector<double>& afv)
{
	if (afv.empty())
		return 0;

	double n = static_cast<double> (afv.size()),
		ave = std::reduce(afv.begin(), afv.end()) / n;

	return ave;
}