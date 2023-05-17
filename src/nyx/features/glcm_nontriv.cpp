#include "../environment.h"
#include "glcm.h"
#include "image_matrix_nontriv.h"

void GLCMFeature::osized_calculate(LR& r, ImageLoader& imloader)
{
	// Clear the feature values buffers
	clear_result_buffers();

	// Prepare the tone-binned image matrix
	WriteImageMatrix_nontriv G("GLCMFeature-osized_calculate-G", r.label);
	G.allocate_from_cloud(r.raw_pixels_NT, r.aabb, false);

	for (auto a : angles)
		Extract_Texture_Features2_NT(a, G, r.aux_min, r.aux_max);
}

void GLCMFeature::Extract_Texture_Features2_NT (int angle, WriteImageMatrix_nontriv& grays, PixIntens min_val, PixIntens max_val)
{
	int nrows = grays.get_height();
	int ncols = grays.get_width();

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

	calculateCoocMatAtAngle_NT (P_matrix, dx, dy, grays, min_val, max_val, false);

	// Zero all feature values for blank ROI
	if (sum_p == 0) 
	{
		double f = 0.0;
		fvals_ASM.push_back(f);
		fvals_contrast.push_back(f);
		fvals_correlation.push_back(f);
		fvals_energy.push_back(f);
		fvals_homo.push_back(f);
		fvals_variance.push_back(f);
		fvals_IDM.push_back(f);
		fvals_sum_avg.push_back(f);
		fvals_sum_entropy.push_back(f);
		fvals_sum_var.push_back(f);
		fvals_entropy.push_back(f);
		fvals_diff_var.push_back(f);
		fvals_diff_entropy.push_back(f);
		fvals_diff_avg.push_back(f);
		fvals_meas_corr1.push_back(f);
		fvals_meas_corr2.push_back(f);
		return;
	}

	calculatePxpmy();

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

	f = !theFeatureSet.isEnabled(GLCM_CLUPROM) ? 0 : f_GLCM_CLUPROM(P_matrix, n_levels, mean_x, mean_x);	// mean_x == mean_y due to symmetric GLCM
	fvals_cluprom.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_CLUSHADE) ? 0 : f_GLCM_CLUSHADE(P_matrix, n_levels, mean_x, mean_x);
	fvals_clushade.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_CLUTEND) ? 0 : f_GLCM_CLUTEND(P_matrix, n_levels, mean_x, mean_x);
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

	f = !theFeatureSet.isEnabled(GLCM_JAVE) ? 0 : f_GLCM_JAVE(P_matrix, n_levels, mean_x);
	fvals_jave.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_JE) ? 0 : f_GLCM_JE(P_matrix, n_levels);
	fvals_je.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_JMAX) ? 0 : f_GLCM_JMAX(P_matrix, n_levels);
	fvals_jmax.push_back(f);

	f = !theFeatureSet.isEnabled(GLCM_JVAR) ? 0 : f_GLCM_JVAR(P_matrix, n_levels, mean_x);
	fvals_jvar.push_back(f);
}

void GLCMFeature::calculateCoocMatAtAngle_NT(
	// out
	SimpleMatrix<double>& matrix,
	// in
	int dx,
	int dy,
	WriteImageMatrix_nontriv& grays,
	PixIntens min_val,
	PixIntens max_val,
	bool normalize)
{
	matrix.allocate(n_levels, n_levels);
	matrix.fill(0.0);

	int d = GLCMFeature::offset;
	int count = 0;	// normalizing factor 

	int rows = grays.get_height(),
		cols = grays.get_width();

	PixIntens range = max_val - min_val;

	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++)
		{
			if (row + dy >= 0 && row + dy < rows && col + dx >= 0 && col + dx < cols && grays.yx(row + dy, col + dx))
			{
				// Raw intensities
				auto raw_lvl_y = grays.yx(row, col),
					raw_lvl_x = grays.yx(row + dy, col + dx);

				// Skip non-informative pixels
				if (raw_lvl_x == 0 || raw_lvl_y == 0)
					continue;

				int x = raw_lvl_x - 1,
					y = raw_lvl_y - 1;

				// Cast intensities on the 1-n_levels scale
				if (Environment::ibsi_compliance == false)
				{
					x = GLCMFeature::cast_to_range(raw_lvl_x, min_val, max_val, 1, GLCMFeature::n_levels) - 1;
					y = GLCMFeature::cast_to_range(raw_lvl_y, min_val, max_val, 1, GLCMFeature::n_levels) - 1;
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

