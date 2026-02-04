#include "../environment.h"
#include "glcm.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

void GLCMFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	// Clear the feature values buffers
	clear_result_buffers();

	int offset = STNGS_GLCM_OFFSET(s);

	// Prepare the tone-binned image matrix
	WriteImageMatrix_nontriv G("GLCMFeature-osized_calculate-G", r.label);
	G.allocate_from_cloud(r.raw_pixels_NT, r.aabb, false);

	for (auto a : angles)
		Extract_Texture_Features2_NT(a, G, offset, r.aux_min, r.aux_max, STNGS_IBSI(s));
}

void GLCMFeature::Extract_Texture_Features2_NT (int angle, WriteImageMatrix_nontriv& grays, int offset, PixIntens min_val, PixIntens max_val, bool ibsi)
{
	int nrows = grays.get_height();
	int ncols = grays.get_width();

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

	calculateCoocMatAtAngle_NT (P_matrix, dx, dy, grays, min_val, max_val, false, ibsi);

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
	fvals_ASM.push_back(f_asm(P_matrix));
	fvals_contrast.push_back(f_contrast(P_matrix));
	fvals_correlation.push_back(f_corr());
	fvals_energy.push_back(f_energy(P_matrix));
	fvals_homo.push_back(f_homogeneity());
	fvals_variance.push_back(f_var(P_matrix));
	fvals_IDM.push_back(f_idm());
	fvals_sum_avg.push_back(f_savg());
	fvals_sum_entropy.push_back(f_sentropy());
	fvals_entropy.push_back(f_entropy(P_matrix));
	fvals_diff_var.push_back(f_dvar(P_matrix));
	fvals_diff_entropy.push_back(f_dentropy(P_matrix));
	fvals_diff_avg.push_back(f_difference_avg());
	fvals_meas_corr1.push_back(f_info_meas_corr1(P_matrix));
	fvals_meas_corr2.push_back(f_info_meas_corr2(P_matrix));
	fvals_acor.push_back(f_GLCM_ACOR(P_matrix));
	fvals_cluprom.push_back(f_GLCM_CLUPROM());
	fvals_clushade.push_back(f_GLCM_CLUSHADE());

	// 'cluster tendency' is equivalent to 'sum variance', so calculate it once
	double clutend = f_GLCM_CLUTEND();
	fvals_clutend.push_back(clutend);
	fvals_sum_var.push_back(clutend);

	fvals_dis.push_back(f_GLCM_DIS(P_matrix));
	fvals_hom2.push_back(f_GLCM_HOM2(P_matrix));
	fvals_idmn.push_back(f_GLCM_IDMN());
	fvals_id.push_back(f_GLCM_ID());
	fvals_idn.push_back(f_GLCM_IDN());
	fvals_iv.push_back(f_GLCM_IV());

	double jave = f_GLCM_JAVE();
	fvals_jave.push_back(jave);

	fvals_je.push_back(f_GLCM_JE(P_matrix));
	fvals_jmax.push_back(f_GLCM_JMAX(P_matrix));
	fvals_jvar.push_back(f_GLCM_JVAR(P_matrix, jave));

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
	bool normalize,
	bool ibsi)
{
	matrix.allocate ((int)I.size(), (int)I.size());
	matrix.fill (0.0);

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
				if (ibsi == false)
				{
					x = GLCMFeature::cast_to_range(raw_lvl_x, min_val, max_val, 1, (int)I.size()) - 1;
					y = GLCMFeature::cast_to_range(raw_lvl_y, min_val, max_val, 1, (int)I.size()) - 1;
				}

				// Increment the symmetric count
				count += 2;
				matrix.xy(y,x)++;
				matrix.xy(x,y)++;
			}
		}

	// calculate sum of P for feature calculations
	sum_p = 0;
	for (int i = 0; i < (int)I.size(); ++i)
		for (int j = 0; j < (int)I.size(); ++j)
			sum_p += matrix.xy(i, j);

	// Normalize the matrix
	if (normalize == false)
		return;

	double realCnt = count;
	for (int i = 0; i < (int)I.size(); i++)
		for (int j = 0; j < (int)I.size(); j++)
			matrix.xy(i, j) /= (realCnt + EPSILON);
}

