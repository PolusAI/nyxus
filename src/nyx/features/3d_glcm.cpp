#include <numeric>
#include "3d_glcm.h"
#include "../helpers/helpers.h"
#include "../environment.h"

using namespace Nyxus;

int D3_GLCM_feature::offset = 1;
int D3_GLCM_feature::n_levels = 0;
bool D3_GLCM_feature::symmetric_glcm = false;
std::vector<int> D3_GLCM_feature::angles = { 0, 45, 90, 135 };

D3_GLCM_feature::D3_GLCM_feature() : FeatureMethod("D3_GLCM_feature")
{
	provide_features (D3_GLCM_feature::featureset);
}

void D3_GLCM_feature::calculate (LR& r)
{
	// clear the feature values buffers
	clear_result_buffers();

	int w = r.aux_image_cube.width(),
		h = r.aux_image_cube.height(),
		d = r.aux_image_cube.depth();

	// -- grey-bin intensities
	SimpleCube<PixIntens> D;
	D.allocate (w, h, d);

	auto greyInfo = Nyxus::theEnvironment.get_coarse_gray_depth();
	auto greyInfo_localFeature = D3_GLCM_feature::n_levels;
	if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
		greyInfo = greyInfo_localFeature;
	if (Nyxus::theEnvironment.ibsi_compliance)
		greyInfo = 0;

	bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

	// calculate features for all the directions
	for (auto a : D3_GLCM_feature::angles)
		extract_texture_features_at_angle (a, D, r.aux_min, r.aux_max);
}

void D3_GLCM_feature::clear_result_buffers()
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

void D3_GLCM_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}		// Not supporting the online mode for this feature method

void D3_GLCM_feature::copyfvals(AngledFeatures& dst, const AngledFeatures& src)
{
	dst.assign(src.begin(), src.end());
}

void D3_GLCM_feature::save_value(std::vector<std::vector<double>>& fvals)
{
	copyfvals(fvals[(int)Feature3D::GLCM_ASM], fvals_ASM);

	copyfvals(fvals[(int)Feature3D::GLCM_ACOR], fvals_acor);
	copyfvals(fvals[(int)Feature3D::GLCM_ACOR], fvals_acor);

	copyfvals(fvals[(int)Feature3D::GLCM_CLUPROM], fvals_cluprom);
	copyfvals(fvals[(int)Feature3D::GLCM_CLUSHADE], fvals_clushade);
	copyfvals(fvals[(int)Feature3D::GLCM_CLUTEND], fvals_clutend);
	copyfvals(fvals[(int)Feature3D::GLCM_CONTRAST], fvals_contrast);
	copyfvals(fvals[(int)Feature3D::GLCM_CORRELATION], fvals_correlation);
	copyfvals(fvals[(int)Feature3D::GLCM_DIFAVE], fvals_diff_avg);
	copyfvals(fvals[(int)Feature3D::GLCM_DIFVAR], fvals_diff_var);
	copyfvals(fvals[(int)Feature3D::GLCM_DIFENTRO], fvals_diff_entropy);
	copyfvals(fvals[(int)Feature3D::GLCM_DIS], fvals_dis);
	copyfvals(fvals[(int)Feature3D::GLCM_ENERGY], fvals_energy);
	copyfvals(fvals[(int)Feature3D::GLCM_ENTROPY], fvals_entropy);
	copyfvals(fvals[(int)Feature3D::GLCM_HOM1], fvals_homo);
	copyfvals(fvals[(int)Feature3D::GLCM_HOM2], fvals_hom2);
	copyfvals(fvals[(int)Feature3D::GLCM_ID], fvals_id);
	copyfvals(fvals[(int)Feature3D::GLCM_IDN], fvals_idn);
	copyfvals(fvals[(int)Feature3D::GLCM_IDM], fvals_IDM);
	copyfvals(fvals[(int)Feature3D::GLCM_IDMN], fvals_idmn);
	copyfvals(fvals[(int)Feature3D::GLCM_INFOMEAS1], fvals_meas_corr1);
	copyfvals(fvals[(int)Feature3D::GLCM_INFOMEAS2], fvals_meas_corr2);
	copyfvals(fvals[(int)Feature3D::GLCM_IV], fvals_iv);
	copyfvals(fvals[(int)Feature3D::GLCM_JAVE], fvals_jave);
	copyfvals(fvals[(int)Feature3D::GLCM_JE], fvals_je);
	copyfvals(fvals[(int)Feature3D::GLCM_JMAX], fvals_jmax);
	copyfvals(fvals[(int)Feature3D::GLCM_JVAR], fvals_jvar);
	copyfvals(fvals[(int)Feature3D::GLCM_SUMAVERAGE], fvals_sum_avg);
	copyfvals(fvals[(int)Feature3D::GLCM_SUMVARIANCE], fvals_sum_var);
	copyfvals(fvals[(int)Feature3D::GLCM_SUMENTROPY], fvals_sum_entropy);
	copyfvals(fvals[(int)Feature3D::GLCM_VARIANCE], fvals_variance);

	fvals[(int)Feature3D::GLCM_ASM_AVE][0] = calc_ave(fvals_ASM);
	fvals[(int)Feature3D::GLCM_ACOR_AVE][0] = calc_ave(fvals_acor);
	fvals[(int)Feature3D::GLCM_CLUPROM_AVE][0] = calc_ave(fvals_cluprom);
	fvals[(int)Feature3D::GLCM_CLUSHADE_AVE][0] = calc_ave(fvals_clushade);
	fvals[(int)Feature3D::GLCM_CLUTEND_AVE][0] = calc_ave(fvals_clutend);
	fvals[(int)Feature3D::GLCM_CONTRAST_AVE][0] = calc_ave(fvals_contrast);
	fvals[(int)Feature3D::GLCM_CORRELATION_AVE][0] = calc_ave(fvals_correlation);
	fvals[(int)Feature3D::GLCM_DIFAVE_AVE][0] = calc_ave(fvals_diff_avg);
	fvals[(int)Feature3D::GLCM_DIFVAR_AVE][0] = calc_ave(fvals_diff_var);
	fvals[(int)Feature3D::GLCM_DIFENTRO_AVE][0] = calc_ave(fvals_diff_entropy);
	fvals[(int)Feature3D::GLCM_DIS_AVE][0] = calc_ave(fvals_dis);
	fvals[(int)Feature3D::GLCM_ENERGY_AVE][0] = calc_ave(fvals_energy);
	fvals[(int)Feature3D::GLCM_ENTROPY_AVE][0] = calc_ave(fvals_entropy);
	fvals[(int)Feature3D::GLCM_HOM1_AVE][0] = calc_ave(fvals_homo);
	fvals[(int)Feature3D::GLCM_ID_AVE][0] = calc_ave(fvals_id);
	fvals[(int)Feature3D::GLCM_IDN_AVE][0] = calc_ave(fvals_idn);
	fvals[(int)Feature3D::GLCM_IDM_AVE][0] = calc_ave(fvals_IDM);
	fvals[(int)Feature3D::GLCM_IDMN_AVE][0] = calc_ave(fvals_idmn);
	fvals[(int)Feature3D::GLCM_IV_AVE][0] = calc_ave(fvals_iv);
	fvals[(int)Feature3D::GLCM_JAVE_AVE][0] = calc_ave(fvals_jave);
	fvals[(int)Feature3D::GLCM_JE_AVE][0] = calc_ave(fvals_je);
	fvals[(int)Feature3D::GLCM_INFOMEAS1_AVE][0] = calc_ave(fvals_meas_corr1);
	fvals[(int)Feature3D::GLCM_INFOMEAS2_AVE][0] = calc_ave(fvals_meas_corr2);
	fvals[(int)Feature3D::GLCM_VARIANCE_AVE][0] = calc_ave(fvals_variance);
	fvals[(int)Feature3D::GLCM_JMAX_AVE][0] = calc_ave(fvals_jmax);
	fvals[(int)Feature3D::GLCM_JVAR_AVE][0] = calc_ave(fvals_jvar);
	fvals[(int)Feature3D::GLCM_SUMAVERAGE_AVE][0] = calc_ave(fvals_sum_avg);
	fvals[(int)Feature3D::GLCM_SUMVARIANCE_AVE][0] = calc_ave(fvals_sum_var);
	fvals[(int)Feature3D::GLCM_SUMENTROPY_AVE][0] = calc_ave(fvals_sum_entropy);
	fvals[(int)Feature3D::GLCM_VARIANCE_AVE][0] = calc_ave(fvals_variance);
}

void D3_GLCM_feature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Skip calculation in case of bad data

		//
		// !!! We need to smart-select the greyInfo rather than just theEnvironment.get_coarse_gray_depth()
		//
		auto binnedMin = bin_pixel(r.aux_min, r.aux_min, r.aux_max, theEnvironment.get_coarse_gray_depth());
		auto binnedMax = bin_pixel(r.aux_max, r.aux_min, r.aux_max, theEnvironment.get_coarse_gray_depth());
		if (binnedMin == binnedMax)
		{
			auto w = theEnvironment.resultOptions.noval();		// safe NAN

			// assign it to each angled feature value 
			auto n = angles.size();
			r.fvals[(int)Feature3D::GLCM_ASM].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_ACOR].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_CLUPROM].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_CLUSHADE].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_CLUTEND].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_CONTRAST].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_CORRELATION].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_DIFAVE].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_DIFENTRO].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_DIFVAR].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_DIS].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_ENERGY].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_ENTROPY].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_HOM1].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_HOM2].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_IDMN].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_ID].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_IDN].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_INFOMEAS1].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_INFOMEAS2].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_IDM].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_IV].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_JAVE].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_JE].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_JMAX].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_JVAR].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_SUMAVERAGE].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_SUMENTROPY].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_SUMVARIANCE].assign(n, w);
			r.fvals[(int)Feature3D::GLCM_VARIANCE].assign(n, w);

			r.fvals[(int)Feature3D::GLCM_ASM_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_ACOR_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_CLUPROM_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_CLUSHADE_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_CLUTEND_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_CONTRAST_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_CORRELATION_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_DIFAVE_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_DIFENTRO_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_DIFVAR_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_DIS_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_ENERGY_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_ENTROPY_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_HOM1_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_ID_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_IDN_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_IDM_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_IDMN_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_IV_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_JAVE_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_JE_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_INFOMEAS1_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_INFOMEAS2_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_JMAX_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_JVAR_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_SUMAVERAGE_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_SUMENTROPY_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_SUMVARIANCE_AVE][0] =
				r.fvals[(int)Feature3D::GLCM_VARIANCE_AVE][0] = w;

			// No need to calculate features for this ROI
			continue;
		}

		D3_GLCM_feature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void D3_GLCM_feature::extract_texture_features_at_angle (int angle, const SimpleCube<PixIntens> & binned_greys, PixIntens min_val, PixIntens max_val)
{
	for (int kdz = 0; kdz <= 1; kdz++)
	{
		int dz = offset * kdz;

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

		calculateCoocMatAtAngle (P_matrix, dx, dy, dz, binned_greys, min_val, max_val);

		// Blank cooc-matrix? -- no point to use it, assign each feature value '0' and return.
		if (sum_p == 0)
		{
			auto _ = theEnvironment.resultOptions.noval(); // safe NAN
			fvals_ASM.push_back(_);
			fvals_acor.push_back(_);
			fvals_cluprom.push_back(_);
			fvals_clushade.push_back(_);
			fvals_clutend.push_back(_);
			fvals_contrast.push_back(_);
			fvals_correlation.push_back(_);
			fvals_diff_avg.push_back(_);
			fvals_diff_var.push_back(_);
			fvals_diff_entropy.push_back(_);
			fvals_dis.push_back(_);
			fvals_energy.push_back(_);
			fvals_entropy.push_back(_);
			fvals_homo.push_back(_);
			fvals_hom2.push_back(_);
			fvals_id.push_back(_);
			fvals_idn.push_back(_);
			fvals_IDM.push_back(_);
			fvals_idmn.push_back(_);
			fvals_meas_corr1.push_back(_);
			fvals_meas_corr2.push_back(_);
			fvals_iv.push_back(_);
			fvals_jave.push_back(_);
			fvals_je.push_back(_);
			fvals_jmax.push_back(_);
			fvals_jvar.push_back(_);
			fvals_sum_avg.push_back(_);
			fvals_sum_var.push_back(_);
			fvals_sum_entropy.push_back(_);
			fvals_variance.push_back(_);
			return;
		}

		// Output - 'Pxpy' (meaning: 'x plus y') and 'Pxmy' (meaning: 'x minus y')
		calculatePxpmy();

		// Calculate by-row mean. (Output - 'by_row_mean')
		calculate_by_row_mean();

		// Compute Haralick statistics 
		double f;
		f = theFeatureSet.isEnabled(Feature3D::GLCM_ASM) ? f_asm(P_matrix) : 0.0;
		fvals_ASM.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_CONTRAST) ? f_contrast(P_matrix) : 0.0;
		fvals_contrast.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_CORRELATION) ? f_corr() : 0.0;
		fvals_correlation.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_ENERGY) ? f_energy(P_matrix) : 0.0;
		fvals_energy.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_HOM1) ? f_homogeneity() : 0.0;
		fvals_homo.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_VARIANCE) ? f_var(P_matrix) : 0.0;
		fvals_variance.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_IDM) ? f_idm() : 0.0;
		fvals_IDM.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_SUMAVERAGE) ? f_savg() : 0.0;
		fvals_sum_avg.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_SUMENTROPY) ? f_sentropy() : 0.0;
		fvals_sum_entropy.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_ENTROPY) ? f_entropy(P_matrix) : 0.0;
		fvals_entropy.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_DIFVAR) ? f_dvar(P_matrix) : 0.0;
		fvals_diff_var.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_DIFENTRO) ? f_dentropy(P_matrix) : 0.0;
		fvals_diff_entropy.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_DIFAVE) ? f_difference_avg() : 0.0;
		fvals_diff_avg.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_INFOMEAS1) ? f_info_meas_corr1(P_matrix) : 0.0;
		fvals_meas_corr1.push_back(f);

		f = theFeatureSet.isEnabled(Feature3D::GLCM_INFOMEAS2) ? f_info_meas_corr2(P_matrix) : 0.0;
		fvals_meas_corr2.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_ACOR) ? 0 : f_GLCM_ACOR(P_matrix);
		fvals_acor.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_CLUPROM) ? 0 : f_GLCM_CLUPROM();
		fvals_cluprom.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_CLUSHADE) ? 0 : f_GLCM_CLUSHADE();
		fvals_clushade.push_back(f);

		// 'cluster tendency' is equivalent to 'sum variance', so calculate it once
		f = (theFeatureSet.isEnabled(Feature3D::GLCM_CLUTEND) || theFeatureSet.isEnabled(Feature3D::GLCM_SUMVARIANCE)) ? f_GLCM_CLUTEND() : 0.0;
		fvals_clutend.push_back(f);
		fvals_sum_var.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_DIS) ? 0 : f_GLCM_DIS(P_matrix);
		fvals_dis.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_HOM2) ? 0 : f_GLCM_HOM2(P_matrix);
		fvals_hom2.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_IDMN) ? 0 : f_GLCM_IDMN();
		fvals_idmn.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_ID) ? 0 : f_GLCM_ID();
		fvals_id.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_IDN) ? 0 : f_GLCM_IDN();
		fvals_idn.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_IV) ? 0 : f_GLCM_IV();
		fvals_iv.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_JAVE) ? 0 : f_GLCM_JAVE();
		fvals_jave.push_back(f);
		auto jave = f;

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_JE) ? 0 : f_GLCM_JE(P_matrix);
		fvals_je.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_JMAX) ? 0 : f_GLCM_JMAX(P_matrix);
		fvals_jmax.push_back(f);

		f = !theFeatureSet.isEnabled(Feature3D::GLCM_JVAR) ? 0 : f_GLCM_JVAR(P_matrix, jave);
		fvals_jvar.push_back(f);
	}
}

// prerequisite: needs previously grey-binned ROI. No grey binning is done in this method
void D3_GLCM_feature::calculateCoocMatAtAngle(
	// out
	SimpleMatrix<double>& GLCM,
	// in
	int dx,
	int dy,
	int dz,
	const SimpleCube<PixIntens> & D,		// grey-binned ROI
	PixIntens grays_min_val,
	PixIntens grays_max_val)
{
	int w = D.width(),
		h = D.height(),
		d = D.depth();

	// we need the following info about how 'D' was grey-binned
	auto greyInfo = Nyxus::theEnvironment.get_coarse_gray_depth();
	auto greyInfo_localFeature = D3_GLCM_feature::n_levels;
	if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
		greyInfo = greyInfo_localFeature;
	if (Nyxus::theEnvironment.ibsi_compliance)
		greyInfo = 0;

	// allocate the cooc and intensities matrices
	if (radiomics_grey_binning(greyInfo))
	{
		// unique intensities
		std::unordered_set<PixIntens> U (D.begin(), D.end());
		U.erase(0);	// discard intensity '0'

		I.assign(U.begin(), U.end());
		std::sort(I.begin(), I.end());

		GLCM.allocate((int)I.size(), (int)I.size());
	}
	else
		if (matlab_grey_binning(greyInfo))
		{
			auto n_matlab_levels = greyInfo;
			I.resize(n_matlab_levels);
			for (int i = 0; i < n_matlab_levels; i++)
				I[i] = i + 1;

			GLCM.allocate(n_matlab_levels, n_matlab_levels);
		}
		else
		{
			// IBSI
			auto ibsi_levels_it = std::max_element (D.begin(), D.end());
			auto n_ibsi_levels = *ibsi_levels_it;

			I.resize(n_ibsi_levels);
			for (int i = 0; i < n_ibsi_levels; i++)
				I[i] = i + 1;

			GLCM.allocate(n_ibsi_levels, n_ibsi_levels);
		}
	if (GLCM.capacity() < GLCM.width() * GLCM.height())
	{
		std::cerr << "Error: cannot allocate a " << GLCM.width() << "x" << GLCM.height() << " matrix \n";
		throw std::runtime_error("Allocation error in GLCMFeature::calculateCoocMatAtAngle(): requested " + std::to_string(GLCM.width() * GLCM.height()) + " but received " + std::to_string(GLCM.capacity()));
	}

	std::fill (GLCM.begin(), GLCM.end(), 0.0);

	for (int zslice=0; zslice < d; zslice++)
	for (int row = 0; row < h; row++)
		for (int col = 0; col < w; col++)
		{
			if (D.safe(zslice+dz, row+dy, col+dx))
			{
				// Raw intensities
				PixIntens lvl_b = D.zyx(zslice, row, col),
					lvl_a = D.zyx (zslice + dz, row + dy, col + dx);

				// Skip 0-intensity pixels (usually out of mask pixels)
				if (ibsi_grey_binning(greyInfo))
					if (lvl_a == 0 || lvl_b == 0)
						continue;

				// 0-based grey tone indices, hence '-1'
				int a = lvl_a,
					b = lvl_b;

				// raw intensities need to be modified for different grey binning paradigms (Matlab, PyRadiomics, IBSI)
				if (radiomics_grey_binning(greyInfo))
				{
					// skip zeroes
					if (a == 0 || b == 0)
						continue;

					// index of 'a'
					auto lower = std::lower_bound(I.begin(), I.end(), a);	// enjoy sorted vector 'I'
					a = int(lower - I.begin());	// intensity index in array of unique intensities 'I'
					// index of 'b'
					lower = std::lower_bound(I.begin(), I.end(), b);	// enjoy sorted vector 'I'
					b = int(lower - I.begin());	// intensity index in array of unique intensities 'I'
				}
				else // matlab and IBSI
				{
					a = a - 1;
					b = b - 1;
				}

				(GLCM.xy(a, b))++;

				// Radiomics GLCM is symmetric, Matlab one is not
				if (D3_GLCM_feature::symmetric_glcm || radiomics_grey_binning(greyInfo) || ibsi_grey_binning(greyInfo))
					(GLCM.xy(b, a))++;
			}
		}

	// Calculate sum of GLCM for feature calculations
	sum_p = 0;
	for (int i = 0; i < GLCM.width(); ++i)
		for (int j = 0; j < GLCM.height(); ++j)
			sum_p += GLCM.xy(i, j);
}

void D3_GLCM_feature::calculatePxpmy()
{
	int Ng = I.size();

	Pxpy.resize(2 * Ng);
	std::fill(Pxpy.begin(), Pxpy.end(), 0);

	Pxmy.resize(Ng);
	std::fill(Pxmy.begin(), Pxmy.end(), 0);

	kValuesSum.resize(2 * Ng);
	std::fill(kValuesSum.begin(), kValuesSum.end(), 0);

	kValuesDiff.resize(Ng, 0);
	std::fill(kValuesDiff.begin(), kValuesDiff.end(), 0);

	for (int x = 0; x < Ng; x++)
		for (int y = 0; y < Ng; y++)
		{
			// normalize with '/sum_p' per IBSI
			Pxpy[x + y] += P_matrix.yx(x, y) / sum_p;
			Pxmy[std::abs(x - y)] += P_matrix.yx(x, y) / sum_p;

			double xval = I[x], yval = I[y];
			kValuesSum[x + y] = xval + yval;
			kValuesDiff[std::abs(x - y)] = std::abs(xval - yval);
		}
}

void D3_GLCM_feature::calculate_by_row_mean()
{
	int n_levels = I.size();

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
	{
		double ival = I[i];
		by_row_mean += px[i] * ival;
	}
}

/* Angular Second Moment
*
* The angular second-moment feature (ASM) f1 is a measure of homogeneity
* of the image. In a homogeneous image, there are very few dominant
* gray-tone transitions. Hence the P matrix for such an image will have
* fewer entries of large magnitude.
*/
double D3_GLCM_feature::f_asm(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

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
double D3_GLCM_feature::f_contrast(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	double sum = 0;

	for (int i = 0; i < Ng; i++)
	{
		double ival = I[i];
		for (int j = 0; j < Ng; j++)
		{
			double jval = I[j];
			double d = ival - jval;
			sum += P.yx(i, j) * d * d;
		}
	}

	return sum / sum_p;
}

/* Correlation
*
* This correlation feature is a measure of gray-tone linear-dependencies
* in the image.
*
* Returns marginal totals 'px' and their mean 'meanx'
*/
double D3_GLCM_feature::f_corr()
{
	auto Ng = P_matrix.width();

	// radiomics
	double mr = 0;
	for (int c = 0; c < Ng; c++)
		for (int r = 0; r < Ng; r++)
			mr += P_matrix.yx(r, c) * double(I[r]);

	mr /= sum_p;

	double mc = 0;
	for (int c = 0; c < Ng; c++)
		for (int r = 0; r < Ng; r++)
			mc += P_matrix.yx(r, c) * double(I[c]);

	mc /= sum_p;

	double s2r = 0;
	for (int c = 0; c < Ng; c++)
		for (int r = 0; r < Ng; r++)
		{
			double dr = double(I[r]) - mr;
			s2r += P_matrix.yx(r, c) / sum_p * dr * dr;
		}
	double sr = sqrt(s2r);

	double s2c = 0;
	for (int c = 0; c < Ng; c++)
		for (int r = 0; r < Ng; r++)
		{
			double dc = double(I[c]) - mc;
			s2c += P_matrix.yx(r, c) / sum_p * dc * dc;
		}
	double sc = sqrt(s2c);

	double tmp1 = 0;
	for (int c = 0; c < Ng; c++)
		for (int r = 0; r < Ng; r++)
			tmp1 += (double(I[r]) - mr) * (double(I[c]) - mc) * P_matrix.yx(r, c) / sum_p;
	double cor = tmp1 / (sr * sc);

	return cor;
}

// Variance aka 'Sum of Squares' aka IBSI 'Joint Variance'
double D3_GLCM_feature::f_var(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	double mean = 0, var = 0;

	/*- Corrected by James Darrell McCauley, 16 Aug 1991
	*  calculates the mean intensity level instead of the mean of
	*  cooccurrence matrix elements
	*/
	for (int r = 0; r < Ng; r++)
	{
		double sum_x = 0;
		for (int c = 0; c < Ng; c++)
			sum_x += P.yx(r, c);
		mean += sum_x * I[r];
	}

	mean /= sum_p;

	for (int r = 0; r < Ng; r++)
	{
		double rval = I[r];
		double d = rval - mean;
		for (int c = 0; c < Ng; c++)
			var += d * d * P.yx(r, c);
	}

	return var / sum_p;
}

/* Inverse Difference Moment */
double D3_GLCM_feature::f_idm()
{
	int n_levels = I.size();

	double idm = 0;

	for (int k = 0; k < n_levels; ++k) {
		idm += Pxmy[k] / (1 + (k * k));
	}

	return idm;
}

/* Sum Average */
double D3_GLCM_feature::f_savg()
{
	// \textit{sum average} = \sum^{2N_g}_{k=2}{p_{x+y}(k)k}

	double f = 0;
	auto n = Pxpy.size();

	for (int i = 0; i < n; i++)
		f += kValuesSum[i] * Pxpy[i];

	return f;
}

/* Sum Entropy */
double D3_GLCM_feature::f_sentropy()
{
	double f = 0;
	auto n = Pxpy.size();

	for (int k = 0; k < n; k++)
	{
		double p = Pxpy[k];
		f += p * fast_log10(p + EPSILON) / LOG10_2;
	}

	return -f;
}

/* Entropy */
double D3_GLCM_feature::f_entropy(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	int i, j;
	double entropy = 0;

	for (j = 0; j < Ng; j++)
		for (i = 0; i < Ng; i++)
			entropy += P.xy(i, j) * fast_log10(P.xy(i, j) + EPSILON) / LOG10_2;

	return -entropy;
}

/* Difference Variance */
double D3_GLCM_feature::f_dvar(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	int n_levels = I.size();

	size_t n = Pxmy.size();

	double diffAvg = f_difference_avg();
	std::vector<double> var(n, 0);

	for (int x = 0; x < n; x++)
	{
		for (int k = 0; k < n; k++)
		{
			var[k] += pow((k - diffAvg), 2) * Pxmy[k];
		}
	}

	double sum = 0;
	for (int x = 0; x < n; x++)
		sum += var[x];

	return sum / double(n);
}

/* Difference Entropy */
double D3_GLCM_feature::f_dentropy(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	int n_levels = I.size();

	std::vector<double> entropy(n_levels, 0);
	double sum = 0;

	for (int k = 0; k < n_levels; ++k) {
		if (Pxmy[k] == 0) continue; // avoid NaN from log2 (note that Pxmy will never be negative)
		sum += Pxmy[k] * fast_log10(Pxmy[k] + EPSILON) / LOG10_2;
	}

	return -sum;
}

double D3_GLCM_feature::f_difference_avg()
{
	double f = 0;
	auto n = Pxmy.size();

	for (int i = 0; i < n; i++)
		f += kValuesDiff[i] * Pxmy[i];

	return f;
}

void D3_GLCM_feature::calcH(const SimpleMatrix<double>& P, std::vector<double>& px, std::vector<double>& py)
{
	auto Ng = P.width();

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

double D3_GLCM_feature::f_info_meas_corr1(const SimpleMatrix<double>& P)
{

	auto Ng = P.width();

	double HX = 0, HXY = 0, HXY1 = 0;

	std::vector<double> px, py;
	px.resize(Ng);
	py.resize(Ng);
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

double D3_GLCM_feature::f_info_meas_corr2(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	double HX = 0, HXY = 0, HXY2 = 0;

	std::vector<double> px, py;
	px.resize(Ng);
	py.resize(Ng);
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

double D3_GLCM_feature::f_energy(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	int n_levels = I.size();

	double energy = 0.0;

	for (int x = 0; x < Ng; x++)
		for (int y = 0; y < Ng; y++)
		{
			auto p = P.xy(x, y) / sum_p;
			energy += p * p;
		}

	return energy;
}

double D3_GLCM_feature::f_homogeneity()
{
	int Ng = I.size();

	double homogeneity = 0.0;

	for (int r = 0; r < Ng; r++)
		for (int c = 0; c < Ng; c++)
			homogeneity += P_matrix.yx(r, c) / sum_p / (1.0 + (double)std::abs(r - c));

	return homogeneity;
}

double D3_GLCM_feature::f_GLCM_ACOR(const SimpleMatrix<double>& P)
{
	auto Ng = P.width();

	// autocorrelation = \sum^{N_g}_{i=1} sum^{N_g}_{j=1} p(i,j) i j

	double f = 0;

	for (int x = 0; x < Ng; x++)
	{
		double xval = (double)I[x];
		for (int y = 0; y < Ng; y++)
		{
			double yval = (double)I[y];
			f += P.xy(x, y) * xval * yval;
		}
	}
	f = f / sum_p;
	return f;
}

//
// Argument 'mean_x' is calculated by f_corr()
//
double D3_GLCM_feature::f_GLCM_CLUPROM()
{
	int n_levels = I.size();

	// cluster prominence = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^4 P(i,j)

	double f = 0;

	for (int r = 0; r < n_levels; r++)
	{
		double rval = I[r];
		for (int c = 0; c < n_levels; c++)
		{
			double cval = I[c];
			double m = rval + cval - by_row_mean - by_row_mean;
			f += m * m * m * m * P_matrix.yx(r, c) / sum_p;
		}
	}

	return f;
}

double D3_GLCM_feature::f_GLCM_CLUSHADE()
{
	int n_levels = I.size();

	// cluster shade = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^3 p(i,j)

	double f = 0;

	for (int r = 0; r < n_levels; r++)
	{
		double rval = I[r];
		for (int c = 0; c < n_levels; c++)
		{
			double cval = I[c];
			double m = rval + cval - by_row_mean - by_row_mean;
			f += m * m * m * P_matrix.yx(r, c) / sum_p;
		}
	}

	return f;
}

double D3_GLCM_feature::f_GLCM_CLUTEND()
{
	int n_levels = I.size();

	double f = 0;

	//
	//	if (theEnvironment.ibsi_compliance)
	//		// According to IBSI, feature "cluster tendency" is equivalent to "sum variance"
	//		f = f_svar(P_matrix, theEnv.NovalOptions.noval(), this->Pxpy);
	//	else

		// Calculate it the radiomics way: cluster tendency = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^2 p(i,j)
	for (int x = 0; x < n_levels; x++)
	{
		double xval = I[x];
		for (int y = 0; y < n_levels; y++)
		{
			double yval = I[y];
			double m = xval + yval - by_row_mean * 2.0;
			f += m * m * P_matrix.xy(x, y) / sum_p;
		}
	}

	return f;
}

double D3_GLCM_feature::f_GLCM_DIS(const SimpleMatrix<double>& P_matrix)
{
	int n_levels = I.size();

	// dissimilarity = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} |i-j| p(i,j)

	double f = 0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
			f += std::fabs(double(x + 1) - double(y + 1)) * P_matrix.xy(x, y) / sum_p;

	return f;
}

double D3_GLCM_feature::f_GLCM_HOM2(const SimpleMatrix<double>& P_matrix)
{
	int n_levels = I.size();

	// homogeneity2 = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} \frac {p(i,j)} {1+|i-j|^2}

	double hom2 = 0.0;

	for (int x = 0; x < n_levels; x++)
		for (int y = 0; y < n_levels; y++)
			hom2 += P_matrix.xy(x, y) / (1.0 + (double)std::abs(x - y) * (double)std::abs(x - y));

	return hom2;
}

double D3_GLCM_feature::f_GLCM_IDMN()
{
	auto Ng = P_matrix.width();

	// IDMN = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+\frac{k^2}{N_g^2}}

	double f = 0;

	double Ng2 = double(Ng) * double(Ng);
	for (int k = 0; k < Ng; k++)
		f += Pxmy[k] / (1.0 + (double(k) * double(k)) / Ng2);

	return f;
}

double D3_GLCM_feature::f_GLCM_ID()
{
	// inverse difference = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+k}

	auto Ng = I.size();
	double f = 0;

	for (int k = 0; k < Ng; k++)
		f += Pxmy[k] / (1.0 + double(k));

	return f;
}

double D3_GLCM_feature::f_GLCM_IDN()
{
	// inverse difference normalized = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+\frac{k}{N_g}}

	auto Ng = I.size();

	double f = 0;

	for (int k = 0; k < Ng; k++)
		f += Pxmy[k] / (1.0 + double(k) / Ng);

	return f;
}

double D3_GLCM_feature::f_GLCM_IV()
{
	// inverse variance = \sum^{N_g-1}_{k=1} \frac {p_{x-y}(k)} {k^2}

	double f = 0;
	auto n = Pxmy.size();

	for (int k = 1; k < n; k++)
	{
		//f += Pxmy[k] / (double(I[k]) * double(I[k]));
		double kval = kValuesDiff[k];
		f += Pxmy[k] / (kval * kval);
	}

	return f;
}

double D3_GLCM_feature::f_GLCM_JAVE()
{
	// joint average = \mu_x = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} p(i,j) i

	auto Ng = I.size();
	double f = 0;

	for (int i = 0; i < Ng; i++)
	{
		double ival = I[i];
		for (int j = 0; j < Ng; j++)
			f += P_matrix.yx(i, j) * ival;
	}
	return f / sum_p;
}

double D3_GLCM_feature::f_GLCM_JE(const SimpleMatrix<double>& P)
{
	int Ng = I.size();

	// jointentropy = - \sum^{N_g}_{i=1} sum^{N_g}_{j=1} p(i,j) \log_2 ( p(i,j) + \epsilon )

	double f = 0.0;

	for (int x = 0; x < Ng; x++)
		for (int y = 0; y < Ng; y++)
		{
			double p = P_matrix.xy(x, y) / sum_p;
			f += p * fast_log10(p + EPSILON) / LOG10_2;
		}

	return -f;
}

double D3_GLCM_feature::f_GLCM_JMAX(const SimpleMatrix<double>& P_matrix)
{
	int n_levels = I.size();

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

double D3_GLCM_feature::f_GLCM_JVAR(const SimpleMatrix<double>& P_matrix, double joint_ave)
{
	int n_levels = I.size();

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
double D3_GLCM_feature::calc_ave(const std::vector<double>& afv)
{
	if (afv.empty())
		return 0;

	double n = static_cast<double> (afv.size()),
		ave = std::reduce(afv.begin(), afv.end()) / n;

	return ave;
}

void D3_GLCM_feature::reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_GLCM_feature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

/*static*/ void D3_GLCM_feature::extract (LR& r)
{
	D3_GLCM_feature f;
	f.calculate(r);
	f.save_value(r.fvals);
}


