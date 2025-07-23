#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "texture_feature.h"

class D3_GLCM_feature : public FeatureMethod, public TextureFeature
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
		Nyxus::Feature3D::GLCM_ACOR,		// Autocorrelation, IBSI # QWB0
		Nyxus::Feature3D::GLCM_ASM,		// Angular second moment	IBSI # 8ZQL
		Nyxus::Feature3D::GLCM_CLUPROM,	// Cluster prominence, IBSI # AE86
		Nyxus::Feature3D::GLCM_CLUSHADE,	// Cluster shade, IBSI # 7NFM
		Nyxus::Feature3D::GLCM_CLUTEND,	// Cluster tendency, IBSI # DG8W
		Nyxus::Feature3D::GLCM_CONTRAST,	// Contrast, IBSI # ACUI
		Nyxus::Feature3D::GLCM_CORRELATION,	// Correlation, IBSI # NI2N
		Nyxus::Feature3D::GLCM_DIFAVE,	// Difference average, IBSI # TF7R
		Nyxus::Feature3D::GLCM_DIFENTRO,	// Difference entropy, IBSI # NTRS
		Nyxus::Feature3D::GLCM_DIFVAR,	// Difference variance, IBSI # D3YU
		Nyxus::Feature3D::GLCM_DIS,		// Dissimilarity, IBSI # 8S9J
		Nyxus::Feature3D::GLCM_ENERGY,	// Energy
		Nyxus::Feature3D::GLCM_ENTROPY,	// Entropy
		Nyxus::Feature3D::GLCM_HOM1,		// Homogeneity-1 (PyR)
		Nyxus::Feature3D::GLCM_HOM2,		// Homogeneity-2 (PyR)
		Nyxus::Feature3D::GLCM_ID,		// Inv diff, IBSI # IB1Z
		Nyxus::Feature3D::GLCM_IDN,		// Inv diff normalized, IBSI # NDRX
		Nyxus::Feature3D::GLCM_IDM,		// Inv diff mom, IBSI # WF0Z
		Nyxus::Feature3D::GLCM_IDMN,		// Inv diff mom normalized, IBSI # 1QCO
		Nyxus::Feature3D::GLCM_INFOMEAS1,	// Information measure of correlation 1, IBSI # R8DG
		Nyxus::Feature3D::GLCM_INFOMEAS2,	// Information measure of correlation 2, IBSI # JN9H
		Nyxus::Feature3D::GLCM_IV,		// Inv variance, IBSI # E8JP
		Nyxus::Feature3D::GLCM_JAVE,		// Joint average, IBSI # 60VM
		Nyxus::Feature3D::GLCM_JE,		// Joint entropy, IBSI # TU9B
		Nyxus::Feature3D::GLCM_JMAX,		// Joint max (aka PyR max probability), IBSI # GYBY
		Nyxus::Feature3D::GLCM_JVAR,		// Joint var (aka PyR Sum of Squares), IBSI # UR99
		Nyxus::Feature3D::GLCM_SUMAVERAGE,	// Sum average, IBSI # ZGXS
		Nyxus::Feature3D::GLCM_SUMENTROPY,	// Sum entropy, IBSI # P6QZ
		Nyxus::Feature3D::GLCM_SUMVARIANCE,	// Sum variance, IBSI # OEEB
		Nyxus::Feature3D::GLCM_VARIANCE,	// Variance
		Nyxus::Feature3D::GLCM_ASM_AVE,
		Nyxus::Feature3D::GLCM_ACOR_AVE,
		Nyxus::Feature3D::GLCM_CLUPROM_AVE,
		Nyxus::Feature3D::GLCM_CLUSHADE_AVE,
		Nyxus::Feature3D::GLCM_CLUTEND_AVE,
		Nyxus::Feature3D::GLCM_CONTRAST_AVE,
		Nyxus::Feature3D::GLCM_CORRELATION_AVE,
		Nyxus::Feature3D::GLCM_DIFAVE_AVE,
		Nyxus::Feature3D::GLCM_DIFENTRO_AVE,
		Nyxus::Feature3D::GLCM_DIFVAR_AVE,
		Nyxus::Feature3D::GLCM_DIS_AVE,
		Nyxus::Feature3D::GLCM_ENERGY_AVE,
		Nyxus::Feature3D::GLCM_ENTROPY_AVE,
		Nyxus::Feature3D::GLCM_HOM1_AVE,
		Nyxus::Feature3D::GLCM_ID_AVE,
		Nyxus::Feature3D::GLCM_IDN_AVE,
		Nyxus::Feature3D::GLCM_IDM_AVE,
		Nyxus::Feature3D::GLCM_IDMN_AVE,
		Nyxus::Feature3D::GLCM_IV_AVE,
		Nyxus::Feature3D::GLCM_JAVE_AVE,
		Nyxus::Feature3D::GLCM_JE_AVE,
		Nyxus::Feature3D::GLCM_INFOMEAS1_AVE,
		Nyxus::Feature3D::GLCM_INFOMEAS2_AVE,
		Nyxus::Feature3D::GLCM_VARIANCE_AVE,
		Nyxus::Feature3D::GLCM_JMAX_AVE,
		Nyxus::Feature3D::GLCM_JVAR_AVE,
		Nyxus::Feature3D::GLCM_SUMAVERAGE_AVE,
		Nyxus::Feature3D::GLCM_SUMENTROPY_AVE,
		Nyxus::Feature3D::GLCM_SUMVARIANCE_AVE
	};

	// Features implemented by this class that do not require vector-like angled output. Instead, they are output as a single values
	const constexpr static std::initializer_list<Nyxus::Feature3D> nonAngledFeatures =
	{
		Nyxus::Feature3D::GLCM_ASM_AVE,
		Nyxus::Feature3D::GLCM_ACOR_AVE,
		Nyxus::Feature3D::GLCM_CLUPROM_AVE,
		Nyxus::Feature3D::GLCM_CLUSHADE_AVE,
		Nyxus::Feature3D::GLCM_CLUTEND_AVE,
		Nyxus::Feature3D::GLCM_CONTRAST_AVE,
		Nyxus::Feature3D::GLCM_CORRELATION_AVE,
		Nyxus::Feature3D::GLCM_DIFAVE_AVE,
		Nyxus::Feature3D::GLCM_DIFENTRO_AVE,
		Nyxus::Feature3D::GLCM_DIFVAR_AVE,
		Nyxus::Feature3D::GLCM_DIS_AVE,
		Nyxus::Feature3D::GLCM_ENERGY_AVE,
		Nyxus::Feature3D::GLCM_ENTROPY_AVE,
		Nyxus::Feature3D::GLCM_HOM1_AVE,
		Nyxus::Feature3D::GLCM_ID_AVE,
		Nyxus::Feature3D::GLCM_IDN_AVE,
		Nyxus::Feature3D::GLCM_IDM_AVE,
		Nyxus::Feature3D::GLCM_IDMN_AVE,
		Nyxus::Feature3D::GLCM_IV_AVE,
		Nyxus::Feature3D::GLCM_JAVE_AVE,
		Nyxus::Feature3D::GLCM_JE_AVE,
		Nyxus::Feature3D::GLCM_INFOMEAS1_AVE,
		Nyxus::Feature3D::GLCM_INFOMEAS2_AVE,
		Nyxus::Feature3D::GLCM_VARIANCE_AVE,
		Nyxus::Feature3D::GLCM_JMAX_AVE,
		Nyxus::Feature3D::GLCM_JVAR_AVE,
		Nyxus::Feature3D::GLCM_SUMAVERAGE_AVE,
		Nyxus::Feature3D::GLCM_SUMENTROPY_AVE,
		Nyxus::Feature3D::GLCM_SUMVARIANCE_AVE
	};

	static int offset;	// default value: 1
	static int n_levels;	// default value: 0
	static bool symmetric_glcm;	// default value: false
	static std::vector<int> angles;	// default value: {0,45,90,135} (the supreset)
	double sum_p = 0; // sum of P matrix for normalization

	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled (D3_GLCM_feature::featureset);
	}

	D3_GLCM_feature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void extract(LR& r);

private:

	void Extract_Texture_Features2_NT(int angle, WriteImageMatrix_nontriv& grays, PixIntens min_val, PixIntens max_val);
	void calculateCoocMatAtAngle_NT(
		// out
		SimpleMatrix<double>& matrix,
		// in
		int dx,
		int dy,
		int dz,
		WriteImageMatrix_nontriv& grays,
		PixIntens min_val,
		PixIntens max_val,
		bool normalize);

	void extract_texture_features_at_angle (int angle, const SimpleCube<PixIntens> & grays, PixIntens min_val, PixIntens max_val);

	void calculateCoocMatAtAngle(
		// out
		SimpleMatrix<double>& p_matrix,
		// in
		int dx, 
		int dy, 
		int dz,
		const SimpleCube<PixIntens> & grays,
		PixIntens min_val,
		PixIntens max_val);

	void calculatePxpmy();
	void calculate_by_row_mean();

	static inline int cast_to_range(PixIntens orig_I, PixIntens min_orig_I, PixIntens max_orig_I, int min_target_I, int max_target_I)
	{
		int target_I = (int)(double(orig_I - min_orig_I) / double(max_orig_I - min_orig_I) * double(max_target_I - min_target_I) + min_target_I);
		return target_I;
	}

	double f_asm(const SimpleMatrix<double>& P_matrix);
	double f_contrast(const SimpleMatrix<double>& P_matix);
	double f_corr();
	double f_var(const SimpleMatrix<double>& P_matrix);
	double f_idm();
	double f_savg();
	double f_sentropy();
	double f_svar(const SimpleMatrix<double>& P_matrix, double sum_entropy);
	double f_entropy(const SimpleMatrix<double>& P_matrix);
	double f_dvar(const SimpleMatrix<double>& P_matrix);
	double f_dentropy(const SimpleMatrix<double>& P_matrix);
	double f_GLCM_ACOR(const SimpleMatrix<double>& P_matrix);
	double f_GLCM_CLUPROM();
	double f_GLCM_CLUSHADE();
	double f_GLCM_CLUTEND();
	double f_GLCM_DIS(const SimpleMatrix<double>& P_matrix);
	double f_GLCM_HOM2(const SimpleMatrix<double>& P_matrix);
	double f_GLCM_IDMN();
	double f_GLCM_ID();
	double f_GLCM_IDN();
	double f_GLCM_IV();
	double f_GLCM_JAVE();
	double f_GLCM_JE(const SimpleMatrix<double>& P_matrix);
	double f_GLCM_JMAX(const SimpleMatrix<double>& P_matrix);
	double f_GLCM_JVAR(const SimpleMatrix<double>& P_matrix, double mean_x);

	double calc_ave(const std::vector<double>& angled_feature_vals);

	using AngledFeatures = std::vector<double>;
	void copyfvals(AngledFeatures& dst, const AngledFeatures& src);

	// Angled feature values. Each vector contains 1 to 4 values corresponding to angles 0, 45, 90, and 135 degrees
	std::vector<double> fvals_ASM,
		fvals_acor,
		fvals_cluprom,
		fvals_clushade,
		fvals_clutend,
		fvals_contrast,
		fvals_correlation,
		fvals_diff_avg,
		fvals_diff_var,
		fvals_diff_entropy,
		fvals_dis,
		fvals_energy,
		fvals_entropy,
		fvals_homo,
		fvals_hom2,
		fvals_id,
		fvals_idn,
		fvals_IDM,
		fvals_idmn,
		fvals_meas_corr1,
		fvals_meas_corr2,
		fvals_iv,
		fvals_jave,
		fvals_je,
		fvals_jmax,
		fvals_jvar,
		fvals_sum_avg,
		fvals_sum_var,
		fvals_sum_entropy,
		fvals_variance;

	void clear_result_buffers();

	double hx = -1, hy = -1, hxy = -1, hxy1 = -1, hxy2 = -1;	// Entropies for f12/f13_icorr calculation
	void calcH(const SimpleMatrix<double>& P_matrix, std::vector<double>& px, std::vector<double>& py);
	double f_info_meas_corr1(const SimpleMatrix<double>& P_matrix);
	double f_info_meas_corr2(const SimpleMatrix<double>& P_matrix);

	double f_energy(const SimpleMatrix<double>& P_matrix);
	double f_inv_difference(const SimpleMatrix<double>& P_matrix);
	double f_homogeneity();
	double f_difference_avg();

	const double LOG10_2 = 0.30102999566;	// precalculated log 2 base 10
	SimpleMatrix<double> P_matrix;
	std::vector<PixIntens> I;	// unique sorted intensities
	std::vector<double> Pxpy,
		Pxmy,
		kValuesSum,	// intensities x+y
		kValuesDiff;	// intensities x-y
	double by_row_mean;
	const double EPSILON = 0.000000001;
};

