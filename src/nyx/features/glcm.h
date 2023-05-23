#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"

#define PGM_MAXMAXVAL 255

/// @brief Gray Level Cooccurrence Matrix(GLCM) features
/// Gray Level Cooccurrence Matrix(GLCM) of size : math:`N_g \times N_g` describes the second - order joint probability
///	function of an image region constrained by the mask and is defined as : math:`\textbf{ P }(i, j | \delta, \theta)`.
/// The :math:`(i, j)^ {\text{ th }}` element of this matrix represents the number of times the combination of
/// levels : math:`i`and :math:`j` occur in two pixels in the image, that are separated by a distance of : math:`\delta`
/// pixels along angle : math:`\theta`.
/// The distance : math:`\delta` from the center voxel is defined as the distance according to the infinity norm.
/// For :math:`\delta = 1`, this results in 2 neighbors for each of 13 angles in 3D(26 - connectivity) and for
/// 	:math:`\delta = 2` a 98 - connectivity(49 unique angles).
/// 

class GLCMFeature: public FeatureMethod
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::AvailableFeatures> featureset =
	{
		GLCM_ASM,		// Angular second moment
		GLCM_ACOR,		// Autocorrelation, IBSI # QWB0
		GLCM_CLUPROM,	// Cluster prominence, IBSI # AE86
		GLCM_CLUSHADE,	// Cluster shade, IBSI # 7NFM
		GLCM_CLUTEND,	// Cluster tendency, IBSI # DG8W
		GLCM_CONTRAST,	// Contrast, IBSI # ACUI
		GLCM_CORRELATION,	// Correlation, IBSI # NI2N
		GLCM_DIFAVE,	// Difference average, IBSI # TF7R
		GLCM_DIFENTRO,	// Difference entropy, IBSI # NTRS
		GLCM_DIFVAR,	// Difference variance, IBSI # D3YU
		GLCM_DIS,		// Dissimilarity, IBSI # 8S9J
		GLCM_ENERGY,	// Energy
		GLCM_ENTROPY,	// Entropy
		GLCM_HOM1,		// Homogeneity-1 (PyR)
		GLCM_HOM2,		// Homogeneity-2 (PyR)
		GLCM_IDM,		// Inv diff mom, IBSI # WF0Z
		GLCM_IDMN,		// Inv diff mom normalized, IBSI # 1QCO
		GLCM_ID,		// Inv diff, IBSI # IB1Z
		GLCM_IDN,		// Inv diff normalized, IBSI # NDRX
		GLCM_INFOMEAS1,	// Information measure of correlation 1, IBSI # R8DG
		GLCM_INFOMEAS2,	// Information measure of correlation 2, IBSI # JN9H
		GLCM_IV,		// Inv variance, IBSI # E8JP
		GLCM_JAVE,		// Joint average, IBSI # 60VM
		GLCM_JE,		// Joint entropy, IBSI # TU9B
		GLCM_JMAX,		// Joint max (aka PyR max probability), IBSI # GYBY
		GLCM_JVAR,		// Joint var (aka PyR Sum of Squares), IBSI # UR99
		GLCM_SUMAVERAGE,	// Sum average, IBSI # ZGXS
		GLCM_SUMENTROPY,	// Sum entropy, IBSI # P6QZ
		GLCM_SUMVARIANCE,	// Sum variance, IBSI # OEEB
		GLCM_VARIANCE
	};

	static int offset;	// default value: 1
	static int n_levels;	// default value: 8
	double sum_p = 0; // sum of P matrix for normalization
	static std::vector<int> angles;	// default value: {0,45,90,135} (the supreset)

	static bool required (const FeatureSet& fs)
	{
		return fs.anyEnabled (GLCMFeature::featureset);
	}

	GLCMFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:

	void Extract_Texture_Features2_NT(int angle, WriteImageMatrix_nontriv& grays, PixIntens min_val, PixIntens max_val);
	void calculateCoocMatAtAngle_NT(
		// out
		SimpleMatrix<double>& matrix,
		// in
		int dx,
		int dy,
		WriteImageMatrix_nontriv& grays,
		PixIntens min_val,
		PixIntens max_val,
		bool normalize);

	void Extract_Texture_Features(
		int distance,
		int angle,
		const SimpleMatrix<uint8_t>& grays);	// 'grays' is 0-255 grays 
	void Extract_Texture_Features2 (int angle, const ImageMatrix& grays, PixIntens min_val, PixIntens max_val);

	void calculate_normalized_graytone_matrix (SimpleMatrix<uint8_t>& G, int minI, int maxI, const ImageMatrix& Im);
	void calculate_normalized_graytone_matrix (OOR_ReadMatrix& G, int minI, int maxI, const ImageMatrix& Im);

	void calculateCoocMatAtAngle (
		// out
		SimpleMatrix<double>& p_matrix,
		// in
		int dx, int dy,
		const ImageMatrix& grays,
		PixIntens min_val,
		PixIntens max_val, 
		bool normalize);

	void calculatePxpmy();

	static inline int cast_to_range(PixIntens orig_I, PixIntens min_orig_I, PixIntens max_orig_I, int min_target_I, int max_target_I)
	{
		int target_I = (int)(double(orig_I - min_orig_I) / double(max_orig_I - min_orig_I) * double(max_target_I - min_target_I) + min_target_I);
		return target_I;
	}

	double f_asm (const SimpleMatrix<double>& P_matrix, int tone_count);	
	double f_contrast (const SimpleMatrix<double>& P_matix, int tone_count);	
	double f_corr (const SimpleMatrix<double>& P, int Ng, std::vector<double>& px, double& meanx);
	double f_var (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_idm (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_savg (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_sentropy (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_svar (const SimpleMatrix<double>& P_matrix, int tone_count, double sum_entropy, std::vector<double>& px);
	double f_entropy (const SimpleMatrix<double>& P_matrix, int tone_count);	
	double f_dvar (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_dentropy (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_GLCM_ACOR (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_CLUPROM (const SimpleMatrix<double>& P_matrix, int tone_count, double mean_x, double mean_y);
	double f_GLCM_CLUSHADE (const SimpleMatrix<double>& P_matrix, int tone_count, double mean_x, double mean_y);
	double f_GLCM_CLUTEND (const SimpleMatrix<double>& P_matrix, int tone_count, double mean_x, double mean_y);
	double f_GLCM_DIS (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_HOM2 (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_IDMN (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_ID (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_IDN (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_IV (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_JAVE (const SimpleMatrix<double>& P_matrix, int tone_count, double mean_x);
	double f_GLCM_JE (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_JMAX (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_GLCM_JVAR (const SimpleMatrix<double>& P_matrix, int tone_count, double mean_x);

	using AngledFeatures = std::vector<double>;
	void copyfvals(AngledFeatures& dst, const AngledFeatures& src);

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
	void calcH (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px, std::vector<double>& py);
	double f_info_meas_corr1 (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px, std::vector<double>& py);
	double f_info_meas_corr2 (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px, std::vector<double>& py);

	double f_energy (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_inv_difference (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_homogeneity (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_difference_avg (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);

	const double LOG10_2 = 0.30102999566;	// precalculated log 2 base 10
	SimpleMatrix<double> P_matrix;
	std::vector<double> Pxpy, Pxmy;
	const double EPSILON = 0.000000001;
};

