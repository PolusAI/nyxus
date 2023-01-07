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
	using AngledFeatures = std::vector<double>;

public:

	static int offset;	// default value: 1
	static int n_levels;	// default value: 8
	static std::vector<int> angles;	// default value: {0,45,90,135} (the supreset)

	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled( {
				GLCM_ANGULAR2NDMOMENT,
				GLCM_CONTRAST,
				GLCM_CORRELATION,
				GLCM_DIFFERENCEAVERAGE,	
				GLCM_DIFFERENCEVARIANCE,
				GLCM_DIFFERENCEENTROPY,
				GLCM_ENERGY, 
				GLCM_ENTROPY,
				GLCM_HOMOGENEITY,	
				GLCM_INFOMEAS1,
				GLCM_INFOMEAS2,
				GLCM_INVERSEDIFFERENCEMOMENT,
				GLCM_SUMAVERAGE,
				GLCM_SUMVARIANCE,
				GLCM_SUMENTROPY,
				GLCM_VARIANCE
			});
	}

	GLCMFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:

	void Extract_Texture_Features2_NT (int angle, WriteImageMatrix_nontriv& grays, PixIntens min_val, PixIntens max_val);
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
	double f_corr (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_var (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_idm (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f_savg (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_sentropy (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_svar (const SimpleMatrix<double>& P_matrix, int tone_count, double sum_entropy, std::vector<double>& px);
	double f_entropy (const SimpleMatrix<double>& P_matrix, int tone_count);	
	double f_dvar (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f_dentropy (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);

	void copyfvals(AngledFeatures& dst, const AngledFeatures& src);

	std::vector<double> fvals_ASM,
		fvals_contrast,
		fvals_correlation,
		fvals_energy,
		fvals_homo,
		fvals_variance,
		fvals_IDM,
		fvals_sum_avg,
		fvals_sum_var,
		fvals_sum_entropy,
		fvals_entropy,
		fvals_diff_avg,
		fvals_diff_var,
		fvals_diff_entropy,
		fvals_meas_corr1,
		fvals_meas_corr2,
		fvals_max_corr_coef;

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

