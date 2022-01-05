#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"

typedef struct
{
	double ASM;          /*  (1) Angular Second Moment */
	double contrast;     /*  (2) Contrast */
	double correlation;  /*  (3) Correlation */
	double variance;     /*  (4) Variance */
	double IDM;		    /*  (5) Inverse Diffenence Moment */
	double sum_avg;	    /*  (6) Sum Average */
	double sum_var;	    /*  (7) Sum Variance */
	double sum_entropy;	/*  (8) Sum Entropy */
	double entropy;	    /*  (9) Entropy */
	double diff_var;	    /* (10) Difference Variance */
	double diff_entropy;	/* (11) Diffenence Entropy */
	double meas_corr1;	/* (12) Measure of Correlation 1 */
	double meas_corr2;	/* (13) Measure of Correlation 2 */
	double max_corr_coef; /* (14) Maximal Correlation Coefficient */
} Haralick_feature_values;

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

class GLCM_features
{
	using C_matrix = SimpleMatrix<int>;
	using AngledFeatures = std::vector<double>;

public:
	static bool required(const FeatureSet& fs) { 
		return fs.anyEnabled({
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
				GLCM_INFOMEAS2 });
	}
	GLCM_features (int minI, int maxI, const ImageMatrix& im, double distance);

	void get_AngularSecondMoments (AngledFeatures& af);
	void get_Contrast (AngledFeatures& af);
	void get_Correlation (AngledFeatures& af);
	void get_Variance (AngledFeatures& af);
	void get_InverseDifferenceMoment (AngledFeatures& af);
	void get_SumAverage (AngledFeatures& af);
	void get_SumVariance (AngledFeatures& af);
	void get_SumEntropy (AngledFeatures& af);
	void get_Entropy (AngledFeatures& af);
	void get_DifferenceVariance (AngledFeatures& af);
	void get_DifferenceEntropy (AngledFeatures& af);
	void get_InfoMeas1 (AngledFeatures& af);
	void get_InfoMeas2 (AngledFeatures& af);

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	SimpleMatrix<int> P;	// colocation matrix
	
	//std::vector <Haralick_feature_values> fvals = { {}, {}, {}, {} };	// 4 angles

	std::vector<double> fvals_ASM,          /*  (1) Angular Second Moment */
		fvals_contrast,     /*  (2) Contrast */
		fvals_correlation,  /*  (3) Correlation */
		fvals_variance,     /*  (4) Variance */
		fvals_IDM,		    /*  (5) Inverse Diffenence Moment */
		fvals_sum_avg,	    /*  (6) Sum Average */
		fvals_sum_var,	    /*  (7) Sum Variance */
		fvals_sum_entropy,	/*  (8) Sum Entropy */
		fvals_entropy,	    /*  (9) Entropy */
		fvals_diff_var,	    /* (10) Difference Variance */
		fvals_diff_entropy,	/* (11) Diffenence Entropy */
		fvals_meas_corr1,	/* (12) Measure of Correlation 1 */
		fvals_meas_corr2,	/* (13) Measure of Correlation 2 */
		fvals_max_corr_coef; /* (14) Maximal Correlation Coefficient */

	void Extract_Texture_Features(
		int distance,
		int angle,
		const SimpleMatrix<uint8_t>& grays);	// 'grays' is 0-255 grays 

	void calculate_graytones (SimpleMatrix<uint8_t>& G, int minI, int maxI, const ImageMatrix& Im);

	void CoOcMat_Angle_0(
		// out
		SimpleMatrix<double>& matrix,
		// in
		int distance,
		const SimpleMatrix<uint8_t>& grays,
		const int* tone_LUT,
		int tone_count);
	void CoOcMat_Angle_45(
		// out
		SimpleMatrix<double>& matrix,
		// in
		int distance,
		const SimpleMatrix<uint8_t>& grays,
		const int* tone_LUT,
		int tone_count);
	void CoOcMat_Angle_90(
		// out
		SimpleMatrix<double>& matrix,
		// in
		int distance,
		const SimpleMatrix<uint8_t>& grays,
		const int* tone_LUT,
		int tone_count);
	void CoOcMat_Angle_135(
		// out
		SimpleMatrix<double>& matrix,
		// in
		int distance,
		const SimpleMatrix<uint8_t>& grays,
		const int* tone_LUT,
		int tone_count);

	double f1_asm (const SimpleMatrix<double>& P_matrix, int tone_count);	
	double f2_contrast (const SimpleMatrix<double>& P_matix, int tone_count);	
	double f3_corr (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f4_var (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f5_idm (const SimpleMatrix<double>& P_matrix, int tone_count);
	double f6_savg (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f8_sentropy (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f7_svar (const SimpleMatrix<double>& P_matrix, int tone_count, double sum_entropy, std::vector<double>& px);
	double f9_entropy (const SimpleMatrix<double>& P_matrix, int tone_count);	
	double f10_dvar (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f11_dentropy (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px);
	double f12_icorr (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px, std::vector<double>& py);
	double f13_icorr (const SimpleMatrix<double>& P_matrix, int tone_count, std::vector<double>& px, std::vector<double>& py);

	//---	double* allocate_vector (int nl, int nh);

};

