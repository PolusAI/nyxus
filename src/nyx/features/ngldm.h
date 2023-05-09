#pragma once

#include "../feature_method.h"

/// @brief Neighbouring grey level dependence matrix (NGLDM) based features
/// 
/// NGLDM features quantify the coarseness of the overall texture in a rotationally invariant way

class NGLDMfeature : public FeatureMethod
{
public:
	NGLDMfeature();

	// Overrides
	void calculate (LR& r);
	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, ImageLoader& imloader);
	void save_value (std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Support of manual reduce
	static bool required (const FeatureSet& fs)
	{
		return fs.anyEnabled ({
			NGLDM_LDE,
			NGLDM_HDE,
			NGLDM_LGLCE,
			NGLDM_HGLCE,
			NGLDM_LDLGLE,
			NGLDM_LDHGLE,
			NGLDM_HDLGLE,
			NGLDM_HDHGLE,
			NGLDM_GLNU,
			NGLDM_GLNUN,
			NGLDM_DCNU,
			NGLDM_DCNUN,
			NGLDM_GLM,
			NGLDM_GLV,
			NGLDM_DCM,
			NGLDM_DCV,
			NGLDM_DCE,
			NGLDM_DCENE });
	}

private:

	void clear_buffers();
	template <class Pixelcloud> void gather_unique_intensities (std::vector<PixIntens> & V, Pixelcloud& C, PixIntens max_i);
	template <class Imgmatrix> void calc_ngldm (SimpleMatrix<unsigned int> & NGLDM, Imgmatrix & I, std::vector<PixIntens> & V, PixIntens max_inten);
	void calc_rowwise_and_columnwise_totals (std::vector<double>& Mg, std::vector<double>& Mr, const SimpleMatrix<unsigned int>& NGLDM, const int Ng, const int Nr);
	void calc_features (const std::vector<double>& Mx, const std::vector<double>& Md, SimpleMatrix<unsigned int>& P, unsigned int roi_area);

	const double EPS = 2.2e-16;

	// Variables caching feature values between calculate() and save_value(). 
	double f_LDE;	// Low Dependence Emphasis
	double f_HDE;	// High Dependence Emphasis
	double f_LGLCE;	// Low Grey Level Count Emphasis
	double f_HGLCE;	// High Grey Level Count Emphasis
	double f_LDLGLE;	// Low Dependence Low Grey Level Emphasis
	double f_LDHGLE;	// Low Dependence High Grey Level Emphasis
	double f_HDLGLE;	// High Dependence Low Grey Level Emphasis
	double f_HDHGLE;	// High Dependence High Grey Level Emphasis
	double f_GLNU;	// Grey Level Non-Uniformity
	double f_GLNUN;	// Grey Level Non-Uniformity Normalised
	double f_DCNU;	// Dependence Count Non-Uniformity
	double f_DCNUN;	// Dependence Count Non-Uniformity Normalised
	double f_GLCM;	// Grey Level Count Mean
	double f_GLV;	// Grey Level Variance
	double f_DCM;	// Dependence Count Mean
	double f_DCV;	// Dependence Count Variance
	double f_DCE;	// Dependence Count Entropy
	double f_DCENE;	// Dependence Count Energy
};
