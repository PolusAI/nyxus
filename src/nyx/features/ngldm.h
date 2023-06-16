#pragma once

#include "../feature_method.h"

/// @brief Neighbouring grey level dependence matrix (NGLDM) based features
/// 
/// NGLDM features quantify the coarseness of the overall texture in a rotationally invariant way

class NGLDMfeature : public FeatureMethod
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::AvailableFeatures> featureset =
	{
		NGLDM_LDE,		// Low Dependence Emphasis
		NGLDM_HDE,		// High Dependence Emphasis
		NGLDM_LGLCE,	// Low Grey Level Count Emphasis
		NGLDM_HGLCE,	// High Grey Level Count Emphasis
		NGLDM_LDLGLE,	// Low Dependence Low Grey Level Emphasis
		NGLDM_LDHGLE,	// Low Dependence High Grey Level Emphasis
		NGLDM_HDLGLE,	// High Dependence Low Grey Level Emphasis
		NGLDM_HDHGLE,	// High Dependence High Grey Level Emphasis
		NGLDM_GLNU,		// Grey Level Non-Uniformity
		NGLDM_GLNUN,	// Grey Level Non-Uniformity Normalised
		NGLDM_DCNU,		// Dependence Count Non-Uniformity
		NGLDM_DCNUN,	// Dependence Count Non-Uniformity Normalised
		NGLDM_GLM,		// Grey Level Mean
		NGLDM_DCP,		// Dependence Count Percentage
		NGLDM_GLV,		// Grey Level Variance
		NGLDM_DCM,		// Dependence Count Mean
		NGLDM_DCV,		// Dependence Count Variance
		NGLDM_DCENT,	// Dependence Count Entropy
		NGLDM_DCENE		// Dependence Count Energy
	};

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
		return fs.anyEnabled (NGLDMfeature::featureset);
	}

	// Support of unit testing
	void prepare_NGLDM_matrix_kit (SimpleMatrix<unsigned int>& NGLDM, std::vector<PixIntens>& grey_levels_LUT, int& Ng, int& Nr, LR& r);

private:

	void clear_buffers();
	template <class Pixelcloud> void gather_unique_intensities (std::vector<PixIntens> & V, Pixelcloud& C, PixIntens max_i);
	void gather_unique_intensities2 (std::vector<PixIntens>& V, const pixData& I, PixIntens max_inten);
	template <class Imgmatrix> void calc_ngld_matrix (SimpleMatrix<unsigned int> & NGLDM, int & max_dep, Imgmatrix & I, const std::vector<PixIntens> & V, PixIntens max_inten);
	void calc_rowwise_and_columnwise_totals (std::vector<double>& Mg, std::vector<double>& Mr, const SimpleMatrix<unsigned int>& NGLDM, const int Ng, const int Nr);
	void calc_features (const std::vector<double>& Mx, const std::vector<double>& Md, SimpleMatrix<unsigned int>& NGLDM, int Nr, const std::vector<PixIntens> U, unsigned int roi_area);

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
	double f_DCP;	// Dependence Count Percentage
	double f_DCV;	// Dependence Count Variance
	double f_DCENT;	// Dependence Count Entropy
	double f_DCENE;	// Dependence Count Energy
};
