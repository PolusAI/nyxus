#pragma once

#include "../feature_method.h"

/// @brief Grey Level Distance Zone (GLDZM) features
/// 
/// Grey Level Dsitance Zone (GLDZM) quantifies distances zones of same intensity to the ROI border

class GLDZMFeature : public FeatureMethod
{
public:	
	
	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::AvailableFeatures> featureset =
	{
		GLDZM_SDE,		// Small Distance Emphasis
		GLDZM_LDE,		// Large Distance Emphasis
		GLDZM_LGLE,		// Low Grey Level Emphasis
		GLDZM_HGLE,		// High GreyLevel Emphasis
		GLDZM_SDLGLE,	// Small Distance Low Grey Level Emphasis
		GLDZM_SDHGLE,	// Small Distance High GreyLevel Emphasis
		GLDZM_LDLGLE,	// Large Distance Low Grey Level Emphasis
		GLDZM_LDHGLE,	// Large Distance High Grey Level Emphasis
		GLDZM_GLNU,		// Grey Level Non Uniformity
		GLDZM_GLNUN,	// Grey Level Non Uniformity Normalized
		GLDZM_ZDNU,		// Zone Distance Non Uniformity
		GLDZM_ZDNUN,	// Zone Distance Non Uniformity Normalized
		GLDZM_ZP,		// Zone Percentage
		GLDZM_GLM,		// Grey Level Mean
		GLDZM_GLV,		// Grey Level Variance
		GLDZM_ZDM,		// Zone Distance Mean
		GLDZM_ZDV,		// Zone Distance Variance
		GLDZM_ZDE		// Zone Distance Entropy
	};

	GLDZMFeature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Support of manual reduce
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled (GLDZMFeature::featureset);
	}

private:

	void clear_buffers();	
	template <class Imgmatrx> int dist2border (Imgmatrx & I, const int x, const int y);
	template <class Imgmatrx> void calc_row_and_column_sum_vectors (std::vector<double>& Mx, std::vector<double>& Md, Imgmatrx & P, const int Ng, const int Nd);
	template <class Imgmatrx> void calc_features (const std::vector<double>& Mx, const std::vector<double>& Md, Imgmatrx& P, unsigned int roi_area);

	const double EPS = 2.2e-16;

	// Variables caching feature values between calculate() and save_value(). 
	double f_SDE,		// Small Distance Emphasis
		f_LDE,		// Large Distance Emphasis
		f_LGLE,		// Low Grey Level Emphasis
		f_HGLE,		// High GreyLevel Emphasis
		f_SDLGLE,	// Small Distance Low Grey Level Emphasis
		f_SDHGLE,	// Small Distance High GreyLevel Emphasis
		f_LDLGLE,	// Large Distance Low Grey Level Emphasis
		f_LDHGLE,	// Large Distance High Grey Level Emphasis
		f_GLNU,		// Grey Level Non Uniformity
		f_GLNUN,	// Grey Level Non Uniformity Normalized
		f_ZDNU,		// Zone Distance Non Uniformity
		f_ZDNUN,	// Zone Distance Non Uniformity Normalized
		f_ZP,		// Zone Percentage
		f_GLM,		// Grey Level Mean
		f_GLV,		// Grey Level Variance
		f_ZDM,		// Zone Distance Mean
		f_ZDV,		// Zone Distance Variance
		f_ZDE,		// Zone Distance Entropy
		f_GLE;		// Grey Level Entropy
};