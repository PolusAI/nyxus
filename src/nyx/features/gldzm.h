#pragma once

#include "../feature_method.h"

/// @brief Grey Level Distance Zone (GLDZM) features
/// 
/// Grey Level Dsitance Zone (GLDZM) quantifies distances zones of same intensity to the ROI border

class GLDZMFeature : public FeatureMethod
{
public:
	GLDZMFeature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Support of manual reduce
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled({
			GLDZM_SDE,
			GLDZM_LDE,
			GLDZM_LGLE,
			GLDZM_HGLE,
			GLDZM_SDLGLE,
			GLDZM_SDHGLE,
			GLDZM_LDLGLE,
			GLDZM_LDHGLE,
			GLDZM_GLNU,
			GLDZM_GLNUN,
			GLDZM_ZDNU,
			GLDZM_ZDNUN,
			GLDZM_ZP,
			GLDZM_GLM,
			GLDZM_GLV,
			GLDZM_ZDM,
			GLDZM_ZDV,
			GLDZM_ZDE
			});
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