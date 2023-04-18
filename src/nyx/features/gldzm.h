#pragma once

#include "../feature_method.h"

/// @brief Gray Level Distance Zone(GLSZM) features
/// 
/// Gray Level Size Zone(GLSZM) quantifies distances from locations to gray level zones in an image.

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

	void calc_auxiliary_sums (std::vector<double>& Mx, std::vector<double>& Md, const SimpleMatrix<int>& P, const int Ng, const int Nd);

	void clear_buffers()
	{
		f_SDE = 
		f_LDE = 
		f_LGLE = 
		f_HGLE = 
		f_SDLGLE = 
		f_SDHGLE = 
		f_LDLGLE = 
		f_LDHGLE = 
		f_GLNU = 
		f_GLNUN = 
		f_ZDNU = 
		f_ZDNUN = 
		f_ZP = 
		f_GLM = 
		f_GLV = 
		f_ZDM = 
		f_ZDV = 
		f_ZDE = 
		f_GLE = 0;
	}

	const double EPS = 2.2e-16;

	int dist2closestRoiBorder (const pixData& D, const int x, const int y);

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