#pragma once

#include "../feature_method.h"
#include "texture_feature.h"

/// @brief Grey Level Distance Zone (GLDZM) features
/// 
/// GLDZM quantifies distance zones of same intensity to the ROI border

class D3_GLDZM_feature : public FeatureMethod, public TextureFeature
{
public:

	// features implemented by this class
	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
		Nyxus::Feature3D::GLDZM_SDE,		// Small Distance Emphasis
		Nyxus::Feature3D::GLDZM_LDE,		// Large Distance Emphasis
		Nyxus::Feature3D::GLDZM_LGLZE,		// Low Grey Level Zone Emphasis
		Nyxus::Feature3D::GLDZM_HGLZE,		// High Grey Level Zone Emphasis
		Nyxus::Feature3D::GLDZM_SDLGLE,	// Small Distance Low Grey Level Emphasis
		Nyxus::Feature3D::GLDZM_SDHGLE,	// Small Distance High GreyLevel Emphasis
		Nyxus::Feature3D::GLDZM_LDLGLE,	// Large Distance Low Grey Level Emphasis
		Nyxus::Feature3D::GLDZM_LDHGLE,	// Large Distance High Grey Level Emphasis
		Nyxus::Feature3D::GLDZM_GLNU,		// Grey Level Non Uniformity
		Nyxus::Feature3D::GLDZM_GLNUN,	// Grey Level Non Uniformity Normalized
		Nyxus::Feature3D::GLDZM_ZDNU,		// Zone Distance Non Uniformity
		Nyxus::Feature3D::GLDZM_ZDNUN,	// Zone Distance Non Uniformity Normalized
		Nyxus::Feature3D::GLDZM_ZP,		// Zone Percentage
		Nyxus::Feature3D::GLDZM_GLM,		// Grey Level Mean
		Nyxus::Feature3D::GLDZM_GLV,		// Grey Level Variance
		Nyxus::Feature3D::GLDZM_ZDM,		// Zone Distance Mean
		Nyxus::Feature3D::GLDZM_ZDV,		// Zone Distance Variance
		Nyxus::Feature3D::GLDZM_ZDE		// Zone Distance Entropy
	};

	D3_GLDZM_feature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Support of manual reduce
	static bool required(const FeatureSet& fs) { return fs.anyEnabled(D3_GLDZM_feature::featureset); }

	// Calculates the GLDZ-matrix, its dimensions, and a vector of sorted grey levels
	void prepare_GLDZM_matrix_kit (SimpleMatrix<unsigned int>& GLDZM, int& Ng, int& Nd, std::vector<PixIntens>& greyLevelsLUT, LR& r);

	static int n_levels; // default value: 0

private:

	void clear_buffers();
	template <class Cube> int dist2border (Cube& I, const int x, const int y, const int z);
	template <class Imgmatrx> void calc_row_and_column_sum_vectors(std::vector<double>& Mx, std::vector<double>& Md, Imgmatrx& P, const int Ng, const int Nd, const std::vector<PixIntens>& greysLUT);

	using IDZ_cluster_indo = std::tuple<PixIntens, int, int>;	// <intensity, distance metric, zone size>
	void calc_gldzm_matrix(SimpleMatrix<unsigned int>& GLDZM, const std::vector<IDZ_cluster_indo>& Z, const std::vector<PixIntens>& greysLUT);

	template <class Imgmatrx> void calc_features(const std::vector<double>& Mx, const std::vector<double>& Md, Imgmatrx& P, const std::vector<PixIntens>& greyLevelsLUT, unsigned int roi_area);

	const double EPS = 2.2e-16;

	// Variables caching feature values between calculate() and save_value(). 
	double f_SDE,	// Small Distance Emphasis
		f_LDE,		// Large Distance Emphasis
		f_LGLZE,	// Low Grey Level Zone Emphasis
		f_HGLZE,	// High Grey Level Zone Emphasis
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

