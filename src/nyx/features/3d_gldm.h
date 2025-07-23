#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "texture_feature.h"

// Inspired by 
//		https://stackoverflow.com/questions/25019840/neighboring-gray-level-dependence-matrix-ngldm-in-matlab?fbclid=IwAR14fT0kpmjmOXRhKcguFMH3tCg0G4ubDLRxyHZoXdpKdbPxF7Zuq-WKE8o
//		https://qiita.com/tatsunidas/items/fd49ef6ac7c3deb141e0
//

class D3_GLDM_feature : public FeatureMethod, public TextureFeature
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
		Nyxus::Feature3D::GLDM_SDE,		// Small Dependence Emphasis
		Nyxus::Feature3D::GLDM_LDE,		// Large Dependence Emphasis
		Nyxus::Feature3D::GLDM_GLN,		// Gray Level Non-Uniformity
		Nyxus::Feature3D::GLDM_DN,		// Dependence Non-Uniformity
		Nyxus::Feature3D::GLDM_DNN,		// Dependence Non-Uniformity Normalized
		Nyxus::Feature3D::GLDM_GLV,		// Gray Level Variance
		Nyxus::Feature3D::GLDM_DV,		// Dependence Variance
		Nyxus::Feature3D::GLDM_DE,		// Dependence Entropy
		Nyxus::Feature3D::GLDM_LGLE,		// Low Gray Level Emphasis
		Nyxus::Feature3D::GLDM_HGLE,		// High Gray Level Emphasis
		Nyxus::Feature3D::GLDM_SDLGLE,	// Small Dependence Low Gray Level Emphasis
		Nyxus::Feature3D::GLDM_SDHGLE,	// Small Dependence High Gray Level Emphasis
		Nyxus::Feature3D::GLDM_LDLGLE,	// Large Dependence Low Gray Level Emphasis
		Nyxus::Feature3D::GLDM_LDHGLE	// Large Dependence High Gray Level Emphasis
	};

	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled (D3_GLDM_feature::featureset);
	}

	D3_GLDM_feature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void extract (LR& r);

	// 1. Small Dependence Emphasis(SDE)
	double calc_SDE();
	// 2. Large Dependence Emphasis (LDE)
	double calc_LDE();
	// 3. Gray Level Non-Uniformity (GLN)
	double calc_GLN();
	// 4. Dependence Non-Uniformity (DN)
	double calc_DN();
	// 5. Dependence Non-Uniformity Normalized (DNN)
	double calc_DNN();
	// 6. Gray Level Variance (GLV)
	double calc_GLV();
	// 7. Dependence Variance (DV)
	double calc_DV();
	// 8. Dependence Entropy (DE)
	double calc_DE();
	// 9. Low Gray Level Emphasis (LGLE)
	double calc_LGLE();
	// 10. High Gray Level Emphasis (HGLE)
	double calc_HGLE();
	// 11. Small Dependence Low Gray Level Emphasis (SDLGLE)
	double calc_SDLGLE();
	// 12. Small Dependence High Gray Level Emphasis (SDHGLE)
	double calc_SDHGLE();
	// 13. Large Dependence Low Gray Level Emphasis (LDLGLE)
	double calc_LDLGLE();
	// 14. Large Dependence High Gray Level Emphasis (LDHGLE)
	double calc_LDHGLE();

private:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	int Ng = 0;	// number of discrete intensity values in the image
	int Nd = 0; // number of discrete dependency sizes in the image
	int Nz = 0; // number of dependency zones in the ROI, Nz = sum(sum(P[i,j]))

	SimpleMatrix<int> P;	// dependence matrix

	std::vector<PixIntens> I;	// sorted unique intensities after image greyscale binning

	void clear_buffers();

	const double BAD_ROI_FVAL = 0.0;
	const double EPS = 2.2e-16;
	const double LOG10_2 = 0.30102999566;	// precalculated log 2 base 10

	// feature values cache between calculate() and save_value()
	double fv_SDE,
		fv_LDE,
		fv_GLN,
		fv_DN,
		fv_DNN,
		fv_GLV,
		fv_DV,
		fv_DE,
		fv_LGLE,
		fv_HGLE,
		fv_SDLGLE,
		fv_SDHGLE,
		fv_LDLGLE,
		fv_LDHGLE;

};

