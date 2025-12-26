#pragma once

#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "texture_feature.h"

class D3_GLSZM_feature : public FeatureMethod, public TextureFeature
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
		Nyxus::Feature3D::GLSZM_SAE,		// Small Area Emphasis
		Nyxus::Feature3D::GLSZM_LAE,		// Large Area Emphasis
		Nyxus::Feature3D::GLSZM_GLN,		// Gray Level Non - Uniformity
		Nyxus::Feature3D::GLSZM_GLNN,		// Gray Level Non - Uniformity Normalized
		Nyxus::Feature3D::GLSZM_SZN,		// Size - Zone Non - Uniformity
		Nyxus::Feature3D::GLSZM_SZNN,		// Size - Zone Non - Uniformity Normalized
		Nyxus::Feature3D::GLSZM_ZP,		// Zone Percentage
		Nyxus::Feature3D::GLSZM_GLV,		// Gray Level Variance
		Nyxus::Feature3D::GLSZM_ZV,		// Zone Variance
		Nyxus::Feature3D::GLSZM_ZE,		// Zone Entropy
		Nyxus::Feature3D::GLSZM_LGLZE,	// Low Gray Level Zone Emphasis
		Nyxus::Feature3D::GLSZM_HGLZE,	// High Gray Level Zone Emphasis
		Nyxus::Feature3D::GLSZM_SALGLE,	// Small Area Low Gray Level Emphasis
		Nyxus::Feature3D::GLSZM_SAHGLE,	// Small Area High Gray Level Emphasis
		Nyxus::Feature3D::GLSZM_LALGLE,	// Large Area Low Gray Level Emphasis
		Nyxus::Feature3D::GLSZM_LAHGLE,	// Large Area High Gray Level Emphasis
	};

	D3_GLSZM_feature();

	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);
	static void extract (LR& r, const Fsettings& s);

	// Compatibility with the manual reduce
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled (D3_GLSZM_feature::featureset);
	}

	// Small Area Emphasis
	double calc_SAE();

	// Large Area Emphasis
	double calc_LAE();

	// Gray Level Non - Uniformity
	double calc_GLN();

	// Gray Level Non - Uniformity Normalized
	double calc_GLNN();

	// Size - Zone Non - Uniformity
	double calc_SZN();

	// Size - Zone Non - Uniformity Normalized
	double calc_SZNN();

	// Zone Percentage
	double calc_ZP();

	// Gray Level Variance
	double calc_GLV();

	// Zone Variance
	double calc_ZV();

	// Zone Entropy
	double calc_ZE();

	// Low Gray Level Zone Emphasis
	double calc_LGLZE();

	// High Gray Level Zone Emphasis
	double calc_HGLZE();

	// Small Area Low Gray Level Emphasis
	double calc_SALGLE();

	// Small Area High Gray Level Emphasis
	double calc_SAHGLE();

	// Large Area Low Gray Level Emphasis
	double calc_LALGLE();

	// Large Area High Gray Level Emphasis
	double calc_LAHGLE();

	static void gather_size_zones (std::vector<std::pair<PixIntens, int>> & zones, SimpleCube <PixIntens> & greybinned_image, PixIntens zero_intensity);

private:

	int Ng = 0;	// number of discrete intensity values in the image
	int Ns = 0; // number of discrete zone sizes in the image
	int Np = 0; // number of voxels in the image
	int Nz = 0; // number of zones in the ROI, 1<=Nz<=Np
	SimpleMatrix<int> P;
	double sum_p = 0;	// number of zones (= total of GLSZM)
	std::vector<PixIntens> I;	// sorted unique intensities

	// Sum of P required by GLN, GLNN, LGLZE, HGLZE
	std::vector<double> sj;
	// Sum of P required by SAE, LAE, SZN, SZNN
	std::vector<double> si;
	double f_LAHGLE, f_LALGLE, f_SAHGLE, f_SALGLE, f_ZE, mu_GLV, mu_ZV;
	void calc_sums_of_P();

	void clear_buffers()
	{
		Ng = 0;	// number of discrete intensity values in the image
		Ns = 0; // number of discrete zone sizes in the image
		Np = 0; // number of voxels in the image
		Nz = 0; // number of zones in the ROI, 1<=Nz<=Np
		P.clear();
		sum_p = 0;
	}

	const double EPS = 2.2e-16;
	const double BAD_ROI_FVAL = 0.0;
	const double LOG10_2 = 0.30102999566;	// precalculated log 2 base 10

	// feature value cache
	double fv_SAE,
		fv_LAE,
		fv_GLN,
		fv_GLNN,
		fv_SZN,
		fv_SZNN,
		fv_ZP,
		fv_GLV,
		fv_ZV,
		fv_ZE,
		fv_LGLZE,
		fv_HGLZE,
		fv_SALGLE,
		fv_SAHGLE,
		fv_LALGLE,
		fv_LAHGLE;

	void invalidate (double soft_nan);	// assigns each cached feature value a safe NAN
};
