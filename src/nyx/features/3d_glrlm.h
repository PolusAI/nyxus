#pragma once

#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "image_matrix.h"
#include "texture_feature.h"

/// @brief Gray Level Run Length Matrix(GLRLM) features
/// Gray Level Run Length Matrix(GLRLM) quantifies gray level runs, which are defined as the length in number of
/// pixels, of consecutive pixels that have the same gray level value.In a gray level run length matrix
/// 	: math:`\textbf{ P }(i, j | \theta)`, the :math:`(i, j)^ {\text{ th }}` element describes the number of runs with gray level
/// 	: math:`i`and length :math:`j` occur in the image(ROI) along angle : math:`\theta`.
/// 

class D3_GLRLM_feature : public FeatureMethod, public TextureFeature
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
		Nyxus::Feature3D::GLRLM_SRE,		// Short Run Emphasis 
		Nyxus::Feature3D::GLRLM_LRE,		// Long Run Emphasis 
		Nyxus::Feature3D::GLRLM_GLN,		// Gray Level Non-Uniformity 
		Nyxus::Feature3D::GLRLM_GLNN,		// Gray Level Non-Uniformity Normalized 
		Nyxus::Feature3D::GLRLM_RLN,		// Run Length Non-Uniformity
		Nyxus::Feature3D::GLRLM_RLNN,		// Run Length Non-Uniformity Normalized 
		Nyxus::Feature3D::GLRLM_RP,		// Run Percentage
		Nyxus::Feature3D::GLRLM_GLV,		// Gray Level Variance 
		Nyxus::Feature3D::GLRLM_RV,		// Run Variance 
		Nyxus::Feature3D::GLRLM_RE,		// Run Entropy 
		Nyxus::Feature3D::GLRLM_LGLRE,	// Low Gray Level Run Emphasis 
		Nyxus::Feature3D::GLRLM_HGLRE,	// High Gray Level Run Emphasis 
		Nyxus::Feature3D::GLRLM_SRLGLE,	// Short Run Low Gray Level Emphasis 
		Nyxus::Feature3D::GLRLM_SRHGLE,	// Short Run High Gray Level Emphasis 
		Nyxus::Feature3D::GLRLM_LRLGLE,	// Long Run Low Gray Level Emphasis 
		Nyxus::Feature3D::GLRLM_LRHGLE,	// Long Run High Gray Level Emphasis 
		// averaged features:
		Nyxus::Feature3D::GLRLM_SRE_AVE,
		Nyxus::Feature3D::GLRLM_LRE_AVE,
		Nyxus::Feature3D::GLRLM_GLN_AVE,
		Nyxus::Feature3D::GLRLM_GLNN_AVE,
		Nyxus::Feature3D::GLRLM_RLN_AVE,
		Nyxus::Feature3D::GLRLM_RLNN_AVE,
		Nyxus::Feature3D::GLRLM_RP_AVE,
		Nyxus::Feature3D::GLRLM_GLV_AVE,
		Nyxus::Feature3D::GLRLM_RV_AVE,
		Nyxus::Feature3D::GLRLM_RE_AVE,
		Nyxus::Feature3D::GLRLM_LGLRE_AVE,
		Nyxus::Feature3D::GLRLM_HGLRE_AVE,
		Nyxus::Feature3D::GLRLM_SRLGLE_AVE,
		Nyxus::Feature3D::GLRLM_SRHGLE_AVE,
		Nyxus::Feature3D::GLRLM_LRLGLE_AVE,
		Nyxus::Feature3D::GLRLM_LRHGLE_AVE
	};

	// Features implemented by this class that do not require vector-like angled output. Instead, they are output as a single values
	const constexpr static std::initializer_list<Nyxus::Feature3D> nonAngledFeatures =
	{
		Nyxus::Feature3D::GLRLM_SRE_AVE,
		Nyxus::Feature3D::GLRLM_LRE_AVE,
		Nyxus::Feature3D::GLRLM_GLN_AVE,
		Nyxus::Feature3D::GLRLM_GLNN_AVE,
		Nyxus::Feature3D::GLRLM_RLN_AVE,
		Nyxus::Feature3D::GLRLM_RLNN_AVE,
		Nyxus::Feature3D::GLRLM_RP_AVE,
		Nyxus::Feature3D::GLRLM_GLV_AVE,
		Nyxus::Feature3D::GLRLM_RV_AVE,
		Nyxus::Feature3D::GLRLM_RE_AVE,
		Nyxus::Feature3D::GLRLM_LGLRE_AVE,
		Nyxus::Feature3D::GLRLM_HGLRE_AVE,
		Nyxus::Feature3D::GLRLM_SRLGLE_AVE,
		Nyxus::Feature3D::GLRLM_SRHGLE_AVE,
		Nyxus::Feature3D::GLRLM_LRLGLE_AVE,
		Nyxus::Feature3D::GLRLM_LRHGLE_AVE
	};

	D3_GLRLM_feature();

	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);
	static void extract (LR& r, const Fsettings& s);

	// Compatibility with the manual reduce
	static int required(const FeatureSet& fs)
	{
		return fs.anyEnabled(D3_GLRLM_feature::featureset);
	}

	using P_matrix = SimpleMatrix<int>;
	using AngledFtrs = std::vector<double>;

	// 1. Short Run Emphasis 
	double calc_SRE (const SimpleMatrix<int>& P, const double sum_p);
	// 2. Long Run Emphasis 
	double calc_LRE (const SimpleMatrix<int>& P, const double sum_p);
	// 3. Gray Level Non-Uniformity 
	double calc_GLN (const SimpleMatrix<int>& P, const double sum_p);
	// 4. Gray Level Non-Uniformity Normalized 
	double calc_GLNN (const SimpleMatrix<int>& P, const double sum_p);
	// 5. Run Length Non-Uniformity
	double calc_RLN (const SimpleMatrix<int>& P, const double sum_p);
	// 6. Run Length Non-Uniformity Normalized 
	double calc_RLNN (const SimpleMatrix<int>& P, const double sum_p);
	// 7. Run Percentage
	// 8. Gray Level Variance 
	double calc_GLV (const SimpleMatrix<int>& P, const std::vector<PixIntens>& I, const double sum_p);
	// 9. Run Variance 
	double calc_RV (const SimpleMatrix<int>& P, const double sum_p);
	// 10. Run Entropy 
	double calc_RE (const SimpleMatrix<int>& P, const double sum_p);
	// 11. Low Gray Level Run Emphasis 
	double calc_LGLRE (const SimpleMatrix<int>& P, const std::vector<PixIntens>& I, const double sum_p);
	// 12. High Gray Level Run Emphasis 
	double calc_HGLRE (const SimpleMatrix<int>& P, const std::vector<PixIntens>& I, const double sum_p);
	// 13. Short Run Low Gray Level Emphasis 
	double calc_SRLGLE (const SimpleMatrix<int>& P, const std::vector<PixIntens>& I, const double sum_p);
	// 14. Short Run High Gray Level Emphasis 
	double calc_SRHGLE (const SimpleMatrix<int>& P, const std::vector<PixIntens>& I, const double sum_p);
	// 15. Long Run Low Gray Level Emphasis 
	double calc_LRLGLE (const SimpleMatrix<int>& P, const std::vector<PixIntens>& I, const double sum_p);
	// 16. Long Run High Gray Level Emphasis 
	double calc_LRHGLE (const SimpleMatrix<int>& P, const std::vector<PixIntens>& I, const double sum_p);

	constexpr static int rotAngles[] = {0, 45, 90, 135, 0, 45, 90, 135, 0, 45, 90, 135, 0};	// used to name featureset columns

	static void gather_rl_zones (std::vector<std::pair<PixIntens, int>> &Zones, const AngleShift &sh, SimpleCube <PixIntens> &D, PixIntens zeroI);

private:

	int n_angles_ = -1; // dependent on user-specified settings, set in calculate(settings)

	std::vector<double> angled_SRE,
		angled_LRE,
		angled_GLN,
		angled_GLNN,
		angled_RLN,
		angled_RLNN,
		angled_RP,
		angled_GLV,
		angled_RV,
		angled_RE,
		angled_LGLRE,
		angled_HGLRE,
		angled_SRLGLE,
		angled_SRHGLE,
		angled_LRLGLE,
		angled_LRHGLE;

	void clear_buffers();

	const double EPS = 2.2e-16;
	const double BAD_ROI_FVAL = 0.0;
	const double LOG10_2 = 0.30102999566;	// precalculated log 2 base 10
};