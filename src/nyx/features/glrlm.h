#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "../feature_method.h"
#include "image_matrix.h"
#include "texture_feature.h"

/// @brief Gray Level Run Length Matrix(GLRLM) features
/// Gray Level Run Length Matrix(GLRLM) quantifies gray level runs, which are defined as the length in number of
/// pixels, of consecutive pixels that have the same gray level value.In a gray level run length matrix
/// 	: math:`\textbf{ P }(i, j | \theta)`, the :math:`(i, j)^ {\text{ th }}` element describes the number of runs with gray level
/// 	: math:`i`and length :math:`j` occur in the image(ROI) along angle : math:`\theta`.
/// 

class GLRLMFeature : public FeatureMethod, public TextureFeature
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::GLRLM_SRE,		// Short Run Emphasis 
		Nyxus::Feature2D::GLRLM_LRE,		// Long Run Emphasis 
		Nyxus::Feature2D::GLRLM_GLN,		// Gray Level Non-Uniformity 
		Nyxus::Feature2D::GLRLM_GLNN,		// Gray Level Non-Uniformity Normalized 
		Nyxus::Feature2D::GLRLM_RLN,		// Run Length Non-Uniformity
		Nyxus::Feature2D::GLRLM_RLNN,		// Run Length Non-Uniformity Normalized 
		Nyxus::Feature2D::GLRLM_RP,		// Run Percentage
		Nyxus::Feature2D::GLRLM_GLV,		// Gray Level Variance 
		Nyxus::Feature2D::GLRLM_RV,		// Run Variance 
		Nyxus::Feature2D::GLRLM_RE,		// Run Entropy 
		Nyxus::Feature2D::GLRLM_LGLRE,	// Low Gray Level Run Emphasis 
		Nyxus::Feature2D::GLRLM_HGLRE,	// High Gray Level Run Emphasis 
		Nyxus::Feature2D::GLRLM_SRLGLE,	// Short Run Low Gray Level Emphasis 
		Nyxus::Feature2D::GLRLM_SRHGLE,	// Short Run High Gray Level Emphasis 
		Nyxus::Feature2D::GLRLM_LRLGLE,	// Long Run Low Gray Level Emphasis 
		Nyxus::Feature2D::GLRLM_LRHGLE,	// Long Run High Gray Level Emphasis 
		// averaged features:
		Nyxus::Feature2D::GLRLM_SRE_AVE,
		Nyxus::Feature2D::GLRLM_LRE_AVE,
		Nyxus::Feature2D::GLRLM_GLN_AVE,
		Nyxus::Feature2D::GLRLM_GLNN_AVE,
		Nyxus::Feature2D::GLRLM_RLN_AVE,
		Nyxus::Feature2D::GLRLM_RLNN_AVE,
		Nyxus::Feature2D::GLRLM_RP_AVE,
		Nyxus::Feature2D::GLRLM_GLV_AVE,
		Nyxus::Feature2D::GLRLM_RV_AVE,
		Nyxus::Feature2D::GLRLM_RE_AVE,
		Nyxus::Feature2D::GLRLM_LGLRE_AVE,
		Nyxus::Feature2D::GLRLM_HGLRE_AVE,
		Nyxus::Feature2D::GLRLM_SRLGLE_AVE,
		Nyxus::Feature2D::GLRLM_SRHGLE_AVE,
		Nyxus::Feature2D::GLRLM_LRLGLE_AVE,
		Nyxus::Feature2D::GLRLM_LRHGLE_AVE
	};

	// Features implemented by this class that do not require vector-like angled output. Instead, they are output as a single values
	const constexpr static std::initializer_list<Nyxus::Feature2D> nonAngledFeatures =
	{
		Nyxus::Feature2D::GLRLM_SRE_AVE,
		Nyxus::Feature2D::GLRLM_LRE_AVE,
		Nyxus::Feature2D::GLRLM_GLN_AVE,
		Nyxus::Feature2D::GLRLM_GLNN_AVE,
		Nyxus::Feature2D::GLRLM_RLN_AVE,
		Nyxus::Feature2D::GLRLM_RLNN_AVE,
		Nyxus::Feature2D::GLRLM_RP_AVE,
		Nyxus::Feature2D::GLRLM_GLV_AVE,
		Nyxus::Feature2D::GLRLM_RV_AVE,
		Nyxus::Feature2D::GLRLM_RE_AVE,
		Nyxus::Feature2D::GLRLM_LGLRE_AVE,
		Nyxus::Feature2D::GLRLM_HGLRE_AVE,
		Nyxus::Feature2D::GLRLM_SRLGLE_AVE,
		Nyxus::Feature2D::GLRLM_SRHGLE_AVE,
		Nyxus::Feature2D::GLRLM_LRLGLE_AVE,
		Nyxus::Feature2D::GLRLM_LRHGLE_AVE
	};

	GLRLMFeature();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void extract (LR& roi);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Compatibility with the manual reduce
	static int required(const FeatureSet& fs)
	{
		return fs.anyEnabled(GLRLMFeature::featureset);
	}

	static int n_levels;	// default value: 0
	using P_matrix = SimpleMatrix<int>;
	using AngledFtrs = std::vector<double>;

	// 1. Short Run Emphasis 
	void calc_SRE(AngledFtrs& af);
	// 2. Long Run Emphasis 
	void calc_LRE(AngledFtrs& af);
	// 3. Gray Level Non-Uniformity 
	void calc_GLN(AngledFtrs& af);
	// 4. Gray Level Non-Uniformity Normalized 
	void calc_GLNN(AngledFtrs& af);
	// 5. Run Length Non-Uniformity
	void calc_RLN(AngledFtrs& af);
	// 6. Run Length Non-Uniformity Normalized 
	void calc_RLNN(AngledFtrs& af);
	// 7. Run Percentage
	void calc_RP(AngledFtrs& af);
	// 8. Gray Level Variance 
	void calc_GLV(AngledFtrs& af);
	// 9. Run Variance 
	void calc_RV(AngledFtrs& af);
	// 10. Run Entropy 
	void calc_RE(AngledFtrs& af);
	// 11. Low Gray Level Run Emphasis 
	void calc_LGLRE(AngledFtrs& af);
	// 12. High Gray Level Run Emphasis 
	void calc_HGLRE(AngledFtrs& af);
	// 13. Short Run Low Gray Level Emphasis 
	void calc_SRLGLE(AngledFtrs& af);
	// 14. Short Run High Gray Level Emphasis 
	void calc_SRHGLE(AngledFtrs& af);
	// 15. Long Run Low Gray Level Emphasis 
	void calc_LRLGLE(AngledFtrs& af);
	// 16. Long Run High Gray Level Emphasis 
	void calc_LRHGLE(AngledFtrs& af);

	constexpr static int rotAngles[] = { 0, 45, 90, 135 };

private:

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

	double calc_ave(const std::vector<double>& angled_feature_vals);

	int Na = 4;	// number of angles

	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	std::vector<int> angles_Ng;	// number of discrete intensity values in the image
	std::vector<int> angles_Nr; // number of discrete run lengths in the image
	std::vector<int> angles_Np; // number of voxels in the image
	std::vector<P_matrix> angles_P;
	std::vector<double> sum_p;
	std::vector<PixIntens> I;	// sorted unique intensities

	void clear_buffers();

	const double EPS = 2.2e-16;
	const double BAD_ROI_FVAL = 0.0;
	const double LOG10_2 = 0.30102999566;	// precalculated log 2 base 10
};