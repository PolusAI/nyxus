#pragma once

#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "texture_feature.h"

// Inspired by 
//		https://stackoverflow.com/questions/25019840/neighboring-gray-level-dependence-matrix-ngldm-in-matlab?fbclid=IwAR14fT0kpmjmOXRhKcguFMH3tCg0G4ubDLRxyHZoXdpKdbPxF7Zuq-WKE8o
//		https://qiita.com/tatsunidas/items/fd49ef6ac7c3deb141e0
//

/// @brief Gray Level Dependence Matrix (GLDM) features
/// Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies in an image.
///	A gray level dependency is defined as a the number of connected voxels within distance : math:`\delta` that are
///	dependent on the center voxel.
///	A neighbouring voxel with gray level : math:`j` is considered dependent on center voxel with gray level : math:`i`
///	if :math:`|i - j | \le\alpha`. In a gray level dependence matrix : math:`\textbf{ P }(i, j)` the :math:`(i, j)`\ :sup:`th`
///	element describes the number of times a voxel with gray level : math:`i` with : math : `j` dependent voxels
///	in its neighbourhood appears in image.
/// 

class GLDMFeature: public FeatureMethod, public TextureFeature
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::GLDM_SDE,		// Small Dependence Emphasis
		Nyxus::Feature2D::GLDM_LDE,		// Large Dependence Emphasis
		Nyxus::Feature2D::GLDM_GLN,		// Gray Level Non-Uniformity
		Nyxus::Feature2D::GLDM_DN,		// Dependence Non-Uniformity
		Nyxus::Feature2D::GLDM_DNN,		// Dependence Non-Uniformity Normalized
		Nyxus::Feature2D::GLDM_GLV,		// Gray Level Variance
		Nyxus::Feature2D::GLDM_DV,		// Dependence Variance
		Nyxus::Feature2D::GLDM_DE,		// Dependence Entropy
		Nyxus::Feature2D::GLDM_LGLE,		// Low Gray Level Emphasis
		Nyxus::Feature2D::GLDM_HGLE,		// High Gray Level Emphasis
		Nyxus::Feature2D::GLDM_SDLGLE,	// Small Dependence Low Gray Level Emphasis
		Nyxus::Feature2D::GLDM_SDHGLE,	// Small Dependence High Gray Level Emphasis
		Nyxus::Feature2D::GLDM_LDLGLE,	// Large Dependence Low Gray Level Emphasis
		Nyxus::Feature2D::GLDM_LDHGLE	// Large Dependence High Gray Level Emphasis
	};

	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled(GLDMFeature::featureset);
	}

	GLDMFeature ();

	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value (std::vector<std::vector<double>>& feature_vals);
	static void extract (LR& roi, const Fsettings& s);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);

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

