#pragma once

#include <unordered_map>
#include "../roi_data.h"
#include "image_matrix.h"

/// @brief Gray Level Run Length Matrix(GLRLM) features
/// Gray Level Run Length Matrix(GLRLM) quantifies gray level runs, which are defined as the length in number of
/// pixels, of consecutive pixels that have the same gray level value.In a gray level run length matrix
/// 	: math:`\textbf{ P }(i, j | \theta)`, the :math:`(i, j)^ {\text{ th }}` element describes the number of runs with gray level
/// 	: math:`i`and length :math:`j` occur in the image(ROI) along angle : math:`\theta`.
/// 

class GLRLM_features
{
	using P_matrix = SimpleMatrix<int>;
	using AngledFtrs = std::vector<double>;

public:
	static int required(const FeatureSet& fs) {
		return fs.anyEnabled({
				GLRLM_SRE,
				GLRLM_LRE,
				GLRLM_GLN,
				GLRLM_GLNN,
				GLRLM_RLN,
				GLRLM_RLNN,
				GLRLM_RP,
				GLRLM_GLV,
				GLRLM_RV,
				GLRLM_RE,
				GLRLM_LGLRE,
				GLRLM_HGLRE,
				GLRLM_SRLGLE,
				GLRLM_SRHGLE,
				GLRLM_LRLGLE,
				GLRLM_LRHGLE
			});
	}
	GLRLM_features (int minI, int maxI, const ImageMatrix& im);

	// 1. Short Run Emphasis 
	void calc_SRE (AngledFtrs& af);
	// 2. Long Run Emphasis 
	void calc_LRE (AngledFtrs& af);
	// 3. Gray Level Non-Uniformity 
	void calc_GLN (AngledFtrs& af);
	// 4. Gray Level Non-Uniformity Normalized 
	void calc_GLNN (AngledFtrs& af);
	// 5. Run Length Non-Uniformity
	void calc_RLN (AngledFtrs& af);
	// 6. Run Length Non-Uniformity Normalized 
	void calc_RLNN (AngledFtrs& af);
	// 7. Run Percentage
	void calc_RP (AngledFtrs& af);
	// 8. Gray Level Variance 
	void calc_GLV (AngledFtrs& af);
	// 9. Run Variance 
	void calc_RV (AngledFtrs& af);
	// 10. Run Entropy 
	void calc_RE (AngledFtrs& af);
	// 11. Low Gray Level Run Emphasis 
	void calc_LGLRE (AngledFtrs& af);
	// 12. High Gray Level Run Emphasis 
	void calc_HGLRE (AngledFtrs& af);
	// 13. Short Run Low Gray Level Emphasis 
	void calc_SRLGLE (AngledFtrs& af);
	// 14. Short Run High Gray Level Emphasis 
	void calc_SRHGLE (AngledFtrs& af);
	// 15. Long Run Low Gray Level Emphasis 
	void calc_LRLGLE (AngledFtrs& af);
	// 16. Long Run High Gray Level Emphasis 
	void calc_LRHGLE (AngledFtrs& af);

	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	constexpr static int rotAngles [] = {0, 45, 90, 135};

private:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	std::vector<int> angles_Ng;	// number of discreet intensity values in the image
	std::vector<int> angles_Nr; // number of discreet run lengths in the image
	std::vector<int> angles_Np; // number of voxels in the image
	std::vector<P_matrix> angles_P;

	const double EPS = 2.2e-16;
	const double BAD_ROI_FVAL = 0.0;
};