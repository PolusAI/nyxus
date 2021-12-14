#pragma once
#include <unordered_map>
#include "../roi_data.h"
#include "image_matrix.h"

// Inspired by 
//		https://stackoverflow.com/questions/25019840/neighboring-gray-level-dependence-matrix-ngldm-in-matlab?fbclid=IwAR14fT0kpmjmOXRhKcguFMH3tCg0G4ubDLRxyHZoXdpKdbPxF7Zuq-WKE8o
//		https://qiita.com/tatsunidas/items/fd49ef6ac7c3deb141e0

class GLDM_features
{
	using P_matrix = SimpleMatrix<int>;

public:
	GLDM_features() {}
	void initialize(int minI, int maxI, const ImageMatrix& im);

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

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

protected:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	int Ng = 0;	// number of discreet intensity values in the image
	int Nd = 0; // number of discreet dependency sizes in the image
	int Nz = 0; // number of dependency zones in the ROI, Nz = sum(sum(P[i,j]))
	SimpleMatrix<int> P;	// dependence matrix
};

