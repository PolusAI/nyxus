#pragma once

#include "image_matrix.h"

class GLSZM_features
{
public:
	GLSZM_features() {}

	void initialize (int minI, int maxI, const ImageMatrix & );

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

protected:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	int Ng = 0;	// number of discreet intensity values in the image
	int Ns = 0; // number of discreet zone sizes in the image
	int Np = 0; // number of voxels in the image
	int Nz = 0; // number of zones in the ROI, 1<=Nz<=Np
	SimpleMatrix<int> P;
};