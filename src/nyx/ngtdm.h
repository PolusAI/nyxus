#pragma once

#include "image_matrix.h"

// https://qiita.com/tatsunidas/items/50f4bee7236eb0392aaf

class NGTDM_features
{
	using P_matrix = SimpleMatrix<int>;
public:
	NGTDM_features() {}
	void initialize(int minI, int maxI, const ImageMatrix& im);

	// Coarseness
	double calc_Coarseness();
	// Contrast
	double calc_Contrast();
	// Busyness
	double calc_Busyness();
	// Complexity
	double calc_Complexity();
	// Strength
	double calc_Strength();

protected:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	int Ng = 0;	// number of discreet intensity values in the image
	int Ngp = 0; // number of non-zero gray levels. Since we keep only informative (non-zero) levels, Ngp is always ==Ng
	int Nvp = 0;	// number of "valid voxels" i.e. those voxels that have at least 1 neighbor
	int Nd = 0; // number of discreet dependency sizes in the image
	int Nz = 0; // number of dependency zones in the ROI, Nz = sum(sum(P[i,j]))
	std::vector <double> P, S;
	std::vector<int> N;
};