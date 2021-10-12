#pragma once

#include "image_matrix.h"


class GLRLM_features
{
	using P_matrix = SimpleMatrix<int>;
	using AngledFtrs = std::vector<double>;

public:

	GLRLM_features() {}
	void initialize(int minI, int maxI, const ImageMatrix& im);

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

	static std::vector<double> rotAngles;

protected:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	std::vector<int> angles_Ng;	// number of discreet intensity values in the image
	std::vector<int> angles_Nr; // number of discreet run lengths in the image
	std::vector<int> angles_Np; // number of voxels in the image
	//XXX int Nz = 0; // number of zones in the ROI, 1<=Nz<=Np
	std::vector<P_matrix> angles_P;
};