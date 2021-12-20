#pragma once

#include <unordered_map>
#include "../roi_data.h"
#include "image_matrix.h"

// Inspired by https://qiita.com/tatsunidas/items/50f4bee7236eb0392aaf

/// @brief Neighbouring Gray Tone Difference Matrix (NGTDM) features
/// Neighbouring Gray Tone Difference Matrix quantifies the difference between a gray value and the average gray value
/// of its neighbours within distance : math:`\delta`. The sum of absolute differences for gray level : math:`i` is stored in the matrix.
/// Let :math:`\textbf{ X }_{ gl }` be a set of segmented voxelsand :math:`x_{gl}(j_x, j_y, j_z) \in \textbf{ X }_{ gl }` be the gray level of a voxel at postion
/// 	: math:`(j_x, j_y, j_z)`, then the average gray level of the neigbourhood is :
/// 
/// 	..math::
/// 
/// 		\bar{ A }_i &= \bar{ A }(j_x, j_y, j_z) \\
/// 		&= \displaystyle\frac{ 1 }{W} \displaystyle\sum_{ k_x = -\delta }^ {\delta}\displaystyle\sum_{ k_y = -\delta }^ {\delta}
/// 	\displaystyle\sum_{ k_z = -\delta }^ {\delta} {x_{ gl }(j_x + k_x, j_y + k_y, j_z + k_z)}, \\
/// 		& \mbox{ where }(k_x, k_y, k_z)\neq(0, 0, 0)\mbox{ and } x_{ gl }(j_x + k_x, j_y + k_y, j_z + k_z) \in \textbf{ X }_{ gl }

class NGTDM_features
{
	using P_matrix = SimpleMatrix<int>;

public:
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled({
			NGTDM_COARSENESS,
			NGTDM_CONTRAST,
			NGTDM_BUSYNESS,
			NGTDM_COMPLEXITY,
			NGTDM_STRENGTH
			});
	}

	NGTDM_features (int minI, int maxI, const ImageMatrix& im);

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

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	int Ng = 0;	// number of discreet intensity values in the image
	int Ngp = 0; // number of non-zero gray levels. Since we keep only informative (non-zero) levels, Ngp is always ==Ng
	int Nvp = 0;	// number of "valid voxels" i.e. those voxels that have at least 1 neighbor
	int Nd = 0; // number of discreet dependency sizes in the image
	int Nz = 0; // number of dependency zones in the ROI, Nz = sum(sum(P[i,j]))
	std::vector <double> P, S;
	std::vector<int> N;

	const double BAD_ROI_FVAL = 0.0;
	const double EPS = 2.2e-16;
};