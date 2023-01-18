#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"

/// @brief Gray Level Size Zone(GLSZM) features
/// Gray Level Size Zone(GLSZM) quantifies gray level zones in an image.A gray level zone is defined as a the number
/// of connected voxels that share the same gray level intensity.A voxel is considered connected if the distance is 1
/// according to the infinity norm(26 - connected region in a 3D, 8 - connected region in 2D).
/// In a gray level size zone matrix : math:`P(i, j)` the :math:`(i, j)^ {\text{ th }}` element equals the number of zones
/// with gray level : math:`i`and size :math:`j` appear in image.Contrary to GLCMand GLRLM, the GLSZM is rotation
/// independent, with only one matrix calculated for all directions in the ROI.

class GLSZMFeature: public FeatureMethod
{
public:
	GLSZMFeature ();

	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Compatibility with the manual reduce
	static bool required(const FeatureSet& fs) {
		return fs.anyEnabled({
			GLSZM_SAE,
			GLSZM_LAE,
			GLSZM_GLN,
			GLSZM_GLNN,
			GLSZM_SZN,
			GLSZM_SZNN,
			GLSZM_ZP,
			GLSZM_GLV,
			GLSZM_ZV,
			GLSZM_ZE,
			GLSZM_LGLZE,
			GLSZM_HGLZE,
			GLSZM_SALGLE,
			GLSZM_SAHGLE,
			GLSZM_LALGLE,
			GLSZM_LAHGLE
			});
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

private:
	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	int Ng = 0;	// number of discreet intensity values in the image
	int Ns = 0; // number of discreet zone sizes in the image
	int Np = 0; // number of voxels in the image
	int Nz = 0; // number of zones in the ROI, 1<=Nz<=Np
	SimpleMatrix<int> P;
	double sum_p = 0;

	const double EPS = 2.2e-16;
	const double BAD_ROI_FVAL = 0.0;
	const double LOG10_2 = 0.30102999566;	// precalculated log 2 base 10
};