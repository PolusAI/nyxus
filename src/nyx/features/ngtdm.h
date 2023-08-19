#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "../feature_method.h"
#include "image_matrix.h"

// Inspired by https://qiita.com/tatsunidas/items/50f4bee7236eb0392aaf

/// @brief Neighbouring Gray Tone Difference Matrix (NGTDM) features
/// Neighbouring Gray Tone Difference Matrix quantifies the difference between a gray value and the average gray value
/// of its neighbours within distance : math:`\delta`. The sum of absolute differences for gray level : math:`i` is stored in the matrix.
/// Let :math:`\textbf{ X }_{ gl }` be a set of segmented voxelsand :math:`x_{gl}(j_x, j_y, j_z) \in \textbf{ X }_{ gl }` be the gray level of a voxel at position
/// 	: math:`(j_x, j_y, j_z)`, then the average gray level of the neighborhood is :
/// 
/// 	..math::
/// 
/// 		\bar{ A }_i &= \bar{ A }(j_x, j_y, j_z) \\
/// 		&= \displaystyle\frac{ 1 }{W} \displaystyle\sum_{ k_x = -\delta }^ {\delta}\displaystyle\sum_{ k_y = -\delta }^ {\delta}
/// 	\displaystyle\sum_{ k_z = -\delta }^ {\delta} {x_{ gl }(j_x + k_x, j_y + k_y, j_z + k_z)}, \\
/// 		& \mbox{ where }(k_x, k_y, k_z)\neq(0, 0, 0)\mbox{ and } x_{ gl }(j_x + k_x, j_y + k_y, j_z + k_z) \in \textbf{ X }_{ gl }

class NGTDMFeature: public FeatureMethod
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::AvailableFeatures> featureset =
	{
		NGTDM_COARSENESS,
		NGTDM_CONTRAST,
		NGTDM_BUSYNESS,
		NGTDM_COMPLEXITY,
		NGTDM_STRENGTH
	};

	NGTDMFeature(); 
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);

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

	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

	// Comaptibility with manual reduce
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled (NGTDMFeature::featureset);
	}

private:

	bool bad_roi_data = false;	// used to prevent calculation of degenerate ROIs
	int Ng = 0;	// number of discrete intensity values in the image
	int Ngp = 0; // number of non-zero gray levels. Since we keep only informative (non-zero) levels, Ngp is always ==Ng
	int Nvp = 0;	// number of "valid voxels" i.e. those voxels that have at least 1 neighbor
	int Nd = 0; // number of discrete dependency sizes in the image
	int Nz = 0; // number of dependency zones in the ROI, Nz = sum(sum(P[i,j]))
	double Nvc = 0; // sum of N vector
	std::vector <double> P, S;
	std::vector<int> N;

	void clear_buffers();

	const double BAD_ROI_FVAL = 0.0;
	const double EPS = 2.2e-16;

	double _coarseness = 0, 
		_contrast = 0, 
		_busyness = 0, 
		_complexity = 0, 
		_strength = 0;
};