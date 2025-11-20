#pragma once

#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "image_matrix.h"
#include "texture_feature.h"


class D3_NGTDM_feature : public FeatureMethod, public TextureFeature
{
public:

	// Codes of features implemented by this class. Used in feature manager's mechanisms, 
	// in the feature group nickname expansion, and in the feature value output 
	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
		Nyxus::Feature3D::NGTDM_COARSENESS,
		Nyxus::Feature3D::NGTDM_CONTRAST,
		Nyxus::Feature3D::NGTDM_BUSYNESS,
		Nyxus::Feature3D::NGTDM_COMPLEXITY,
		Nyxus::Feature3D::NGTDM_STRENGTH
	};

	D3_NGTDM_feature();
	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
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

	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);
	static void extract (LR& r, const Fsettings& s);

	// Comaptibility with manual reduce
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled (D3_NGTDM_feature::featureset);
	}

	static int n_levels; // default value: 0

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

	std::vector<PixIntens> I;	// sorted unique intensities after image greyscale binning

	void clear_buffers();

	const double BAD_ROI_FVAL = 0.0;
	const double EPS = 2.2e-16;

	double _coarseness = 0,
		_contrast = 0,
		_busyness = 0,
		_complexity = 0,
		_strength = 0;

};

