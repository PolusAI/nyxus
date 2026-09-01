#pragma once

#include <vector>
#include <unordered_map>
#include "../dataset.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "../roi_cache.h"
#include "pixel.h"

/// @brief Features describing the radial intensity distribution within a ROI - the fraction of the
/// ROI's pixels at a given radius, the mean intensity of the pixels at that radius, and the
/// coefficient of variation of intensity across the wedges of a ring.
///
/// The three feature names are CellProfiler's RadialDistribution_* names, but the quantities are not
/// CellProfiler's: each member below documents what it actually returns. Which of the two sets of
/// semantics is the intended one is unresolved - the measured comparison against CellProfiler
/// MeasureObjectIntensityDistribution is in
/// tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md.
class RadialDistributionFeature: public FeatureMethod
{
public:
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::FRAC_AT_D, 
		Nyxus::Feature2D::MEAN_FRAC, 
		Nyxus::Feature2D::RADIAL_CV
	};

	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled (featureset);
	}

	RadialDistributionFeature(); 
	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void extract (LR& roi, const Fsettings& s);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);

	// Constants used in the output
	const static int num_bins = 8,
		num_features_FracAtD = 8,
		num_features_MeanFrac = 8,
		num_features_RadialCV = 8;

private:
	// Fraction of the ROI's pixel COUNT falling in a radial bin. Intensity never enters it, so this
	// is a function of the mask alone and not of the image.
	void get_FracAtD();

	// The radial bin's mean intensity, in the image's own units. Not a fraction, and not normalized
	// by the ROI's mean intensity.
	void get_MeanFrac();

	// Coefficient of variation of a ring's 8 wedge intensity SUMS, taken over all 8 wedges including
	// the empty ones, with the population (biased) standard deviation.
	void get_RadialCV();

	// Returns the index of the pixel in parameter 'cloud' having maximum distance from 'contour'
	size_t find_center_NT (const OutOfRamPixelCloud& cloud, const std::vector<Pixel2>& contour);

	// Zeros the counters
	void reset_buffers();

	// Return-ready feature values
	std::vector<double> values_FracAtD,
		values_MeanFrac,
		values_RadialCV;

	// Counters
	std::vector<int> radial_count_bins;
	std::vector<double> radial_intensity_bins;
	std::vector<std::vector<size_t>> banded_wedges;

	// Helpers
	int cached_center_x = -1, 
		cached_center_y = -1;
	int cached_num_pixels = 0;
	const double epsilon = 0.000000001;
};