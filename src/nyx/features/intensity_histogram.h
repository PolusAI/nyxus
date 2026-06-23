#pragma once
#include <utility>
#include <vector>
#include "../feature_method.h"

// IBSI Intensity Histogram (IH) family (46 features). All features are derived from a
// single dedicated N-bin histogram over the per-ROI intensity range [min,max]
// (N = coarse gray depth):
//   * histogram = N equal-width bins over [aux_min, aux_max];
//   * median = center of the bin where the running count first exceeds count/2
//     (bin-center median, NOT an interpolated quantile);
//   * p10/p25/p75/p90 = interpolated histogram quantiles;
//   * "...Index" features = 1-based bin index of the corresponding value
//     (floor((v-min)/binW), clamped to [0,N-1]).
//
// For floating-point images the histogram is built in the original float intensity
// domain (see float_domain_map()); for integer images it uses the native values.
//
// Availability is gated on IBSI mode: the family is only enabled when IBSI
// compliance is on (see Environment::expand_featuregroups()). A defensive
// compute-time guard fills NaN if invoked with IBSI off.

class IntensityHistogramFeatures : public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		// "...Value" family (20)
		Nyxus::Feature2D::IH_MEAN_VAL,
		Nyxus::Feature2D::IH_VARIANCE_VAL,
		Nyxus::Feature2D::IH_SKEWNESS_VAL,
		Nyxus::Feature2D::IH_EXCESS_KURTOSIS_VAL,
		Nyxus::Feature2D::IH_MEDIAN_VAL,
		Nyxus::Feature2D::IH_MINIMUM_VAL,
		Nyxus::Feature2D::IH_P10_VAL,
		Nyxus::Feature2D::IH_P90_VAL,
		Nyxus::Feature2D::IH_MAXIMUM_VAL,
		Nyxus::Feature2D::IH_MODE_VAL,
		Nyxus::Feature2D::IH_INTERQUANTILE_RANGE_VAL,
		Nyxus::Feature2D::IH_RANGE_VAL,
		Nyxus::Feature2D::IH_MEAN_ABSOLUTE_DEVIATION_VAL,
		Nyxus::Feature2D::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL,
		Nyxus::Feature2D::IH_MEDIAN_ABSOLUTE_DEVIATION_VAL,
		Nyxus::Feature2D::IH_COEFFICIENT_OF_VARIATION_VAL,
		Nyxus::Feature2D::IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL,
		Nyxus::Feature2D::IH_ENTROPY_VAL,
		Nyxus::Feature2D::IH_UNIFORMITY_VAL,
		Nyxus::Feature2D::IH_ROBUST_MEAN_VAL,
		// "...Index" family (19)
		Nyxus::Feature2D::IH_MEAN_IDX,
		Nyxus::Feature2D::IH_VARIANCE_IDX,
		Nyxus::Feature2D::IH_SKEWNESS_IDX,
		Nyxus::Feature2D::IH_EXCESS_KURTOSIS_IDX,
		Nyxus::Feature2D::IH_MEDIAN_IDX,
		Nyxus::Feature2D::IH_MINIMUM_IDX,
		Nyxus::Feature2D::IH_P10_IDX,
		Nyxus::Feature2D::IH_P90_IDX,
		Nyxus::Feature2D::IH_MAXIMUM_IDX,
		Nyxus::Feature2D::IH_MODE_IDX,
		Nyxus::Feature2D::IH_INTERQUANTILE_RANGE_IDX,
		Nyxus::Feature2D::IH_RANGE_IDX,
		Nyxus::Feature2D::IH_MEAN_ABSOLUTE_DEVIATION_IDX,
		Nyxus::Feature2D::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX,
		Nyxus::Feature2D::IH_MEDIAN_ABSOLUTE_DEVIATION_IDX,
		Nyxus::Feature2D::IH_COEFFICIENT_OF_VARIATION_IDX,
		Nyxus::Feature2D::IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX,
		Nyxus::Feature2D::IH_ENTROPY_IDX,
		Nyxus::Feature2D::IH_UNIFORMITY_IDX,
		// gradient + bookkeeping (7)
		Nyxus::Feature2D::IH_MAX_GRADIENT,
		Nyxus::Feature2D::IH_MAX_GRADIENT_IDX,
		Nyxus::Feature2D::IH_MIN_GRADIENT,
		Nyxus::Feature2D::IH_MIN_GRADIENT_IDX,
		Nyxus::Feature2D::IH_ROBUST_MEAN_IDX,
		Nyxus::Feature2D::IH_NUM_BINS,
		Nyxus::Feature2D::IH_BIN_SIZE
	};

	IntensityHistogramFeatures();

	void calculate (LR& roi, const Fsettings& settings) { throw std::runtime_error("wrong way, use IntensityHistogramFeatures::calculate(roi, settings, dataset)"); }
	void calculate (LR& roi, const Fsettings& settings, const Dataset& dataset);
	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& roi, const Fsettings& s, const Dataset& dataset, ImageLoader& ldr);
	void osized_calculate (LR& roi, const Fsettings& s, ImageLoader& ldr) { throw std::runtime_error("illegal use of IntensityHistogramFeatures::osized_calculate()"); }
	void save_value (std::vector<std::vector<double>>& feature_vals);
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings& settings, const Dataset& dataset);
	static void extract (LR& roi, const Fsettings& settings, const Dataset& dataset);
	void cleanup_instance();

	// list dependencies of this class
	static bool required (const FeatureSet& fs);

private:
	// Core routine: fills 'result_' (one entry per IH feature code) from the
	// per-ROI histogram. Pixel intensities are mapped as (poffset + pscale*inten)
	// before binning: for float images this undoes Nyxus's load-time uint
	// quantization so the histogram lives in the original float domain;
	// for integer images pscale=1/poffset=0 is a no-op. minVal/maxVal are the ROI
	// bounds in that same (mapped) domain.
	template <class Pixelcloud>
	void compute (const Pixelcloud& pixels, double minVal, double maxVal, size_t count, int nbins,
	              double nan_val, double pscale, double poffset);
	void fill_invalid (double nan_val);

	// Derive the affine (pscale,poffset) that maps stored uint intensities back to the
	// original float domain for float images (no-op for integer images / no slide).
	static void float_domain_map (LR& r, const Fsettings& s, const Dataset& ds,
	                              double& pscale, double& poffset);

	// (feature code, value) pairs in the order produced by compute(); written out
	// verbatim by save_value(). Cleared by cleanup_instance().
	std::vector<std::pair<int, double>> result_;
};
