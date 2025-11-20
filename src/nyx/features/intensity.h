#pragma once
#include "../feature_method.h"

class PixelIntensityFeatures : public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::COV,
		Nyxus::Feature2D::COVERED_IMAGE_INTENSITY_RANGE,
		Nyxus::Feature2D::ENERGY,
		Nyxus::Feature2D::ENTROPY,
		Nyxus::Feature2D::EXCESS_KURTOSIS,
		Nyxus::Feature2D::HYPERFLATNESS,
		Nyxus::Feature2D::HYPERSKEWNESS,
		Nyxus::Feature2D::INTEGRATED_INTENSITY,
		Nyxus::Feature2D::INTERQUARTILE_RANGE,
		Nyxus::Feature2D::KURTOSIS,
		Nyxus::Feature2D::MAX,
		Nyxus::Feature2D::MEAN,
		Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION,
		Nyxus::Feature2D::MEDIAN,
		Nyxus::Feature2D::MEDIAN_ABSOLUTE_DEVIATION,
		Nyxus::Feature2D::MIN,
		Nyxus::Feature2D::MODE,
		Nyxus::Feature2D::P01, Nyxus::Feature2D::P10, Nyxus::Feature2D::P25, Nyxus::Feature2D::P75, Nyxus::Feature2D::P90, Nyxus::Feature2D::P99,
		Nyxus::Feature2D::QCOD,
		Nyxus::Feature2D::RANGE,
		Nyxus::Feature2D::ROBUST_MEAN,
		Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION,
		Nyxus::Feature2D::ROOT_MEAN_SQUARED,
		Nyxus::Feature2D::SKEWNESS,
		Nyxus::Feature2D::STANDARD_DEVIATION,
		Nyxus::Feature2D::STANDARD_DEVIATION_BIASED,
		Nyxus::Feature2D::STANDARD_ERROR,
		Nyxus::Feature2D::VARIANCE,
		Nyxus::Feature2D::VARIANCE_BIASED,
		Nyxus::Feature2D::UNIFORMITY,
		Nyxus::Feature2D::UNIFORMITY_PIU
	};

	PixelIntensityFeatures();
	void calculate(LR& roi, const Fsettings& settings) { throw std::runtime_error("wrong way, use a different overloaded PixelIntensityFeatures::calculate()"); }
	void calculate (LR& roi, const Fsettings & settings, const Dataset & dataset);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR & roi, const Fsettings & s, const Dataset & dataset, ImageLoader& ldr);
	void osized_calculate (LR& roi, const Fsettings& s, ImageLoader& ldr) { throw std::runtime_error("illegal use of PixelIntensityFeatures::osized_calculate()"); }
	void save_value (std::vector<std::vector<double>>& feature_vals);
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & settings, const Dataset & dataset);
	static void extract (LR & roi, const Fsettings & settings, const Dataset & dataset);
	void cleanup_instance();

	// list dependencies of this class
	static bool required(const FeatureSet& fs);

private:
	double
		val_INTEGRATED_INTENSITY = 0,
		val_MEAN = 0,
		val_MEDIAN = 0,
		val_MIN = 0,
		val_MAX = 0,
		val_RANGE = 0,
		val_COVERED_IMAGE_INTENSITY_RANGE = 0,
		val_STANDARD_DEVIATION = 0,
		val_STANDARD_ERROR = 0,
		val_SKEWNESS = 0,
		val_KURTOSIS = 0,
		val_EXCESS_KURTOSIS = 0,
		val_HYPERSKEWNESS = 0,
		val_HYPERFLATNESS = 0,
		val_MEAN_ABSOLUTE_DEVIATION = 0,
		val_MEDIAN_ABSOLUTE_DEVIATION = 0,
		val_ENERGY = 0,
		val_ROOT_MEAN_SQUARED = 0,
		val_ENTROPY = 0,
		val_MODE = 0,
		val_UNIFORMITY = 0,
		val_UNIFORMITY_PIU = 0,
		val_P01 = 0, val_P10 = 0, val_P25 = 0, val_P75 = 0, val_P90 = 0, val_P99 = 0,
		val_QCOD = 0,
		val_INTERQUARTILE_RANGE = 0,
		val_ROBUST_MEAN = 0, 
		val_ROBUST_MEAN_ABSOLUTE_DEVIATION = 0,
		val_COV = 0,
		val_STANDARD_DEVIATION_BIASED = 0,
		val_VARIANCE = 0,
		val_VARIANCE_BIASED = 0;
};
