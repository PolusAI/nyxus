#pragma once
#include "../feature_method.h"

class D3_PixelIntensityFeatures : public FeatureMethod
{
public:
	D3_PixelIntensityFeatures();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();

	// list dependencies of this class
	static bool required (const FeatureSet& fs);

	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
			Nyxus::Feature3D::COV,
			Nyxus::Feature3D::COVERED_IMAGE_INTENSITY_RANGE,
			Nyxus::Feature3D::ENERGY,
			Nyxus::Feature3D::ENTROPY,
			Nyxus::Feature3D::EXCESS_KURTOSIS,
			Nyxus::Feature3D::HYPERFLATNESS,
			Nyxus::Feature3D::HYPERSKEWNESS,
			Nyxus::Feature3D::INTEGRATED_INTENSITY,
			Nyxus::Feature3D::INTERQUARTILE_RANGE,
			Nyxus::Feature3D::KURTOSIS,
			Nyxus::Feature3D::MAX,
			Nyxus::Feature3D::MEAN,
			Nyxus::Feature3D::MEAN_ABSOLUTE_DEVIATION,
			Nyxus::Feature3D::MEDIAN,
			Nyxus::Feature3D::MEDIAN_ABSOLUTE_DEVIATION,
			Nyxus::Feature3D::MIN,
			Nyxus::Feature3D::MODE,
			Nyxus::Feature3D::P01,
			Nyxus::Feature3D::P10,
			Nyxus::Feature3D::P25,
			Nyxus::Feature3D::P75,
			Nyxus::Feature3D::P90,
			Nyxus::Feature3D::P99,
			Nyxus::Feature3D::QCOD,
			Nyxus::Feature3D::RANGE,
			Nyxus::Feature3D::ROBUST_MEAN,
			Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION,
			Nyxus::Feature3D::ROOT_MEAN_SQUARED,
			Nyxus::Feature3D::SKEWNESS,
			Nyxus::Feature3D::STANDARD_DEVIATION,
			Nyxus::Feature3D::STANDARD_DEVIATION_BIASED,
			Nyxus::Feature3D::STANDARD_ERROR,
			Nyxus::Feature3D::VARIANCE,
			Nyxus::Feature3D::VARIANCE_BIASED,
			Nyxus::Feature3D::UNIFORMITY,
			Nyxus::Feature3D::UNIFORMITY_PIU
	};

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
