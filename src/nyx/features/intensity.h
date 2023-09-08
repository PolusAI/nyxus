#pragma once
#include "../feature_method.h"

class PixelIntensityFeatures : public FeatureMethod
{
public:
	PixelIntensityFeatures();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value (std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();

	// list dependencies of this class
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled(
			{
				COV,
				COVERED_IMAGE_INTENSITY_RANGE,
				ENERGY,
				ENTROPY,
				EXCESS_KURTOSIS,
				HYPERFLATNESS,
				HYPERSKEWNESS,
				INTEGRATED_INTENSITY,
				INTERQUARTILE_RANGE,
				KURTOSIS,
				MAX,
				MEAN,
				MEAN_ABSOLUTE_DEVIATION,
				MEDIAN,
				MEDIAN_ABSOLUTE_DEVIATION,
				MIN,
				MODE,
				P01, P10, P25, P75, P90, P99,
				QCOD,
				RANGE,
				ROBUST_MEAN,
				ROBUST_MEAN_ABSOLUTE_DEVIATION,
				ROOT_MEAN_SQUARED,
				SKEWNESS,
				STANDARD_DEVIATION,
				STANDARD_DEVIATION_BIASED,
				STANDARD_ERROR,
				VARIANCE,
				VARIANCE_BIASED,
				UNIFORMITY,
				UNIFORMITY_PIU
			});
	}

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
