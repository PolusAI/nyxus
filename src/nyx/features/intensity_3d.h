#pragma once
#include "intensity.h"

class PixelIntensityFeatures_3D : public FeatureMethod
{
public:
	PixelIntensityFeatures_3D();
	static bool required(const FeatureSet& fs)
	{
		return fs.anyEnabled({ 
			D3_INTEGRATED_INTENSITY,
			D3_MEAN,
			D3_MEDIAN,
			D3_MIN,
			D3_MAX,
			D3_RANGE,
			D3_STANDARD_DEVIATION,
			D3_STANDARD_ERROR,
			D3_SKEWNESS,
			D3_KURTOSIS,
			D3_HYPERSKEWNESS,
			D3_HYPERFLATNESS,
			D3_MEAN_ABSOLUTE_DEVIATION,
			D3_ENERGY,
			D3_ROOT_MEAN_SQUARED,
			D3_ENTROPY,
			D3_MODE,
			D3_UNIFORMITY,
			D3_UNIFORMITY_PIU,
			D3_P01, D3_P10, D3_P25, D3_P75, D3_P90, D3_P99,
			D3_INTERQUARTILE_RANGE,
			D3_ROBUST_MEAN_ABSOLUTE_DEVIATION 
		});
	}
	void calculate(LR& r);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();

	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);

protected:

	double
		val_INTEGRATED_INTENSITY = 0,
		val_MEAN = 0,
		val_MEDIAN = 0,
		val_MIN = 0,
		val_MAX = 0,
		val_RANGE = 0,
		val_STANDARD_DEVIATION = 0,
		val_STANDARD_ERROR = 0,
		val_SKEWNESS = 0,
		val_KURTOSIS = 0,
		val_HYPERSKEWNESS = 0,
		val_HYPERFLATNESS = 0,
		val_MEAN_ABSOLUTE_DEVIATION = 0,
		val_ENERGY = 0,
		val_ROOT_MEAN_SQUARED = 0,
		val_ENTROPY = 0,
		val_MODE = 0,
		val_UNIFORMITY = 0,
		val_UNIFORMITY_PIU = 0,
		val_P01 = 0, val_P10 = 0, val_P25 = 0, val_P75 = 0, val_P90 = 0, val_P99 = 0,
		val_INTERQUARTILE_RANGE = 0,
		val_ROBUST_MEAN_ABSOLUTE_DEVIATION = 0;
};