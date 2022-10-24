#include "../environment.h"
#include "../parallel.h"
#include "histogram.h"
#include "intensity_3d.h"
#include "pixel.h"

PixelIntensityFeatures_3D::PixelIntensityFeatures_3D() : 
	FeatureMethod("PixelIntensityFeatures_3D")
{
	provide_features({
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

void PixelIntensityFeatures_3D::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[D3_INTEGRATED_INTENSITY][0] = val_INTEGRATED_INTENSITY;
	fvals[D3_MEAN][0] = val_MEAN;
	fvals[D3_MEDIAN][0] = val_MEDIAN;
	fvals[D3_MIN][0] = val_MIN;
	fvals[D3_MAX][0] = val_MAX;
	fvals[D3_RANGE][0] = val_RANGE;
	fvals[D3_STANDARD_DEVIATION][0] = val_STANDARD_DEVIATION;
	fvals[D3_STANDARD_ERROR][0] = val_STANDARD_ERROR;
	fvals[D3_SKEWNESS][0] = val_SKEWNESS;
	fvals[D3_KURTOSIS][0] = val_KURTOSIS;
	fvals[D3_HYPERSKEWNESS][0] = val_HYPERSKEWNESS;
	fvals[D3_HYPERFLATNESS][0] = val_HYPERFLATNESS;
	fvals[D3_MEAN_ABSOLUTE_DEVIATION][0] = val_MEAN_ABSOLUTE_DEVIATION;
	fvals[D3_ENERGY][0] = val_ENERGY;
	fvals[D3_ROOT_MEAN_SQUARED][0] = val_ROOT_MEAN_SQUARED;
	fvals[D3_ENTROPY][0] = val_ENTROPY;
	fvals[D3_MODE][0] = val_MODE;
	fvals[D3_UNIFORMITY][0] = val_UNIFORMITY;
	fvals[D3_UNIFORMITY_PIU][0] = val_UNIFORMITY_PIU;
	fvals[D3_P01][0] = val_P01;
	fvals[D3_P10][0] = val_P10;
	fvals[D3_P25][0] = val_P25;
	fvals[D3_P75][0] = val_P75;
	fvals[D3_P90][0] = val_P90;
	fvals[D3_P99][0] = val_P99;
	fvals[D3_INTERQUARTILE_RANGE][0] = val_INTERQUARTILE_RANGE;
	fvals[D3_ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = val_ROBUST_MEAN_ABSOLUTE_DEVIATION;
}

void PixelIntensityFeatures_3D::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel (PixelIntensityFeatures_3D::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void PixelIntensityFeatures_3D::reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		PixelIntensityFeatures_3D f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void PixelIntensityFeatures_3D::calculate (LR& r)
{
	// --MIN, MAX
	val_MIN = r.aux_min;
	val_MAX = r.aux_max;
	val_RANGE = val_MAX - val_MIN;

	double n = r.aux_area;

	// --MEAN, ENERGY, CENTROID_XY
	double mean_ = 0.0;
	double energy = 0.0;
	double cen_x = 0.0,
		cen_y = 0.0,
		integInten = 0.0;
	for (auto& px : r.raw_pixels)
	{
		mean_ += px.inten;
		energy += px.inten * px.inten;
		cen_x += px.x;
		cen_y += px.y;
		integInten += px.inten;
	}
	mean_ /= n;
	val_MEAN = mean_;
	val_ENERGY = energy;
	val_ROOT_MEAN_SQUARED = sqrt(val_ENERGY / n);
	val_INTEGRATED_INTENSITY = integInten;

	// --MAD, VARIANCE, STDDEV
	double mad = 0.0,
		var = 0.0;
	for (auto& px : r.raw_pixels)
	{
		double diff = px.inten - mean_;
		mad += std::abs(diff);
		var += diff * diff;
	}
	val_MEAN_ABSOLUTE_DEVIATION = mad / n;
	var = n > 1 ? var / (n - 1) : 0.0;
	double stddev = sqrt(var);
	val_STANDARD_DEVIATION = stddev;

	// --Standard error
	val_STANDARD_ERROR = stddev / sqrt(n);

	//==== Do not calculate features of all-blank intensities (to avoid NANs)
	if (r.aux_min == 0 && r.aux_max == 0)
		return;

	// P10, 25, 75, 90, IQR, RMAD, entropy, uniformity
	TrivialHistogram H;
	H.initialize(r.aux_min, r.aux_max, r.raw_pixels);
	auto [median_, mode_, p01_, p10_, p25_, p75_, p90_, p99_, iqr_, rmad_, entropy_, uniformity_] = H.get_stats();
	val_MEDIAN = median_;
	val_P01 = p01_;
	val_P10 = p10_;
	val_P25 = p25_;
	val_P75 = p75_;
	val_P90 = p90_;
	val_P99 = p99_;
	val_INTERQUARTILE_RANGE = iqr_;
	val_ROBUST_MEAN_ABSOLUTE_DEVIATION = rmad_;
	val_ENTROPY = entropy_;
	val_MODE = mode_;
	val_UNIFORMITY = uniformity_;

	// --Uniformity calculated as PIU, percent image uniformity - see "A comparison of five standard methods for evaluating image intensity uniformity in partially parallel imaging MRI" [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/] and https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.2241606
	double piu = (1.0 - double(r.aux_max - r.aux_min) / double(r.aux_max + r.aux_min)) * 100.0;
	val_UNIFORMITY_PIU = piu;

	// Skewness
	Moments4 mom;
	for (auto& px : r.raw_pixels)
		mom.add(px.inten);
	val_SKEWNESS = mom.skewness();

	// Kurtosis
	val_KURTOSIS = mom.kurtosis();

	double sumPow5 = 0, sumPow6 = 0;
	for (auto& px : r.raw_pixels)
	{
		double diff = px.inten - mean_;
		sumPow5 += std::pow(diff, 5.);
		sumPow6 += std::pow(diff, 6.);
	}

	// Hyperskewness
	double denom = (n * std::pow(val_STANDARD_DEVIATION, 5.));
	val_HYPERSKEWNESS = denom == 0. ? 0. : sumPow5 / denom;

	// Hyperflatness
	denom = (n * std::pow(val_STANDARD_DEVIATION, 6.));
	val_HYPERFLATNESS = denom == 0. ? 0. : sumPow6 / denom;
}

void PixelIntensityFeatures_3D::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	// Calculate the feature for each batch ROI item 
	for (auto i = firstitem; i < lastitem; i++)
	{
		// Get ahold of ROI's label and cache
		int roiLabel = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[roiLabel];

		// Skip the ROI if its data is invalid to prevent nans and infs in the output
		if (r.has_bad_data())
			continue;

		// Calculate the feature and save it in ROI's csv-friendly buffer 'fvals'
		PixelIntensityFeatures_3D f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void PixelIntensityFeatures_3D::cleanup_instance()
{
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
}

void PixelIntensityFeatures_3D::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void PixelIntensityFeatures_3D::osized_calculate(LR& r, ImageLoader& imloader)
{
}
