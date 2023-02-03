#include "../environment.h"
#include "../parallel.h"
#include "histogram.h"
#include "intensity.h"
#include "pixel.h"


PixelIntensityFeatures::PixelIntensityFeatures() : FeatureMethod("PixelIntensityFeatures")
{
	provide_features({
		INTEGRATED_INTENSITY,
		MEAN,
		MEDIAN,
		MIN,
		MAX,
		RANGE,
		STANDARD_DEVIATION,
		STANDARD_ERROR,
		SKEWNESS,
		KURTOSIS,
		HYPERSKEWNESS,
		HYPERFLATNESS,
		MEAN_ABSOLUTE_DEVIATION,
		ENERGY,
		ROOT_MEAN_SQUARED,
		ENTROPY,
		MODE,
		UNIFORMITY,
		UNIFORMITY_PIU,
		P01, P10, P25, P75, P90, P99,
		INTERQUARTILE_RANGE,
		ROBUST_MEAN_ABSOLUTE_DEVIATION
		});
}

void PixelIntensityFeatures::calculate(LR& r)
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

void PixelIntensityFeatures::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void PixelIntensityFeatures::osized_calculate(LR& r, ImageLoader& imloader)
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
	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)
	{
		Pixel2 px = r.raw_pixels_NT[i];
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
	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)
	{
		Pixel2 px = r.raw_pixels_NT[i];
		mad += std::abs(px.inten - mean_);
		var += (px.inten - mean_) * (px.inten - mean_);
	}
	val_MEAN_ABSOLUTE_DEVIATION = mad / n;
	var /= n;
	double stddev = sqrt(var);
	val_STANDARD_DEVIATION = stddev;

	// --Standard error
	val_STANDARD_ERROR = stddev / sqrt(n);

	//==== Do not calculate features of all-blank intensities (to avoid NANs)
	if (r.aux_min == 0 && r.aux_max == 0)
		return;

	// P10, 25, 75, 90, IQR, RMAD, entropy, uniformity
	TrivialHistogram H;
	H.initialize(r.aux_min, r.aux_max, r.raw_pixels_NT);
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

	// --Uniformity calculated as PIU, percent image uniformity - see "A comparison of five standard methods for evaluating image 
	//	intensity uniformity in partially parallel imaging MRI" [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/] 
	//	and https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.2241606
	double piu = (1.0 - double(r.aux_max - r.aux_min) / double(r.aux_max + r.aux_min)) * 100.0;
	val_UNIFORMITY_PIU = piu;

	// Skewness
	Moments4 mom;
	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)
	{
		Pixel2 px = r.raw_pixels_NT[i];
		mom.add(px.inten);
	}

	val_SKEWNESS = mom.skewness();

	// Kurtosis
	val_KURTOSIS = mom.kurtosis();

	// Hyperskewness hs = E[x-mean].^5 / std(x).^5
	val_HYPERSKEWNESS = mom.hyperskewness();

	// Hyperflatness hf = E[x-mean].^6 / std(x).^6
	val_HYPERFLATNESS = mom.hyperflatness();
}

void PixelIntensityFeatures::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[INTEGRATED_INTENSITY][0] = val_INTEGRATED_INTENSITY;
	fvals[MEAN][0] = val_MEAN;
	fvals[MEDIAN][0] = val_MEDIAN;
	fvals[MIN][0] = val_MIN;
	fvals[MAX][0] = val_MAX;
	fvals[RANGE][0] = val_RANGE;
	fvals[STANDARD_DEVIATION][0] = val_STANDARD_DEVIATION;
	fvals[STANDARD_ERROR][0] = val_STANDARD_ERROR;
	fvals[SKEWNESS][0] = val_SKEWNESS;
	fvals[KURTOSIS][0] = val_KURTOSIS;
	fvals[HYPERSKEWNESS][0] = val_HYPERSKEWNESS;
	fvals[HYPERFLATNESS][0] = val_HYPERFLATNESS;
	fvals[MEAN_ABSOLUTE_DEVIATION][0] = val_MEAN_ABSOLUTE_DEVIATION;
	fvals[ENERGY][0] = val_ENERGY;
	fvals[ROOT_MEAN_SQUARED][0] = val_ROOT_MEAN_SQUARED;
	fvals[ENTROPY][0] = val_ENTROPY;
	fvals[MODE][0] = val_MODE;
	fvals[UNIFORMITY][0] = val_UNIFORMITY;
	fvals[UNIFORMITY_PIU][0] = val_UNIFORMITY_PIU;
	fvals[P01][0] = val_P01;
	fvals[P10][0] = val_P10;
	fvals[P25][0] = val_P25;
	fvals[P75][0] = val_P75;
	fvals[P90][0] = val_P90;
	fvals[P99][0] = val_P99;
	fvals[INTERQUARTILE_RANGE][0] = val_INTERQUARTILE_RANGE;
	fvals[ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = val_ROBUST_MEAN_ABSOLUTE_DEVIATION;
}

void PixelIntensityFeatures::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(PixelIntensityFeatures::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void PixelIntensityFeatures::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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
		PixelIntensityFeatures f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void PixelIntensityFeatures::reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		PixelIntensityFeatures f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void PixelIntensityFeatures::cleanup_instance()
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

