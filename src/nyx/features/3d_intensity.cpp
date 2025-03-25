#include "../environment.h"
#include "../parallel.h"
#include "histogram.h"
#include "3d_intensity.h"
#include "pixel.h"

using namespace Nyxus;

bool D3_PixelIntensityFeatures::required (const FeatureSet & fs)
{
	return fs.anyEnabled(
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
		});
}

D3_PixelIntensityFeatures::D3_PixelIntensityFeatures() : FeatureMethod("PixelIntensityFeatures_3D")
{
	provide_features({
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
			Nyxus::Feature3D::UNIFORMITY,
			Nyxus::Feature3D::UNIFORMITY_PIU,
			Nyxus::Feature3D::VARIANCE,
			Nyxus::Feature3D::VARIANCE_BIASED,
		});
}

void D3_PixelIntensityFeatures::calculate(LR& r)
{
	// --MIN, MAX
	val_MIN = r.aux_min;
	val_MAX = r.aux_max;
	val_RANGE = val_MAX - val_MIN;

	// --COVERED_IMAGE_INTENSITY_RANGE
	if (r.slide_idx >= 0)
	{
		const SlideProps& p = LR::dataset_props[r.slide_idx];
		double sir = (p.max_preroi_inten - p.min_preroi_inten); // slide intensity range
		val_COVERED_IMAGE_INTENSITY_RANGE = double(r.aux_max - r.aux_min) / sir;
	}
	else
		val_COVERED_IMAGE_INTENSITY_RANGE = 1;

	double n = r.aux_area;

	// --MEAN, ENERGY, CENTROID_XY
	double mean_ = 0.0;
	double energy = 0.0;
	double cen_x = 0.0,
		cen_y = 0.0,
		cen_z = 0.0,
		integInten = 0.0;
	for (auto& px : r.raw_pixels_3D)
	{
		mean_ += px.inten;
		energy += px.inten * px.inten;
		cen_x += px.x;
		cen_y += px.y;
		cen_z += px.z;
		integInten += px.inten;
	}
	mean_ /= n;
	val_MEAN = mean_;
	val_ENERGY = energy;
	val_ROOT_MEAN_SQUARED = sqrt(val_ENERGY / n);
	val_INTEGRATED_INTENSITY = integInten;

	// --MAD, VARIANCE, STDDEV, COV
	double mad = 0.0,
		var = 0.0;
	for (auto& px : r.raw_pixels_3D)
	{
		double diff = px.inten - mean_;
		mad += std::abs(diff);
		var += diff * diff;
	}
	val_MEAN_ABSOLUTE_DEVIATION = mad / n;
	val_VARIANCE = n > 1 ? var / (n - 1) : 0.0;
	val_VARIANCE_BIASED = n > 1 ? var / n : 0.0;
	val_STANDARD_DEVIATION = sqrt(val_VARIANCE);
	val_STANDARD_DEVIATION_BIASED = sqrt(val_VARIANCE_BIASED);
	val_COV = val_STANDARD_DEVIATION / mean_;

	// --Standard error
	val_STANDARD_ERROR = val_STANDARD_DEVIATION / sqrt(n);

	//==== Do not calculate features of all-blank intensities (to avoid NANs)
	if (r.aux_min == 0 && r.aux_max == 0)
		return;

	// P10, 25, 75, 90, IQR, QCOD, RMAD, entropy, uniformity
	TrivialHistogram H;
	H.initialize(r.aux_min, r.aux_max, r.raw_pixels_3D);
	auto [median_, mode_, p01_, p10_, p25_, p75_, p90_, p99_, iqr_, rmad_, entropy_, uniformity_] = H.get_stats();
	val_MEDIAN = median_;
	val_P01 = p01_;
	val_P10 = p10_;
	val_P25 = p25_;
	val_P75 = p75_;
	val_P90 = p90_;
	val_P99 = p99_;
	val_QCOD = (p75_ - p25_) / (p75_ + p25_);
	val_INTERQUARTILE_RANGE = iqr_;
	val_ROBUST_MEAN_ABSOLUTE_DEVIATION = rmad_;
	val_ENTROPY = entropy_;
	val_MODE = mode_;
	val_UNIFORMITY = uniformity_;

	// Median absolute deviation
	double medad = 0.0;
	for (auto& px : r.raw_pixels_3D)
		medad += std::abs(px.inten - median_);
	val_MEDIAN_ABSOLUTE_DEVIATION = medad / n;

	// --Uniformity calculated as PIU, percent image uniformity - see "A comparison of five standard methods for evaluating image intensity uniformity in partially parallel imaging MRI" [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/] and https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.2241606
	double piu = (1.0 - double(r.aux_max - r.aux_min) / double(r.aux_max + r.aux_min)) * 100.0;
	val_UNIFORMITY_PIU = piu;

	// Skewness
	Moments4 mom;
	for (auto& px : r.raw_pixels_3D)
		mom.add(px.inten);
	val_SKEWNESS = mom.skewness();

	// Pearson's Kurtosis
	val_KURTOSIS = mom.kurtosis();

	// Excess kurtosis
	val_EXCESS_KURTOSIS = mom.excess_kurtosis();

	double sumPow5 = 0, sumPow6 = 0;
	for (auto& px : r.raw_pixels_3D)
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

void D3_PixelIntensityFeatures::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void D3_PixelIntensityFeatures::osized_calculate(LR& r, ImageLoader& imloader)
{
	// --MIN, MAX
	val_MIN = r.aux_min;
	val_MAX = r.aux_max;
	val_RANGE = val_MAX - val_MIN;

	// --COVERED_IMAGE_INTENSITY_RANGE
	if (r.slide_idx >= 0)
	{
		const SlideProps& p = LR::dataset_props[r.slide_idx];
		double sir = (p.max_preroi_inten - p.min_preroi_inten); // slide intensity range
		val_COVERED_IMAGE_INTENSITY_RANGE = (r.aux_max - r.aux_min) / sir;
	}
	else
		val_COVERED_IMAGE_INTENSITY_RANGE = 1;

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

	// --MAD, VARIANCE, STDDEV, COV
	double mad = 0.0,
		var = 0.0;
	for (auto& px : r.raw_pixels_3D)
	{
		double diff = px.inten - mean_;
		mad += std::abs(diff);
		var += diff * diff;
	}
	val_MEAN_ABSOLUTE_DEVIATION = mad / n;
	val_VARIANCE = n > 1 ? var / (n - 1) : 0.0;
	val_VARIANCE_BIASED = n > 1 ? var / n : 0.0;
	val_STANDARD_DEVIATION = sqrt(val_VARIANCE);
	val_STANDARD_DEVIATION_BIASED = sqrt(val_VARIANCE_BIASED);
	val_COV = val_STANDARD_DEVIATION / mean_;

	// --Standard error
	val_STANDARD_ERROR = val_STANDARD_DEVIATION / sqrt(n);

	//==== Do not calculate features of all-blank intensities (to avoid NANs)
	if (r.aux_min == 0 && r.aux_max == 0)
		return;

	// P10, 25, 75, 90, IQR, QCOD, RMAD, entropy, uniformity
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
	val_QCOD = (p75_ - p25_) / (p75_ + p25_);
	val_INTERQUARTILE_RANGE = iqr_;
	val_ROBUST_MEAN_ABSOLUTE_DEVIATION = rmad_;
	val_ENTROPY = entropy_;
	val_MODE = mode_;
	val_UNIFORMITY = uniformity_;

	// Median absolute deviation
	double medad = 0.0;
	for (auto& px : r.raw_pixels_3D)
		medad += std::abs(px.inten - median_);
	val_MEDIAN_ABSOLUTE_DEVIATION = medad / n;

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

	// Excess kurtosis
	val_EXCESS_KURTOSIS = val_KURTOSIS - 3;

	// Hyperskewness hs = E[x-mean].^5 / std(x).^5
	val_HYPERSKEWNESS = mom.hyperskewness();

	// Hyperflatness hf = E[x-mean].^6 / std(x).^6
	val_HYPERFLATNESS = mom.hyperflatness();
}

void D3_PixelIntensityFeatures::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature3D::INTEGRATED_INTENSITY][0] = val_INTEGRATED_INTENSITY;
	fvals[(int)Feature3D::MEAN][0] = val_MEAN;
	fvals[(int)Feature3D::MEDIAN][0] = val_MEDIAN;
	fvals[(int)Feature3D::MIN][0] = val_MIN;
	fvals[(int)Feature3D::MAX][0] = val_MAX;
	fvals[(int)Feature3D::RANGE][0] = val_RANGE;
	fvals[(int)Feature3D::COVERED_IMAGE_INTENSITY_RANGE][0] = val_COVERED_IMAGE_INTENSITY_RANGE;
	fvals[(int)Feature3D::STANDARD_DEVIATION][0] = val_STANDARD_DEVIATION;
	fvals[(int)Feature3D::STANDARD_ERROR][0] = val_STANDARD_ERROR;
	fvals[(int)Feature3D::SKEWNESS][0] = val_SKEWNESS;
	fvals[(int)Feature3D::KURTOSIS][0] = val_KURTOSIS;
	fvals[(int)Feature3D::EXCESS_KURTOSIS][0] = val_EXCESS_KURTOSIS;
	fvals[(int)Feature3D::HYPERSKEWNESS][0] = val_HYPERSKEWNESS;
	fvals[(int)Feature3D::HYPERFLATNESS][0] = val_HYPERFLATNESS;
	fvals[(int)Feature3D::MEAN_ABSOLUTE_DEVIATION][0] = val_MEAN_ABSOLUTE_DEVIATION;
	fvals[(int)Feature3D::MEDIAN_ABSOLUTE_DEVIATION][0] = val_MEDIAN_ABSOLUTE_DEVIATION;
	fvals[(int)Feature3D::ENERGY][0] = val_ENERGY;
	fvals[(int)Feature3D::ROOT_MEAN_SQUARED][0] = val_ROOT_MEAN_SQUARED;
	fvals[(int)Feature3D::ENTROPY][0] = val_ENTROPY;
	fvals[(int)Feature3D::MODE][0] = val_MODE;
	fvals[(int)Feature3D::UNIFORMITY][0] = val_UNIFORMITY;
	fvals[(int)Feature3D::UNIFORMITY_PIU][0] = val_UNIFORMITY_PIU;
	fvals[(int)Feature3D::P01][0] = val_P01;
	fvals[(int)Feature3D::P10][0] = val_P10;
	fvals[(int)Feature3D::P25][0] = val_P25;
	fvals[(int)Feature3D::P75][0] = val_P75;
	fvals[(int)Feature3D::P90][0] = val_P90;
	fvals[(int)Feature3D::P99][0] = val_P99;
	fvals[(int)Feature3D::QCOD][0] = val_QCOD;
	fvals[(int)Feature3D::INTERQUARTILE_RANGE][0] = val_INTERQUARTILE_RANGE;
	fvals[(int)Feature3D::QCOD][0] = val_QCOD;
	fvals[(int)Feature3D::ROBUST_MEAN][0] = val_ROBUST_MEAN;
	fvals[(int)Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = val_ROBUST_MEAN_ABSOLUTE_DEVIATION;
	fvals[(int)Feature3D::COV][0] = val_COV;
	fvals[(int)Feature3D::STANDARD_DEVIATION_BIASED][0] = val_STANDARD_DEVIATION_BIASED;
	fvals[(int)Feature3D::VARIANCE][0] = val_VARIANCE;
	fvals[(int)Feature3D::VARIANCE_BIASED][0] = val_VARIANCE_BIASED;
}

void D3_PixelIntensityFeatures::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(D3_PixelIntensityFeatures::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void D3_PixelIntensityFeatures::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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
		D3_PixelIntensityFeatures f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void D3_PixelIntensityFeatures::reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_PixelIntensityFeatures f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void D3_PixelIntensityFeatures::cleanup_instance()
{
	val_INTEGRATED_INTENSITY = 0;
	val_MEAN = 0;
	val_MEDIAN = 0;
	val_MIN = 0;
	val_MAX = 0;
	val_RANGE = 0;
	val_COVERED_IMAGE_INTENSITY_RANGE = 0;
	val_STANDARD_DEVIATION = 0;
	val_STANDARD_ERROR = 0;
	val_SKEWNESS = 0;
	val_KURTOSIS = 0;
	val_EXCESS_KURTOSIS = 0;
	val_HYPERSKEWNESS = 0;
	val_HYPERFLATNESS = 0;
	val_MEAN_ABSOLUTE_DEVIATION = 0;
	val_MEDIAN_ABSOLUTE_DEVIATION = 0;
	val_ENERGY = 0;
	val_ROOT_MEAN_SQUARED = 0;
	val_ENTROPY = 0;
	val_MODE = 0;
	val_UNIFORMITY = 0;
	val_UNIFORMITY_PIU = 0;
	val_P01 = 0;
	val_P10 = 0;
	val_P25 = 0;
	val_P75 = 0;
	val_P90 = 0;
	val_P99 = 0;
	val_QCOD = 0;
	val_INTERQUARTILE_RANGE = 0;
	val_ROBUST_MEAN = 0;
	val_ROBUST_MEAN_ABSOLUTE_DEVIATION = 0;
	val_COV = 0;
	val_STANDARD_DEVIATION_BIASED = 0;
	val_VARIANCE = 0;
	val_VARIANCE_BIASED = 0;
}

