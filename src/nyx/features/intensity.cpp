#include "../constants.h"
#include "../environment.h"
#include "histogram.h"
#include "intensity.h"
#include "pixel.h"

using namespace Nyxus;

bool PixelIntensityFeatures::required(const FeatureSet& fs)
{
	return fs.anyEnabled(
		{
			Feature2D::COV,
			Feature2D::COVERED_IMAGE_INTENSITY_RANGE,
			Feature2D::ENERGY,
			Feature2D::ENTROPY,
			Feature2D::EXCESS_KURTOSIS,
			Feature2D::HYPERFLATNESS,
			Feature2D::HYPERSKEWNESS,
			Feature2D::INTEGRATED_INTENSITY,
			Feature2D::INTERQUARTILE_RANGE,
			Feature2D::KURTOSIS,
			Feature2D::MAX,
			Feature2D::MEAN,
			Feature2D::MEAN_ABSOLUTE_DEVIATION,
			Feature2D::MEDIAN,
			Feature2D::MEDIAN_ABSOLUTE_DEVIATION,
			Feature2D::MIN,
			Feature2D::MODE,
			Feature2D::P01,
			Feature2D::P10, Feature2D::P25, Feature2D::P75, Feature2D::P90, Feature2D::P99,
			Feature2D::QCOD,
			Feature2D::RANGE,
			Feature2D::ROBUST_MEAN,
			Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION,
			Feature2D::ROOT_MEAN_SQUARED,
			Feature2D::SKEWNESS,
			Feature2D::STANDARD_DEVIATION,
			Feature2D::STANDARD_DEVIATION_BIASED,
			Feature2D::STANDARD_ERROR,
			Feature2D::VARIANCE,
			Feature2D::VARIANCE_BIASED,
			Feature2D::UNIFORMITY,
			Feature2D::UNIFORMITY_PIU
		});
}

PixelIntensityFeatures::PixelIntensityFeatures() : FeatureMethod("PixelIntensityFeatures")
{
	provide_features (PixelIntensityFeatures::featureset);
}

void PixelIntensityFeatures::calculate (LR& r, const Fsettings & fsett, const Dataset & ds)
{
	// intercept blank ROIs
	if (r.aux_max == r.aux_min)
	{
		val_MEAN =
			val_MEDIAN =
			val_MIN =
			val_MAX = r.aux_min;
		val_RANGE = 0;

		val_INTEGRATED_INTENSITY =
			val_COVERED_IMAGE_INTENSITY_RANGE =
			val_STANDARD_DEVIATION =
			val_STANDARD_ERROR =
			val_SKEWNESS =
			val_KURTOSIS =
			val_EXCESS_KURTOSIS =
			val_HYPERSKEWNESS =
			val_HYPERFLATNESS =
			val_MEAN_ABSOLUTE_DEVIATION =
			val_MEDIAN_ABSOLUTE_DEVIATION =
			val_ENERGY =
			val_ROOT_MEAN_SQUARED =
			val_ENTROPY =
			val_MODE =
			val_UNIFORMITY =
			val_UNIFORMITY_PIU =
			val_P01 = val_P10 = val_P25 = val_P75 = val_P90 = val_P99 =
			val_QCOD =
			val_INTERQUARTILE_RANGE =
			val_ROBUST_MEAN =
			val_ROBUST_MEAN_ABSOLUTE_DEVIATION =
			val_COV =
			val_STANDARD_DEVIATION_BIASED =
			val_VARIANCE =
			val_VARIANCE_BIASED = fsett[(int)NyxSetting::SOFTNAN].rval; // former theEnvironment.resultOptions.noval();

		return;
	}

	// --MIN, MAX
	val_MIN = r.aux_min;
	val_MAX = r.aux_max;
	val_RANGE = val_MAX - val_MIN;

	// --COVERED_IMAGE_INTENSITY_RANGE
	if (r.slide_idx >= 0)	// slide may not always be available in varoius scenarios (segmented, wsi, in-memory, etc.)
	{
		const SlideProps& p = ds.dataset_props [r.slide_idx];
		double sir = (p.max_preroi_inten - p.min_preroi_inten); // slide intensity range
		val_COVERED_IMAGE_INTENSITY_RANGE = double(r.aux_max - r.aux_min) / sir;
	}

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

	// --MAD, VARIANCE, STDDEV, COV
	double mad = 0.0,
		var = 0.0;
	for (auto& px : r.raw_pixels)
	{
		double diff = px.inten - mean_;
		mad += std::abs(diff);
		var += diff * diff;
	}
	val_MEAN_ABSOLUTE_DEVIATION = mad / n;
	val_VARIANCE = n>1 ? var/(n-1) : 0.0;
	val_VARIANCE_BIASED = n>1 ? var/n : 0.0;
	val_STANDARD_DEVIATION = sqrt(val_VARIANCE);
	val_STANDARD_DEVIATION_BIASED = sqrt(val_VARIANCE_BIASED);
	val_COV = val_STANDARD_DEVIATION / mean_;

	// --Standard error
	val_STANDARD_ERROR = val_STANDARD_DEVIATION / sqrt(n);

	//==== Do not calculate features of all-blank intensities (to avoid NANs)
	if (r.aux_min == 0 && r.aux_max == 0)
		return;

	// P10, 25, 75, 90, IQR, QCOD, RMAD, entropy, uniformity
	int n_radiomicsGreyBins = STNGS_MISSING(fsett) ? DEFAULT_NUM_HISTO_BINS : STNGS_NGREYS(fsett);
	TrivialHistogram H;
	H.initialize (n_radiomicsGreyBins, r.aux_min, r.aux_max, r.raw_pixels);
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
	for (auto& px : r.raw_pixels)
		medad += std::abs(px.inten - median_);
	val_MEDIAN_ABSOLUTE_DEVIATION = medad / n;

	// --Uniformity calculated as PIU, percent image uniformity - see "A comparison of five standard methods for evaluating image intensity uniformity in partially parallel imaging MRI" [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/] and https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.2241606
	double piu = (1.0 - double(r.aux_max - r.aux_min) / double(r.aux_max + r.aux_min)) * 100.0;
	val_UNIFORMITY_PIU = piu;

	// Skewness
	Moments4 mom;
	for (auto& px : r.raw_pixels)
		mom.add(px.inten);
	val_SKEWNESS = mom.skewness();

	// Pearson's Kurtosis
	val_KURTOSIS = mom.kurtosis();

	// Excess kurtosis
	val_EXCESS_KURTOSIS = mom.excess_kurtosis();

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

void PixelIntensityFeatures::osized_calculate (LR& r, const Fsettings& stng, const Dataset& ds, ImageLoader& imloader)
{
	// --MIN, MAX
	val_MIN = r.aux_min;
	val_MAX = r.aux_max;
	val_RANGE = val_MAX - val_MIN;

	// --COVERED_IMAGE_INTENSITY_RANGE
	if (r.slide_idx >= 0)
	{
		const SlideProps& p = ds.dataset_props [r.slide_idx];
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
	for (auto& px : r.raw_pixels)
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
	int n_greybins = STNGS_NGREYS(stng);
	TrivialHistogram H;
	H.initialize (n_greybins, r.aux_min, r.aux_max, r.raw_pixels_NT);
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
	for (auto& px : r.raw_pixels)
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

void PixelIntensityFeatures::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::INTEGRATED_INTENSITY][0] = val_INTEGRATED_INTENSITY;
	fvals[(int)Feature2D::MEAN][0] = val_MEAN;
	fvals[(int)Feature2D::MEDIAN][0] = val_MEDIAN;
	fvals[(int)Feature2D::MIN][0] = val_MIN;
	fvals[(int)Feature2D::MAX][0] = val_MAX;
	fvals[(int)Feature2D::RANGE][0] = val_RANGE;
	fvals[(int)Feature2D::COVERED_IMAGE_INTENSITY_RANGE][0] = val_COVERED_IMAGE_INTENSITY_RANGE;
	fvals[(int)Feature2D::STANDARD_DEVIATION][0] = val_STANDARD_DEVIATION;
	fvals[(int)Feature2D::STANDARD_ERROR][0] = val_STANDARD_ERROR;
	fvals[(int)Feature2D::SKEWNESS][0] = val_SKEWNESS;
	fvals[(int)Feature2D::KURTOSIS][0] = val_KURTOSIS;
	fvals[(int)Feature2D::EXCESS_KURTOSIS][0] = val_EXCESS_KURTOSIS;
	fvals[(int)Feature2D::HYPERSKEWNESS][0] = val_HYPERSKEWNESS;
	fvals[(int)Feature2D::HYPERFLATNESS][0] = val_HYPERFLATNESS;
	fvals[(int)Feature2D::MEAN_ABSOLUTE_DEVIATION][0] = val_MEAN_ABSOLUTE_DEVIATION;
	fvals[(int)Feature2D::MEDIAN_ABSOLUTE_DEVIATION][0] = val_MEDIAN_ABSOLUTE_DEVIATION;
	fvals[(int)Feature2D::ENERGY][0] = val_ENERGY;
	fvals[(int)Feature2D::ROOT_MEAN_SQUARED][0] = val_ROOT_MEAN_SQUARED;
	fvals[(int)Feature2D::ENTROPY][0] = val_ENTROPY;
	fvals[(int)Feature2D::MODE][0] = val_MODE;
	fvals[(int)Feature2D::UNIFORMITY][0] = val_UNIFORMITY;
	fvals[(int)Feature2D::UNIFORMITY_PIU][0] = val_UNIFORMITY_PIU;
	fvals[(int)Feature2D::P01][0] = val_P01;
	fvals[(int)Feature2D::P10][0] = val_P10;
	fvals[(int)Feature2D::P25][0] = val_P25;
	fvals[(int)Feature2D::P75][0] = val_P75;
	fvals[(int)Feature2D::P90][0] = val_P90;
	fvals[(int)Feature2D::P99][0] = val_P99;
	fvals[(int)Feature2D::QCOD][0] = val_QCOD;
	fvals[(int)Feature2D::INTERQUARTILE_RANGE][0] = val_INTERQUARTILE_RANGE;
	fvals[(int)Feature2D::QCOD][0] = val_QCOD;
	fvals[(int)Feature2D::ROBUST_MEAN][0] = val_ROBUST_MEAN;
	fvals[(int)Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0] = val_ROBUST_MEAN_ABSOLUTE_DEVIATION;
	fvals[(int)Feature2D::COV][0] = val_COV;
	fvals[(int)Feature2D::STANDARD_DEVIATION_BIASED][0] = val_STANDARD_DEVIATION_BIASED;
	fvals[(int)Feature2D::VARIANCE][0] = val_VARIANCE;
	fvals[(int)Feature2D::VARIANCE_BIASED][0] = val_VARIANCE_BIASED;
}

void PixelIntensityFeatures::extract (LR& r, const Fsettings & fs, const Dataset & ds)
{		
	PixelIntensityFeatures f;
	f.calculate (r, fs, ds);
	f.save_value (r.fvals);
}

void PixelIntensityFeatures::reduce (
	size_t start, 
	size_t end, 
	std::vector<int>* roi_labels, 
	std::unordered_map <int, LR>* roi_data, 
	const Fsettings & fsett,
	const Dataset & ds)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*roi_labels)[i];
		LR& r = (*roi_data)[lab];
		extract (r, fsett, ds);
	}
}

void PixelIntensityFeatures::cleanup_instance()
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

