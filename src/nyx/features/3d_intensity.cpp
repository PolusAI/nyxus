#include <cassert>
#include "../constants.h"
#include "../environment.h"
#include "histogram.h"
#include "3d_intensity.h"
#include "pixel.h"

using namespace Nyxus;

bool D3_VoxelIntensityFeatures::required (const FeatureSet & fs)
{
	return fs.anyEnabled ({D3_VoxelIntensityFeatures::featureset});
}

D3_VoxelIntensityFeatures::D3_VoxelIntensityFeatures() : FeatureMethod("PixelIntensityFeatures_3D")
{
	provide_features ({D3_VoxelIntensityFeatures::featureset});
}

bool matlab_grey_binning (int greybinning_info) { return greybinning_info > 0; }
bool radiomics_grey_binning (int greybinning_info) { return greybinning_info < 0; }
// returns 1-based bin indices
PixIntens to_grayscale_radiomix(PixIntens x, PixIntens min__, PixIntens max__, int binCount)
{
	if (x)
	{
		double binW = double(max__ - min__) / double(binCount);
		PixIntens y = (PixIntens)(double(x - min__) / binW + 1);
		if (y > binCount)
			y = binCount;	// the last bin is +1 unit wider
		return y;
	}
	else
		return 0;
}

void bin_intensities_3d (std::vector <Pixel3> &S, const std::vector <Pixel3> &I, PixIntens min_I_inten, PixIntens max_I_inten, int greybin_info)
{
	// radiomics binning
	auto n = I.size();
	for (size_t i = 0; i < n; i++)
		S[i].inten = to_grayscale_radiomix (I[i].inten, min_I_inten, max_I_inten, std::abs(greybin_info));
}

void D3_VoxelIntensityFeatures::calculate (LR &r, const Fsettings& s, const Dataset &ds)
{
	// bin intensities
	std::vector <Pixel3> &B = r.raw_pixels_3D;
	PixIntens binned_min = r.aux_min, 
		binned_max = r.aux_max;

	// --MIN, MAX
	val_MIN = r.aux_min;
	val_MAX = r.aux_max;
	val_RANGE = val_MAX - val_MIN;

	// --COVERED_IMAGE_INTENSITY_RANGE
	if (r.slide_idx >= 0)
	{
		const SlideProps& p = ds.dataset_props [r.slide_idx];
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
	for (auto &px : B)
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
	for (auto &px : B)
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
	int n_radiomicGreyBins = STNGS_MISSING(s) ? DEFAULT_NUM_HISTO_BINS : STNGS_NGREYS(s);
	TrivialHistogram H;
	H.initialize (n_radiomicGreyBins, binned_min, binned_max, B);
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
	for (auto& px : B)
		medad += std::abs(px.inten - median_);
	val_MEDIAN_ABSOLUTE_DEVIATION = medad / n;

	// FIX 3ROBUST_MEAN: was never assigned in this (trivial) path so it defaulted to 0, unlike the
	// 2D PixelIntensityFeatures which computes it (intensity.cpp) -> port the same definition:
	// mean of voxels within the [P10,P90] robust window (p10_/p90_ from H.get_stats() above).
	double robustMean = 0.0;
	size_t robustCount = 0;
	for (auto& px : B)
		if (px.inten >= p10_ && px.inten <= p90_)
		{
			robustMean += px.inten;
			robustCount++;
		}
	val_ROBUST_MEAN = robustCount ? robustMean / double(robustCount) : 0.0;

	// --Uniformity calculated as PIU, percent image uniformity - see "A comparison of five standard methods for evaluating image intensity uniformity in partially parallel imaging MRI" [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/] and https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.2241606
	double piu = (1.0 - double(r.aux_max - r.aux_min) / double(r.aux_max + r.aux_min)) * 100.0;
	val_UNIFORMITY_PIU = piu;

	// Skewness
	Moments4 mom;
	for (auto& px : B)
		mom.add(px.inten);
	val_SKEWNESS = mom.skewness();

	// Pearson's Kurtosis
	val_KURTOSIS = mom.kurtosis();

	// Excess kurtosis
	val_EXCESS_KURTOSIS = mom.excess_kurtosis();

	double sumPow5 = 0, sumPow6 = 0;
	for (auto& px : B)
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

void D3_VoxelIntensityFeatures::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void D3_VoxelIntensityFeatures::osized_calculate (LR & r, const Fsettings & s, const Dataset & ds, ImageLoader & ldr)
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
	const size_t nvox = r.raw_voxels_NT.size();

	// Stream voxels off disk (raw_voxels_NT) rather than holding the cube (raw_pixels_3D) or
	// reading the 2D z-less cloud (raw_pixels_NT). Each disk pass decodes the whole cloud, so the
	// per-voxel scalars are gathered in as few passes as their data dependencies allow (mean must
	// be known before dispersion; the percentiles before median-abs-dev / robust mean).

	// --- Pass 1: raw sums + online moments (MEAN, ENERGY, CENTROID, INTEGRATED_INTENSITY, and the
	// Moments4 accumulator for SKEWNESS/KURTOSIS -- folded in here instead of its own later pass)
	double mean_ = 0.0;
	double energy = 0.0;
	double cen_x = 0.0,
		cen_y = 0.0,
		cen_z = 0.0,
		integInten = 0.0;
	Moments4 mom;
	for (size_t i = 0; i < nvox; i++)
	{
		Pixel3 px = r.raw_voxels_NT[i];
		mean_ += px.inten;
		energy += px.inten * px.inten;
		cen_x += px.x;
		cen_y += px.y;
		cen_z += px.z;
		integInten += px.inten;
		mom.add(px.inten);
	}
	mean_ /= n;
	val_MEAN = mean_;
	val_ENERGY = energy;
	val_ROOT_MEAN_SQUARED = sqrt(val_ENERGY / n);
	val_INTEGRATED_INTENSITY = integInten;

	// --- Pass 2: dispersion (MAD, VARIANCE, STDDEV, COV) -- needs mean_
	double mad = 0.0,
		var = 0.0;
	for (size_t i = 0; i < nvox; i++)
	{
		double diff = r.raw_voxels_NT[i].inten - mean_;
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

	//==== Do not calculate features of all-blank intensities (to avoid NANs). Kept before the
	// Moments4-derived and percentile-derived features so a blank ROI leaves them at their
	// defaults exactly as before (the accumulator is filled in pass 1 but not consumed until here).
	if (r.aux_min == 0 && r.aux_max == 0)
		return;

	// Skewness / Kurtosis from the pass-1 accumulator
	val_SKEWNESS = mom.skewness();
	val_KURTOSIS = mom.kurtosis();
	// Excess kurtosis. NOTE: this is NOT val_KURTOSIS-3 -- Moments4::excess_kurtosis() has its own
	// independent zero-variance (M2==0) guard returning exactly 0.0, whereas kurtosis() ALSO
	// returns exactly 0.0 under that same guard; subtracting 3 from the latter wrongly yields -3.0
	// for a degenerate (constant-intensity) ROI instead of 0.0. Only diverges from kurtosis()-3 in
	// that degenerate case (for n>4 the two formulas are algebraically identical), which is why
	// every non-degenerate fixture matched before this was caught by a blank-ROI OOC test.
	val_EXCESS_KURTOSIS = mom.excess_kurtosis();

	// P10, 25, 75, 90, IQR, QCOD, RMAD, entropy, uniformity
	int n_greybins = STNGS_NGREYS(s);
	TrivialHistogram H;
	H.initialize (n_greybins, r.aux_min, r.aux_max, r.raw_voxels_NT);
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

	// --Uniformity calculated as PIU, percent image uniformity - see "A comparison of five standard methods for evaluating image
	//	intensity uniformity in partially parallel imaging MRI" [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/]
	//	and https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.2241606
	double piu = (1.0 - double(r.aux_max - r.aux_min) / double(r.aux_max + r.aux_min)) * 100.0;
	val_UNIFORMITY_PIU = piu;

	// --- Pass 3: the percentile-dependent scalars in one sweep -- median absolute deviation,
	// robust mean (FIX 3ROBUST_MEAN: the OOC path used to default this to 0; mean of voxels in the
	// [P10,P90] window, matching the 2D implementation), and the hyperskewness/hyperflatness
	// sum-of-powers (the in-core calculate()'s explicit definition; mom.hyperskewness()/
	// hyperflatness() use a different definition and diverge). Each accumulator sums in the same
	// voxel order as when these were three separate passes, so the values are unchanged.
	double medad = 0.0;
	double robustMean = 0.0;
	size_t robustCount = 0;
	double sumPow5 = 0, sumPow6 = 0;
	for (size_t i = 0; i < nvox; i++)
	{
		double inten = double(r.raw_voxels_NT[i].inten);
		medad += std::abs(inten - median_);
		if (inten >= p10_ && inten <= p90_)
		{
			robustMean += inten;
			robustCount++;
		}
		double diff = inten - mean_;
		sumPow5 += std::pow(diff, 5.);
		sumPow6 += std::pow(diff, 6.);
	}
	val_MEDIAN_ABSOLUTE_DEVIATION = medad / n;
	val_ROBUST_MEAN = robustCount ? robustMean / double(robustCount) : 0.0;
	double denomHS = (n * std::pow(val_STANDARD_DEVIATION, 5.));
	val_HYPERSKEWNESS = denomHS == 0. ? 0. : sumPow5 / denomHS;
	double denomHF = (n * std::pow(val_STANDARD_DEVIATION, 6.));
	val_HYPERFLATNESS = denomHF == 0. ? 0. : sumPow6 / denomHF;
}

void D3_VoxelIntensityFeatures::save_value(std::vector<std::vector<double>>& fvals)
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

void D3_VoxelIntensityFeatures::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_VoxelIntensityFeatures f;
		f.calculate (r, s, ds);
		f.save_value (r.fvals);
	}
}

/*static*/ void D3_VoxelIntensityFeatures::extract (LR& r, const Fsettings& s, const Dataset& ds)
{
	D3_VoxelIntensityFeatures f;
	f.calculate (r, s, ds);		// FIX: was the 2-arg calculate, which is an "illegal call" stub
	f.save_value (r.fvals);
}

void D3_VoxelIntensityFeatures::cleanup_instance()
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

