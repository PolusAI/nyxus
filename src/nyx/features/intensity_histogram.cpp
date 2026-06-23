#include <cmath>
#include <limits>
#include "../constants.h"
#include "../environment.h"
#include "../slideprops.h"
#include "intensity_histogram.h"
#include "pixel.h"

using namespace Nyxus;

bool IntensityHistogramFeatures::required (const FeatureSet& fs)
{
	return fs.anyEnabled (IntensityHistogramFeatures::featureset);
}

IntensityHistogramFeatures::IntensityHistogramFeatures() : FeatureMethod("IntensityHistogramFeatures")
{
	provide_features (IntensityHistogramFeatures::featureset);
}

void IntensityHistogramFeatures::fill_invalid (double nan_val)
{
	result_.clear();
	result_.reserve (featureset.size());
	for (auto fc : featureset)
		result_.push_back ({ (int)fc, nan_val });
}

// IBSI Intensity Histogram core: all 46 features are derived from a single N-bin
// histogram over [minVal,maxVal].
template <class Pixelcloud>
void IntensityHistogramFeatures::compute (const Pixelcloud& pixels, double minVal, double maxVal, size_t count, int nbins,
                                          double nan_val, double pscale, double poffset)
{
	const int N = nbins;
	const double binWidth = (maxVal - minVal) / double(N);

	// --- Build the N-bin frequency histogram (bin index = floor((v-min)/binW)
	//     clamped to [0,N-1]; v==max folds into the last bin).
	std::vector<double> freq ((size_t)N, 0.0);
	const size_t M = pixels.size();
	for (size_t j = 0; j < M; j++)
	{
		double v = poffset + pscale * (double) pixels[j].inten;
		int idx = (int) std::floor ((v - minVal) / binWidth);
		if (idx < 0) idx = 0;
		if (idx >= N) idx = N - 1;
		freq[(size_t)idx] += 1.0;
	}

	const double totalFrequency = (double) count;

	// bin geometry
	auto binMin    = [&](int i) { return minVal + double(i) * binWidth; };
	auto binMax    = [&](int i) { return minVal + double(i + 1) * binWidth; };
	auto binCenter = [&](int i) { return minVal + (double(i) + 0.5) * binWidth; };

	// Bin index (0-based) of an arbitrary value
	auto getIndexOf = [&](double value) -> int
	{
		int idx = (int) std::floor ((value - minVal) / binWidth);
		if (idx < 0) idx = 0;
		if (idx >= N) idx = N - 1;
		return idx;
	};

	// Histogram quantile (1D), interpolated within the bin
	auto quantile = [&](double p) -> double
	{
		double cumulated = 0.0, p_n_prev = 0.0, p_n, f_n = 0.0, binProportion, mn, mx, interval;
		if (p < 0.5)
		{
			int n = 0; p_n = 0.0;
			do
			{
				f_n = freq[(size_t)n];
				cumulated += f_n;
				p_n_prev = p_n;
				p_n = cumulated / totalFrequency;
				++n;
			} while (n < N && p_n < p);
			binProportion = f_n / totalFrequency;
			mn = binMin (n - 1); mx = binMax (n - 1); interval = mx - mn;
			return mn + ((p - p_n_prev) / binProportion) * interval;
		}
		else
		{
			int n = N - 1, m = 0; p_n = 1.0;
			do
			{
				f_n = freq[(size_t)n];
				cumulated += f_n;
				p_n_prev = p_n;
				p_n = 1.0 - cumulated / totalFrequency;
				--n; ++m;
			} while (m < N && p_n > p);
			binProportion = f_n / totalFrequency;
			mn = binMin (n + 1); mx = binMax (n + 1); interval = mx - mn;
			return mx - ((p_n_prev - p) / binProportion) * interval;
		}
	};

	// Median -> center of the bin where the running count first exceeds count/2
	// (bin-center median, NOT an interpolated quantile)
	auto medianFromBins = [&]() -> double
	{
		double total = 0.0;
		double half = double (count / 2);	// integer division
		int bin = 0;
		while (total <= half && bin < N)
		{
			total += freq[(size_t)bin];
			++bin;
		}
		--bin;
		return binCenter (bin);
	};

	const double Log2 = std::log (2.0);

	// Values + their bin indices
	double medianValue  = medianFromBins();         int medianIndex  = getIndexOf (medianValue);
	double minimumValue = minVal;                   int minimumIndex = getIndexOf (minimumValue);
	double p10Value     = quantile (0.10);          int p10Index     = getIndexOf (p10Value);
	double p25Value     = quantile (0.25);          int p25Index     = getIndexOf (p25Value);
	double p75Value     = quantile (0.75);          int p75Index     = getIndexOf (p75Value);
	double p90Value     = quantile (0.90);          int p90Index     = getIndexOf (p90Value);
	double maximumValue = maxVal;                   int maximumIndex = getIndexOf (maximumValue);

	// First pass: mean + robust mean (over [p10Index, p90Index])
	double meanValue = 0.0, meanIndex = 0.0;
	double robustMeanValue = 0.0, robustMeanIndex = 0.0, robustCount = 0.0;
	for (int i = 0; i < N; i++)
	{
		double f = freq[(size_t)i];
		double probability = f / totalFrequency;
		double voxelValue = binCenter (i);

		meanValue += probability * voxelValue;
		meanIndex += probability * double(i);

		if (i >= p10Index && i <= p90Index)
		{
			robustMeanValue += f * voxelValue;
			robustMeanIndex += f * double(i);
			robustCount += f;
		}
	}
	robustMeanValue /= robustCount;
	robustMeanIndex /= robustCount;

	// Accumulators for the second pass
	double varianceValue = 0, varianceIndex = 0,
		skewnessValue = 0, skewnessIndex = 0,
		kurtosisValue = 0, kurtosisIndex = 0,
		modeValue = 0, modeIndex = 0, modeFrequence = 0,
		meanAbsoluteDeviationValue = 0, meanAbsoluteDeviationIndex = 0,
		robustMeanAbsoluteDeviationValue = 0, robustMeanAbsoluteDeviationIndex = 0,
		medianAbsoluteDeviationValue = 0, medianAbsoluteDeviationIndex = 0,
		entropyValue = 0, entropyIndex = 0,
		uniformityValue = 0, uniformityIndex = 0;

	// Gradient extremes seed from numeric_limits<double>::min()/max()
	double maximumGradientValue = std::numeric_limits<double>::min();
	double maximumGradientIndex = 0;
	double minimumGradientValue = std::numeric_limits<double>::max();
	double minimumGradientIndex = 0;
	double gradient = 0;

	for (int i = 0; i < N; i++)
	{
		double f = freq[(size_t)i];
		double probability = f / totalFrequency;
		double voxelValue = binCenter (i);

		double deltaValue = voxelValue - meanValue;
		double deltaIndex = double(i) - meanIndex;

		varianceValue += probability * deltaValue * deltaValue;
		varianceIndex += probability * deltaIndex * deltaIndex;
		skewnessValue += probability * deltaValue * deltaValue * deltaValue;
		skewnessIndex += probability * deltaIndex * deltaIndex * deltaIndex;
		kurtosisValue += probability * deltaValue * deltaValue * deltaValue * deltaValue;
		kurtosisIndex += probability * deltaIndex * deltaIndex * deltaIndex * deltaIndex;

		if (modeFrequence < f)
		{
			modeFrequence = f;
			modeValue = voxelValue;
			modeIndex = double(i);
		}
		meanAbsoluteDeviationValue += probability * std::abs (deltaValue);
		meanAbsoluteDeviationIndex += probability * std::abs (deltaIndex);
		if (i >= p10Index && i <= p90Index)
		{
			robustMeanAbsoluteDeviationValue += f * std::abs (voxelValue - robustMeanValue);
			robustMeanAbsoluteDeviationIndex += f * std::abs (double(i) - robustMeanIndex);
		}
		medianAbsoluteDeviationValue += probability * std::abs (voxelValue - medianValue);
		medianAbsoluteDeviationIndex += probability * std::abs (double(i) - double(medianIndex));
		if (probability > 0.0000001)
		{
			entropyValue -= probability * std::log (probability) / Log2;
			entropyIndex = entropyValue;
		}
		uniformityValue += probability * probability;
		uniformityIndex = uniformityValue;

		if (i == 0)
			gradient = freq[1] - freq[0];
		else if (i == N - 1)
			gradient = freq[(size_t)i] - freq[(size_t)(i - 1)];
		else
			gradient = (freq[(size_t)(i + 1)] - freq[(size_t)(i - 1)]) / 2.0;

		if (gradient > maximumGradientValue)
		{
			maximumGradientValue = gradient;
			maximumGradientIndex = i + 1;
		}
		if (gradient < minimumGradientValue)
		{
			minimumGradientValue = gradient;
			minimumGradientIndex = i + 1;
		}
	}

	skewnessValue = skewnessValue / (varianceValue * std::sqrt (varianceValue));
	skewnessIndex = skewnessIndex / (varianceIndex * std::sqrt (varianceIndex));
	kurtosisValue = kurtosisValue / (varianceValue * varianceValue) - 3;	// excess kurtosis
	kurtosisIndex = kurtosisIndex / (varianceIndex * varianceIndex) - 3;
	double coefficientOfVariationValue = std::sqrt (varianceValue) / meanValue;
	double coefficientOfVariationIndex = std::sqrt (varianceIndex) / (meanIndex + 1);
	double quantileCoefficientOfDispersionValue = (p75Value - p25Value) / (p75Value + p25Value);
	double quantileCoefficientOfDispersionIndex = (double(p75Index) - double(p25Index)) / (double(p75Index) + 1.0 + double(p25Index) + 1.0);
	robustMeanAbsoluteDeviationValue /= robustCount;
	robustMeanAbsoluteDeviationIndex /= robustCount;

	// Emit in the exact order of 'featureset'
	double out[] =
	{
		// Value family (20)
		meanValue,
		varianceValue,
		skewnessValue,
		kurtosisValue,
		medianValue,
		minimumValue,
		p10Value,
		p90Value,
		maximumValue,
		modeValue,
		p75Value - p25Value,
		maximumValue - minimumValue,
		meanAbsoluteDeviationValue,
		robustMeanAbsoluteDeviationValue,
		medianAbsoluteDeviationValue,
		coefficientOfVariationValue,
		quantileCoefficientOfDispersionValue,
		entropyValue,
		uniformityValue,
		robustMeanValue,
		// Index family (19) -- 1-based bin indices
		meanIndex + 1,
		varianceIndex,
		skewnessIndex,
		kurtosisIndex,
		double(medianIndex) + 1,
		double(minimumIndex) + 1,
		double(p10Index) + 1,
		double(p90Index) + 1,
		double(maximumIndex) + 1,
		modeIndex + 1,
		double(p75Index) - double(p25Index),
		double(maximumIndex) - double(minimumIndex),
		meanAbsoluteDeviationIndex,
		robustMeanAbsoluteDeviationIndex,
		medianAbsoluteDeviationIndex,
		coefficientOfVariationIndex,
		quantileCoefficientOfDispersionIndex,
		entropyIndex,
		uniformityIndex,
		// gradient + bookkeeping (7)
		maximumGradientValue,
		maximumGradientIndex,
		minimumGradientValue,
		minimumGradientIndex,
		robustMeanIndex,
		double(N),
		binWidth
	};

	result_.clear();
	result_.reserve (featureset.size());
	int k = 0;
	for (auto fc : featureset)
		result_.push_back ({ (int)fc, out[k++] });
}

void IntensityHistogramFeatures::calculate (LR& r, const Fsettings& s, const Dataset& ds)
{
	double nanv = STNGS_NAN(s);

	// IBSI gate (defensive; the family is primarily gated at enablement time)
	if (! STNGS_IBSI(s))
	{
		fill_invalid (nanv);
		return;
	}

	int nbins = STNGS_MISSING(s) ? DEFAULT_NUM_HISTO_BINS : STNGS_NGREYS(s);
	double mn = r.aux_min, mx = r.aux_max;
	size_t cnt = r.raw_pixels.size();

	// Degenerate ROI (single intensity, empty, or <2 bins): no meaningful histogram
	if (mx <= mn || nbins < 2 || cnt == 0)
	{
		fill_invalid (nanv);
		return;
	}

	double pscale, poffset;
	float_domain_map (r, s, ds, pscale, poffset);
	compute (r.raw_pixels, poffset + pscale * mn, poffset + pscale * mx, cnt, nbins, nanv, pscale, poffset);
}

// Float intensity domain: when the slide is a float image, recover the linear map
// that undoes Nyxus's load-time float->uint quantization so the IH histogram is
// computed in the ORIGINAL float intensity domain. No-op (scale=1, offset=0) for
// integer images or when slide props are unavailable.
void IntensityHistogramFeatures::float_domain_map (LR& r, const Fsettings& s, const Dataset& ds,
                                                   double& pscale, double& poffset)
{
	pscale = 1.0;
	poffset = 0.0;
	if (STNGS_MISSING(s) || r.slide_idx < 0)
		return;
	const SlideProps& sp = ds.dataset_props[r.slide_idx];
	if (! sp.fp_phys_pivoxels)
		return;	// integer image: stored intensities are already native, no rescale happened

	// Mirror image_loader.cpp exactly: with fp options active the loader quantized
	// float over [fpimg_min, fpimg_max]; otherwise over the per-slide pre-ROI [min,max].
	double DR = STNGS_FPIMG_DR(s);
	double fpmin, fpmax;
	if (STNGS_FPIMG_ACTIVE(s))
	{
		fpmin = STNGS_FPIMG_MIN(s);
		fpmax = STNGS_FPIMG_MAX(s);
	}
	else
	{
		fpmin = sp.min_preroi_inten;
		fpmax = sp.max_preroi_inten;
	}
	if (DR > 0.0 && fpmax > fpmin)
	{
		pscale = (fpmax - fpmin) / DR;	// float per uint step
		poffset = fpmin;
	}
}

void IntensityHistogramFeatures::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity)
{}

void IntensityHistogramFeatures::osized_calculate (LR& r, const Fsettings& s, const Dataset& ds, ImageLoader& ldr)
{
	double nanv = STNGS_NAN(s);

	if (! STNGS_IBSI(s))
	{
		fill_invalid (nanv);
		return;
	}

	int nbins = STNGS_MISSING(s) ? DEFAULT_NUM_HISTO_BINS : STNGS_NGREYS(s);
	double mn = r.aux_min, mx = r.aux_max;
	size_t cnt = r.raw_pixels_NT.size();

	if (mx <= mn || nbins < 2 || cnt == 0)
	{
		fill_invalid (nanv);
		return;
	}

	double pscale, poffset;
	float_domain_map (r, s, ds, pscale, poffset);
	compute (r.raw_pixels_NT, poffset + pscale * mn, poffset + pscale * mx, cnt, nbins, nanv, pscale, poffset);
}

void IntensityHistogramFeatures::save_value (std::vector<std::vector<double>>& fvals)
{
	for (auto& pr : result_)
		fvals[pr.first][0] = pr.second;
}

void IntensityHistogramFeatures::extract (LR& r, const Fsettings& fs, const Dataset& ds)
{
	IntensityHistogramFeatures f;
	f.calculate (r, fs, ds);
	f.save_value (r.fvals);
}

void IntensityHistogramFeatures::reduce (
	size_t start,
	size_t end,
	std::vector<int>* roi_labels,
	std::unordered_map <int, LR>* roi_data,
	const Fsettings& fsett,
	const Dataset& ds)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*roi_labels)[i];
		LR& r = (*roi_data)[lab];
		extract (r, fsett, ds);
	}
}

void IntensityHistogramFeatures::cleanup_instance()
{
	result_.clear();
}
