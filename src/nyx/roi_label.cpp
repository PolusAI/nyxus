#include "sensemaker.h"

void LR::init_aabb(StatsInt x, StatsInt y)
{
	aabb.init_x(x);
	aabb.init_y(y);
	num_neighbors = 0;
}

void LR::update_aabb(StatsInt x, StatsInt y)
{
	aabb.update_x(x);
	aabb.update_y(y);
}

// Prerequisite: availability of 'contour'
void LR::reduce_edge_intensity_features()
{
	StatsReal 
		sumI = 0.0, 
		maxI = -INF, 
		minI = INF, 
		meanI = 0.0, 
		stddevI = 0.0, 
		n = contour.contour_pixels.size();

	// sum, min, max
	for (auto pxl : contour.contour_pixels)
	{
		StatsReal I = pxl.inten;	// cast inten to result type here
		sumI += I;
		maxI = std::max(maxI, I);
		minI = std::min(minI, I);
	}

	// mean
	meanI = sumI / n;

	// stddev
	for (auto pxl : contour.contour_pixels)
	{
		StatsReal I = pxl.inten;	// cast inten to result type here
		stddevI += (I - meanI) * (I - meanI);
	}
	stddevI = std::sqrt(stddevI / n);

	// Save
	CellProfiler_Intensity_IntegratedIntensityEdge = sumI;
	CellProfiler_Intensity_MaxIntensityEdge = maxI;
	CellProfiler_Intensity_MinIntensityEdge = minI;
	CellProfiler_Intensity_MeanIntensityEdge = meanI;
	CellProfiler_Intensity_StddevIntensityEdge = stddevI;	
}

void LR::reduce_pixel_intensity_features()
{
	LR& lr = *this;
	for (auto& pxl : lr.raw_pixels)
	{
		auto intensity = pxl.inten;
		auto x = pxl.x, y = pxl.y;

		// Count of pixels belonging to the label
		auto prev_n = lr.pixelCountRoiArea;	// Previous count
		lr.aux_PrevCount = prev_n;
		auto n = prev_n + 1;	// New count
		lr.pixelCountRoiArea = n;

		// Cumulants for moments calculation
		auto prev_mean = lr.mean;
		auto delta = intensity - prev_mean;
		auto delta_n = delta / n;
		auto delta_n2 = delta_n * delta_n;
		auto term1 = delta * delta_n * prev_n;

		// Mean
		auto mean = prev_mean + delta_n;
		lr.mean = mean;

		// Moments
		lr.aux_M4 = lr.aux_M4 + term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * lr.aux_M2 - 4 * delta_n * lr.aux_M3;
		lr.aux_M3 = lr.aux_M3 + term1 * delta_n * (n - 2) - 3 * delta_n * lr.aux_M2;
		lr.aux_M2 = lr.aux_M2 + term1;

		// Min 
		lr.min = std::min(lr.min, (StatsInt)intensity);

		// Max
		lr.max = std::max(lr.max, (StatsInt)intensity);

		// Energy
		lr.massEnergy = lr.massEnergy + intensity * intensity;

		// Variance and standard deviation
		if (n >= 2)
		{
			double s_prev = lr.variance,
				diff = double(intensity) - prev_mean,
				diff2 = diff * diff;
			lr.variance = (n - 2) * s_prev / (n - 1) + diff2 / n;
		}
		else
			lr.variance = 0;

		// Mean absolute deviation
		lr.MAD = lr.MAD + std::abs(intensity - mean);

		// Weighted centroids. Needs reduction. Do we need to make them 1-based for compatibility with Matlab and WNDCHRM?
		lr.centroid_x = lr.centroid_x + StatsReal(x);
		lr.centroid_y = lr.centroid_y + StatsReal(y);

		// Histogram
		auto ptrH = lr.aux_Histogram;
		ptrH->add_observation(intensity);

		// Previous intensity for succeeding iterations
		lr.aux_PrevIntens = intensity;

		//==== Morphology
		lr.update_aabb(x, y);

	#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
		// Dump intensities for testing
		if (label == SANITY_CHECK_INTENSITIES_FOR_LABEL)	// Put the label code you're tracking
			lr.raw_intensities.push_back(intensity);
	#endif
	}
}


