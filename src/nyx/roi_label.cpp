#include "sensemaker.h"

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