#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include "sensemaker.h"

// Macro to make some file i/o calls platform-independent
#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

void print_label_stats()
{
	std::cout << "\tFeatures by label. Number of processed labels " << uniqueLabels.size() << std::endl;

	// Print stats by label
	/*
	print_by_label("Min", min);
	print_by_label("Max", max);
	print_by_label("Mean", mean);
	print_by_label("Median", median);
	print_by_label("Energy", massEnergy);
	print_by_label("Variance", variance);
	*/
}

// Saves the result of image scanning and feature calculation. Must be called after the reduction phase.
bool save_features_2_csv (std::string inputFpath, std::string outputDir)
{
	// Sort the labels
	std::vector<int> L { uniqueLabels.begin(), uniqueLabels.end() };
	std::sort (L.begin(), L.end());

	// Tear off the directory part of 'inputPath', we don't need it
	std::string fullPath = outputDir + "/" + getPureFname(inputFpath) + ".csv";

	std::cout << "Creating file " << fullPath << std::endl;

	//fullPath.replace("//", "/");
	//fullPath.replace("\\\\", "\\");

	// Output label data
	// -- Create a file
	FILE* fp = nullptr;
	fopen_s(&fp, fullPath.c_str(), "w");
	if (!fp) 
	{
		std::perror("fopen failed"); 
		return false;
	}

	// -- Configure buffered write
	if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0) {
		std::perror("setvbuf failed"); 
		return false;
	}
	
	// -- Header
	fprintf(fp,
		// Intensity stats:
		"label"
		",mean"
		",median"
		",min"
		",max"
		",range"
		",standard_deviation"
		",skewness"
		",kurtosis"
		",mean_absolute_deviation"
		",energy"
		",root_mean_squared"
		",entropy"
		",mode"
		",uniformity"
		",P10,P25,P75,P90"
		",interquartile_range"
		",robust_mean_absolute_deviation"
		",weighted_centroid_y"
		",weighted_centroid_x"

		// Morphology:
		",area"	// aka pixels count
		",centroid_x"
		",centroid_y"
		",bbox_ymin"
		",bbox_xmin"
		",bbox_height"
		",bbox_width"

		",major_axis_length"
		",minor_axis_length"
		",eccentricity"
		",orientation"
		",neighbors"
		",extent"
		",aspect_ratio"

		",equivalent_diameter"
		",convex_hull_area"
		",solidity"
		",perimeter"
		",circularity"
		",integratedIntensityEdge"	
		",maxIntensityEdge"
		",minIntensityEdge"
		",meanIntensityEdge"
		",stddevIntensityEdge"

		",extremaP1_x , extremaP1_y"
		",extremaP2_x , extremaP2_y"
		",extremaP3_x , extremaP3_y"
		",extremaP4_x , extremaP4_y"
		",extremaP5_x , extremaP5_y"
		",extremaP6_x , extremaP6_y"
		",extremaP7_x , extremaP7_y"
		",extremaP8_x , extremaP8_y"

		",minFeretDiameter"
		",maxFeretDiameter"
		",minFeretAngle"
		",maxFeretAngle"
		",stat_feretDiam_min"
		",stat_feretDiam_max"
		",stat_feretDiam_mean"
		",stat_feretDiam_median"
		",stat_feretDiam_stddev"
		",stat_feretDiam_mode"

		",stat_martinDiam_min"
		",stat_martinDiam_max"
		",stat_martinDiam_mean"
		",stat_martinDiam_median"
		",stat_martinDiam_stddev"
		",stat_martinDiam_mode"

		",stat_nassensteinDiam_min"
		",stat_nassensteinDiam_max"
		",stat_nassensteinDiam_mean"
		",stat_nassensteinDiam_median"
		",stat_nassensteinDiam_stddev"
		",stat_nassensteinDiam_mode"

		",euler_nuber"

		",polygonality_ave"
		",hexagonality_ave"
		",hexagonality_stddev"

		",diameter_min_enclosing_circle"
		",diameter_circumscribing_circle"
		",diameter_inscribing_circle"
		",geodeticLength"
		",thickness");

	// Haralick 2D
	auto l = L.begin();
	LR & r = labelData[*l];
	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_AngularSecondMoment_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_Contrast_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_Correlation_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_Variance_%f", a);
			
	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_InverseDifferenceMoment_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_SumAverage_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_SumVariance_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_SumEntropy_%f", a);
			
	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_Entropy_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_DifferenceVariance_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_DifferenceEntropy_%f", a);
			
	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_InfoMeas1_%f", a);

	for (auto a : r.texture_Feature_Angles)
		fprintf(fp, ",Texture_InfoMeas2_%f", a);

	// Zernike 2D
	for (int i = 0; i <= LR::aux_ZERNIKE2D_ORDER; i++)
		if (i % 2)
			for (int j = 1; j <= i; j += 2)
				fprintf(fp, ",Z_%d_%d", i, j);
		else
			for (int j = 0; j <= i; j += 2)
				fprintf(fp, ",Z_%d_%d", i, j);
	
	fprintf(fp, "\n");

	// -- Dump numbers
	unsigned int cnt = 1; 
	for (auto l : L)
	{
		std::stringstream ss;
		
		LR& r = labelData[l];
		auto _pixCnt = r.pixelCountRoiArea;
		auto _min = r.min;
		auto _max = r.max;
		auto _range = _max - _min;
		auto _mean = r.mean;
		auto _median = r.median;
		auto _energy = r.massEnergy;
		auto _stdev = r.stddev;
		auto _skew = r.skewness;
		auto _kurt = r.kurtosis;
		auto _mad = r.MAD;
		auto _rms = r.RMS;
		auto _wcx = r.centroid_x,
			_wcy = r.centroid_y;
		auto _entro = r.entropy;
		auto _p10 = r.p10,
			_p25 = r.p25,
			_p75 = r.p75,
			_p90 = r.p90,
			_iqr = r.IQR,
			_rmad = r.RMAD,
			_mode = r.mode,
			_unifo = r.uniformity;

		auto _bbox_ymin = r.aabb.get_ymin(),
			_bbox_xmin = r.aabb.get_xmin(),
			_bbox_height = r.aabb.get_height(),
			_bbox_width = r.aabb.get_width();

		auto _maj_axis = r.major_axis_length, 
			_min_axis = r.minor_axis_length,
			_eccentricity = r.eccentricity, 
			_orientation = r.orientation;

		auto _num_neigs = r.num_neighbors;
		auto _extent = r.extent;
		auto _asp_rat = r.aspectRatio;

		auto _equiv_diam = r.equivDiam;
		auto _convHullArea = r.convHullArea;
		auto _solidity = r.solidity;
		auto _circ = r.circularity;

		ss << l << " , "
			// pixel intensity stats
			<< _mean << " , "
			<< _median << " , "
			<< _min << " , "
			<< _max << " , "
			<< _range << " , "
			<< _stdev << " , "
			<< _skew << " , "
			<< _kurt << " , "
			<< _mad << " , "
			<< _energy << " , "
			<< _rms << " , "
			<< _entro << " , "
			<< _mode << " , "
			<< _unifo << " , "
			<< _p10 << " , "
			<< _p25 << " , "
			<< _p75 << " , "
			<< _p90 << " , "
			<< _iqr << " , "
			<< _rmad << " , "
			<< _wcy << " , "
			<< _wcx << " , "
			// morphology
			<< _pixCnt << " , "
			<< _wcx << " , "
			<< _wcy << " , "
			<< _bbox_ymin << " , "
			<< _bbox_xmin << " , "
			<< _bbox_height << " , "
			<< _bbox_width << " , "

			<< _maj_axis << " , "
			<< _min_axis << " , "
			<< _eccentricity << " , "
			<< _orientation << " , "

			<< _num_neigs << " , "
			<< _extent << " , "
			<< _asp_rat << " , "
			<< _equiv_diam << " , "
			<< _convHullArea << " , "
			<< _solidity << " , "
			<< r.roiPerimeter << " , "
			<< _circ

			<< "," << r.CellProfiler_Intensity_IntegratedIntensityEdge
			<< "," << r.CellProfiler_Intensity_MaxIntensityEdge
			<< "," << r.CellProfiler_Intensity_MinIntensityEdge
			<< "," << r.CellProfiler_Intensity_MeanIntensityEdge
			<< "," << r.CellProfiler_Intensity_StddevIntensityEdge

			<< r.extremaP1x << " , " << r.extremaP1y << " , "
			<< r.extremaP2x << " , " << r.extremaP2y << " , "
			<< r.extremaP3x << " , " << r.extremaP3y << " , "
			<< r.extremaP4x << " , " << r.extremaP4y << " , "
			<< r.extremaP5x << " , " << r.extremaP5y << " , "
			<< r.extremaP6x << " , " << r.extremaP6y << " , "
			<< r.extremaP7x << " , " << r.extremaP7y << " , "
			<< r.extremaP8x << " , " << r.extremaP8y << " , "

			<< r.minFeretDiameter << " , "
			<< r.maxFeretDiameter << " , "
			<< r.minFeretAngle << " , "
			<< r.maxFeretAngle << " , "
			<< r.feretStats_minD << " , "
			<< r.feretStats_maxD << " , "
			<< r.feretStats_meanD << " , "
			<< r.feretStats_medianD << " , "
			<< r.feretStats_stddevD << " , "
			<< r.feretStats_modeD << " , "

			<< r.martinStats_minD << " , "
			<< r.martinStats_maxD << " , "
			<< r.martinStats_meanD << " , "
			<< r.martinStats_medianD << " , "
			<< r.martinStats_stddevD << " , "
			<< r.martinStats_modeD << " , "

			<< r.nassStats_minD << " , "
			<< r.nassStats_maxD << " , "
			<< r.nassStats_meanD << " , "
			<< r.nassStats_medianD << " , "
			<< r.nassStats_stddevD << " , "
			<< r.nassStats_modeD << " , "
			<< r.euler_number << " , "
			<< r.polygonality_ave << " , "
			<< r.hexagonality_ave << " , "
			<< r.hexagonality_stddev << " , "
			<< r.diameter_min_enclosing_circle << " , "
			<< r.diameter_circumscribing_circle << " , "
			<< r.diameter_inscribing_circle << " , "
			<< r.geodeticLength << " , "
			<< r.thickness;

			// assuming r.texture_* contains items for 0, 45, 90, and 135 degrees to match the header, otherwise we need to adjust the header
			for (auto f : r.texture_AngularSecondMoments)
				ss << "," << f;
			for (auto f : r.texture_Contrast)
				ss << "," << f;
			for (auto f : r.texture_Correlation)
				ss << "," << f;
			for (auto f : r.texture_Variance)
				ss << "," << f;
			for (auto f : r.texture_InverseDifferenceMoment)
				ss << "," << f;
			for (auto f : r.texture_SumAverage)
				ss << "," << f;
			for (auto f : r.texture_SumVariance)
				ss << "," << f;
			for (auto f : r.texture_SumEntropy)
				ss << "," << f;
			for (auto f : r.texture_Entropy)
				ss << "," << f;
			for (auto f : r.texture_DifferenceVariance)
				ss << "," << f;
			for (auto f : r.texture_DifferenceEntropy)
				ss << "," << f;
			for (auto f : r.texture_InfoMeas1)
				ss << "," << f;
			for (auto f : r.texture_InfoMeas2)
				ss << "," << f;

			// Zernike 2D
			int zIdx = 0;
			for (int i = 0; i <= LR::aux_ZERNIKE2D_ORDER; i++)
				if (i % 2)
					for (int j = 1; j <= i; j += 2)
					{
						ss << "," << r.Zernike2D [zIdx++];
						
						#if 0
						// Debug:
						auto z = r.Zernike2D[zIdx++];
						std::cout << " z[" << zIdx << "]_" << i << "," << j << " = " << z;
						ss << "," << z;
						#endif
					}
				else
					for (int j = 0; j <= i; j += 2)
					{
						ss << "," << r.Zernike2D [zIdx++];
						
						#if 0
						// Debug:
						auto z = r.Zernike2D[zIdx++];
						std::cout << " z[" << zIdx << "]_" << i << "," << j << " = " << z;
						ss << "," << z;
						#endif
					}
			
		fprintf (fp, "%s\n", ss.str().c_str());
	}
	std::fflush(fp);
	std::fclose(fp);

	#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
	// Output label's intensities for debug
	for (auto l : L)
	{
		if (l != SANITY_CHECK_INTENSITIES_FOR_LABEL)
			continue;
		
		std::stringstream ss;
		LR& lr = labelData[l];
		auto& I = lr.raw_intensities;
		ss << outputDir << "/" << "intensities_label_" << l << ".txt";
		fullPath = ss.str();	
		std::cout << "Dumping intensities of label " << l << " to file " << fullPath << std::endl;


		fopen_s(&fp, fullPath.c_str(), "w");
		if (fp)
		{
			ss.clear();
			ss << "I_" << l << " = [\n";
			for (auto w : I)
				ss << "\t" << w << ", \n";
			ss << "\t]; \n";
			fprintf (fp, "%s\n", ss.str().c_str());
			std::fclose(fp);
		}		
	}
	#endif

	return true;
}


// Diagnostic function
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns)
{
	std::stringstream ss;

	std::cout << std::endl << featureName << std::endl;
	
	int i = 1;
	for (auto& x : L)
	{
		ss << 'L' << x.first << ':' << x.second;
		std::cout << std::setw(10) << ss.str();
		ss.str (std::string()); // Clear ss

		if (i++ % numColumns == 0)
			std::cout << std::endl;
	}

}

// Another diagnostic function
void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns)
{
	std::stringstream ss;

	std::cout << std::endl << featureName << std::endl;

	int i = 1;
	for (auto& x : L)
	{
		ss << 'L' << x.first << ':' << x.second;
		std::cout << std::setw(30) << ss.str();
		ss.str(std::string()); // Clear ss

		if (i++ % numColumns == 0)
			std::cout << std::endl;
	}

}