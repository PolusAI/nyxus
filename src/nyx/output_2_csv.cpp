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
	print_by_label("Min", labelMins);
	print_by_label("Max", labelMaxs);
	print_by_label("Mean", mean);
	print_by_label("Median", labelMedians);
	print_by_label("Energy", labelMassEnergy);
	print_by_label("Variance", labelVariance);
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
	fprintf (fp,
		// Intensity stats:
		"label , "
		"mean, "
		"median , "
		"min , "
		"max , "
		"range , "
		"standard_deviation , "
		"skewness , "
		"kurtosis , "
		"mean_absolute_deviation , "
		"energy , "
		"root_mean_squared , "
		"entropy , "
		"mode , "
		"uniformity , "
		"P10 , P25 , P75 , P90 , "
		"interquartile_range , "
		"robust_mean_absolute_deviation , "
		"weighted_centroid_y , "
		"weighted_centroid_x , "

		// Morphology:
		"area , "	// aka pixels count
		"centroid_x , "
		"centroid_y , "
		"bbox_ymin , "
		"bbox_xmin , "	
		"bbox_height , "
		"bbox_width , "

		"major_axis_length , "
		"minor_axis_length , "
		"eccentricity , "
		"orientation , "
		"neighbors , "
		"extent , "
		"aspect_ratio , "

		"equivalent_diameter , "
		"convex_hull_area , "
		"solidity , "
		"perimeter , "
		"circularity , "

		"extremaP1_x , extremaP1_y , "
		"extremaP2_x , extremaP2_y , "
		"extremaP3_x , extremaP3_y , "
		"extremaP4_x , extremaP4_y , "
		"extremaP5_x , extremaP5_y , "
		"extremaP6_x , extremaP6_y , "
		"extremaP7_x , extremaP7_y , "
		"extremaP8_x , extremaP8_y "

		"minFeretDiameter , "
		"maxFeretDiameter , "
		"minFeretAngle , "
		"maxFeretAngle , "
		"stat_feretDiam_min , "
		"stat_feretDiam_max , "
		"stat_feretDiam_mean , "
		"stat_feretDiam_median , "
		"stat_feretDiam_stddev , "
		"stat_feretDiam_mode , "

		"stat_martinDiam_min , "
		"stat_martinDiam_max , "
		"stat_martinDiam_mean , "
		"stat_martinDiam_median , "
		"stat_martinDiam_stddev , "
		"stat_martinDiam_mode , "

		"stat_nassensteinDiam_min , "
		"stat_nassensteinDiam_max , "
		"stat_nassensteinDiam_mean , "
		"stat_nassensteinDiam_median , "
		"stat_nassensteinDiam_stddev , "
		"stat_nassensteinDiam_mode , "

		"euler_nuber, "

		"polygonality_ave , "
		"hexagonality_ave , "
		"hexagonality_stddev , "

		"diameter_min_enclosing_circle , "
		"diameter_circumscribing_circle , "
		"diameter_inscribing_circle , "
		"geodeticLength , "
		"thickness"

		"\n");

	// -- Dump numbers
	unsigned int cnt = 1; 
	for (auto l : L)
	{
		std::stringstream ss;
		
		LR& r = labelData[l];
		auto _pixCnt = r.pixelCount;
		auto _min = r.labelMins;
		auto _max = r.labelMaxs;
		auto _range = _max - _min;
		auto _mean = r.mean;
		auto _median = r.labelMedians;
		auto _energy = r.labelMassEnergy;
		auto _stdev = r.labelStddev;
		auto _skew = r.labelSkewness;
		auto _kurt = r.labelKurtosis;
		auto _mad = r.labelMAD;
		auto _rms = r.labelRMS;
		auto _wcx = r.centroid_x,
			_wcy = r.centroid_y;
		auto _entro = r.labelEntropy;
		auto _p10 = r.labelP10,
			_p25 = r.labelP25,
			_p75 = r.labelP75,
			_p90 = r.labelP90,
			_iqr = r.labelIQR,
			_rmad = r.labelRMAD,
			_mode = r.labelMode,
			_unifo = r.labelUniformity;

		auto _bbox_ymin = r.aabb_ymin,
			_bbox_xmin = r.aabb_xmin,
			_bbox_height = r.aabb_ymax - r.aabb_ymin,
			_bbox_width = r.aabb_xmax - r.aabb_xmin;

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

		ss	<< l		<< " , " 
			// pixel intensity stats
			<< _mean	<< " , " 
			<< _median	<< " , " 			
			<< _min		<< " , " 
			<< _max		<< " , " 
			<< _range	<< " , "
			<< _stdev	<< " , "
			<< _skew	<< " , "
			<< _kurt	<< " , "
			<< _mad		<< " , "			
			<< _energy	<< " , " 
			<< _rms		<< " , "
			<< _entro	<< " , "
			<< _mode	<< " , "
			<< _unifo	<< " , "
			<< _p10		<< " , "
			<< _p25		<< " , "
			<< _p75		<< " , " 
			<< _p90		<< " , "
			<< _iqr		<< " , "
			<< _rmad	<< " , "
			<< _wcy		<< " , "
			<< _wcx		<< " , "
			// morphology
			<< _pixCnt	<< " , "
			<< _wcx		<< " , "
			<< _wcy		<< " , "
			<< _bbox_ymin	<< " , "
			<< _bbox_xmin	<< " , "
			<< _bbox_height	<< " , "
			<< _bbox_width	<< " , "

			<< _maj_axis <<		" , "
			<< _min_axis <<		" , "
			<< _eccentricity <<	" , "
			<< _orientation <<	" , "
		
			<< _num_neigs	<< " , "
			<< _extent		<< " , " 
			<< _asp_rat		<< " , "
			<< _equiv_diam	<< " , "
			<< _convHullArea	<< " , "
			<< _solidity	<< " , "
			<< r.roiPerimeter << " , "
			<< _circ	<< " , "
			
			<< r.extremaP1x	<< " , "	<< r.extremaP1y	<< " , "
			<< r.extremaP2x	<< " , "	<< r.extremaP2y	<< " , "
			<< r.extremaP3x	<< " , "	<< r.extremaP3y	<< " , "
			<< r.extremaP4x	<< " , "	<< r.extremaP4y	<< " , "
			<< r.extremaP5x	<< " , "	<< r.extremaP5y	<< " , "
			<< r.extremaP6x	<< " , "	<< r.extremaP6y	<< " , "
			<< r.extremaP7x	<< " , "	<< r.extremaP7y	<< " , "
			<< r.extremaP8x	<< " , "	<< r.extremaP8y

			<< r.minFeretDiameter	<< " , "
			<< r.maxFeretDiameter	<< " , "
			<< r.minFeretAngle << " , "
			<< r.maxFeretAngle << " , "
			<< r.feretStats_minDiameter << " , "
			<< r.feretStats_maxDiameter << " , "
			<< r.feretStats_meanDiameter << " , "
			<< r.feretStats_medianDiameter << " , "
			<< r.feretStats_stddevDiameter << " , "
			<< r.feretStats_modeDiameter << " , "

			<< r.martinStats_minDiameter << " , "
			<< r.martinStats_maxDiameter << " , "
			<< r.martinStats_meanDiameter << " , "
			<< r.martinStats_medianDiameter << " , "
			<< r.martinStats_stddevDiameter << " , "
			<< r.martinStats_modeDiameter << " , "

			<< r.nassStats_minDiameter << " , "
			<< r.nassStats_maxDiameter << " , "
			<< r.nassStats_meanDiameter << " , "
			<< r.nassStats_medianDiameter << " , "
			<< r.nassStats_stddevDiameter << " , "
			<< r.nassStats_modeDiameter << " , "
			<< r.euler_number << " , "
			<< r.polygonality_ave << " , "
			<< r.hexagonality_ave << " , "
			<< r.hexagonality_stddev	<< " , "
			<< r.diameter_min_enclosing_circle	<< " , "
			<< r.diameter_circumscribing_circle	<< " , "
			<< r.diameter_inscribing_circle	<<	" , "
			<< r.geodeticLength	<< " , "
			<< r.thickness
			;

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
