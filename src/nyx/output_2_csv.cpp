#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include "environment.h"
#include "f_radial_distribution.h"
#include "gabor.h"
#include "glrlm.h"
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

	FILE* fp = nullptr;

	static bool mustRenderHeader = true;	// This can be flipped to 'false' in 'singlecsv' scenario

	if (theEnvironment.separateCsv)
	{
		std::string fullPath = outputDir + "/" + getPureFname(inputFpath) + ".csv";
		std::cout << "\t--> " << fullPath << "\n";
		fopen_s(&fp, fullPath.c_str(), "w");
	}
	else
	{
		std::string fullPath = outputDir + "/" + "NyxusFeatures.csv";
		std::cout << "\t--> " << fullPath << "\n";
		auto mode = mustRenderHeader ? "w" : "a";
		fopen_s(&fp, fullPath.c_str(), mode);
	}
	
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

	// Learn what features need to be displayed
	std::vector<std::tuple<std::string, AvailableFeatures>> F = theFeatureSet.getEnabledFeatures();

	// -- Header
	if (mustRenderHeader)
	{
		std::stringstream ssHead;

		ssHead << "mask_image,intensity_image,label";

		for (auto& enabdF : F)
		{
			auto fname = std::get<0>(enabdF);
			auto fcode = std::get<1>(enabdF);

			// Parameterized feature
			// --Texture family
			bool textureFeature =
				fcode == GLCM_ANGULAR2NDMOMENT ||
				fcode == GLCM_CONTRAST ||
				fcode == GLCM_CORRELATION ||
				fcode == GLCM_VARIANCE ||
				fcode == GLCM_INVERSEDIFFERENCEMOMENT ||
				fcode == GLCM_SUMAVERAGE ||
				fcode == GLCM_SUMVARIANCE ||
				fcode == GLCM_SUMENTROPY ||
				fcode == GLCM_ENTROPY ||
				fcode == GLCM_DIFFERENCEVARIANCE ||
				fcode == GLCM_DIFFERENCEENTROPY ||
				fcode == GLCM_INFOMEAS1 ||
				fcode == GLCM_INFOMEAS2;
			if (textureFeature)
			{
				// Polulate with angles
				for (auto ang : theEnvironment.rotAngles)
				{
					// CSV separator
					//if (ang != theEnvironment.rotAngles[0])
					//	ssHead << ",";
					ssHead << "," << fname << "_" << ang;
				}
				// Proceed with other features
				continue;
			}

			// --GLRLM family
			bool glrlmFeature =
				fcode == GLRLM_SRE ||
				fcode == GLRLM_LRE ||
				fcode == GLRLM_GLN ||
				fcode == GLRLM_GLNN ||
				fcode == GLRLM_RLN ||
				fcode == GLRLM_RLNN ||
				fcode == GLRLM_RP ||
				fcode == GLRLM_GLV ||
				fcode == GLRLM_RV ||
				fcode == GLRLM_RE ||
				fcode == GLRLM_LGLRE ||
				fcode == GLRLM_HGLRE ||
				fcode == GLRLM_SRLGLE ||
				fcode == GLRLM_SRHGLE ||
				fcode == GLRLM_LRLGLE ||
				fcode == GLRLM_LRHGLE;
			if (glrlmFeature)
			{
				// Polulate with angles
				for (auto ang : GLRLM_features::rotAngles)
				{
					ssHead << "," << fname << "_" << ang;
				}
				// Proceed with other features
				continue;
			}

			// --Gabor
			if (fcode == GABOR)
			{
				// Generate the feature value list
				for (auto i = 0; i < GaborFeatures::num_features; i++)
					ssHead << "," << fname << "_" << i;

				// Proceed with other features
				continue;
			}

			if (fcode == FRAC_AT_D)
			{
				// Generate the feature value list
				for (auto i = 0; i < RadialDistribution::num_features_FracAtD; i++)
					ssHead << "," << fname << "_" << i;

				// Proceed with other features
				continue;
			}

			if (fcode == MEAN_FRAC)
			{
				// Generate the feature value list
				for (auto i = 0; i < RadialDistribution::num_features_MeanFrac; i++)
					ssHead << "," << fname << "_" << i;

				// Proceed with other features
				continue;
			}

			if (fcode == RADIAL_CV)
			{
				// Generate the feature value list
				for (auto i = 0; i < RadialDistribution::num_features_RadialCV; i++)
					ssHead << "," << fname << "_" << i;

				// Proceed with other features
				continue;
			}

			// --Zernike family
			if (fcode == ZERNIKE2D)
			{
				// Populate with indices
				for (int i = 0; i <= LR::aux_ZERNIKE2D_ORDER; i++)
					if (i % 2)
						for (int j = 1; j <= i; j += 2)
						{
							// CSV separator
							//if (j > 1)
							//	ssHead << ",";
							ssHead << "," << fname << "_" << i << "_" << j;
						}
					else
						for (int j = 0; j <= i; j += 2)
						{
							// CSV separator
							//if (j > 1)
							//	ssHead << ",";
							ssHead << "," << fname << "_" << i << "_" << j;
						}

				// Proceed with other features
				continue;
			}

			// Regular feature
			ssHead << "," << fname;
		}
		fprintf(fp, "%s\n", ssHead.str().c_str());

		// Prevent rendering the header again for another image's portion of labels
		if (theEnvironment.separateCsv == false)
			mustRenderHeader = false;
	}

	// -- Values
	for (auto l : L)
	{
		std::stringstream ssVals;

		LR& r = labelData[l];

		ssVals << r.segFname << "," << r.intFname << "," << l;
			
		for (auto& enabdF : F)
		{
			auto fcode = std::get<1> (enabdF);
			auto fname = std::get<0>(enabdF);	// debug
			auto vv = r.getFeatureValues (fcode);

			// Parameterized feature
			// --Texture family
			bool textureFeature =
				fcode == GLCM_ANGULAR2NDMOMENT ||
				fcode == GLCM_CONTRAST ||
				fcode == GLCM_CORRELATION ||
				fcode == GLCM_VARIANCE ||
				fcode == GLCM_INVERSEDIFFERENCEMOMENT ||
				fcode == GLCM_SUMAVERAGE ||
				fcode == GLCM_SUMVARIANCE ||
				fcode == GLCM_SUMENTROPY ||
				fcode == GLCM_ENTROPY ||
				fcode == GLCM_DIFFERENCEVARIANCE ||
				fcode == GLCM_DIFFERENCEENTROPY ||
				fcode == GLCM_INFOMEAS1 ||
				fcode == GLCM_INFOMEAS2;
			if (textureFeature)
			{
				// Polulate with angles
				for (int i=0; i<theEnvironment.rotAngles.size(); i++)
				{
					ssVals << "," << vv[i];
					//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[i];	
				}
				// Proceed with other features
				continue;
			}

			// --GLRLM family
			bool glrlmFeature =
				fcode == GLRLM_SRE ||
				fcode == GLRLM_LRE ||
				fcode == GLRLM_GLN ||
				fcode == GLRLM_GLNN ||
				fcode == GLRLM_RLN ||
				fcode == GLRLM_RLNN ||
				fcode == GLRLM_RP ||
				fcode == GLRLM_GLV ||
				fcode == GLRLM_RV ||
				fcode == GLRLM_RE ||
				fcode == GLRLM_LGLRE ||
				fcode == GLRLM_HGLRE ||
				fcode == GLRLM_SRLGLE ||
				fcode == GLRLM_SRHGLE ||
				fcode == GLRLM_LRLGLE ||
				fcode == GLRLM_LRHGLE;
			if (glrlmFeature)
			{
				// Polulate with angles
				for (int i=0; i<GLRLM_features::rotAngles.size(); i++)
				{
					ssVals << "," << vv[i];
					//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[i];	
				}
				// Proceed with other features
				continue;
			}

			// --Gabor
			if (fcode == GABOR)
			{
				for (auto i = 0; i < GaborFeatures::num_features; i++)
				{
					ssVals << "," << vv[i];
					//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[i];	
				}

				// Proceed with other features
				continue;
			}

			// --Zernike family
			if (fcode == ZERNIKE2D)
			{
				int zIdx = 0;
				for (int i = 0; i <= LR::aux_ZERNIKE2D_ORDER; i++)
					if (i % 2)
						for (int j = 1; j <= i; j += 2)
						{
							ssVals << "," << vv[zIdx++]; // former r.Zernike2D[zIdx++];
							//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[zIdx++];
						}
					else
						for (int j = 0; j <= i; j += 2)
						{
							ssVals << "," << vv[zIdx++]; // former r.Zernike2D[zIdx++];
							//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[zIdx++];
						}

				// Proceed with other features
				continue;
			}

			// --Radial distribution features
			if (fcode == FRAC_AT_D)
			{
				for (auto i = 0; i < RadialDistribution::num_features_FracAtD; i++)
				{
					ssVals << "," << vv[i];
					//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[i];	
				}
				// Proceed with other features
				continue;
			}
			if (fcode == MEAN_FRAC)
			{
				for (auto i = 0; i < RadialDistribution::num_features_MeanFrac; i++)
				{
					ssVals << "," << vv[i];
					//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[i];	
				}
				// Proceed with other features
				continue;
			}
			if (fcode == RADIAL_CV)
			{
				for (auto i = 0; i < RadialDistribution::num_features_RadialCV; i++)
				{
					ssVals << "," << vv[i];
					//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[i];	
				}
				// Proceed with other features
				continue;
			}

			// Regular feature
			ssVals << "," << vv[0];
			//--diagnoze misalignment-- ssVals << "," << fname << "-" << vv[0];	
		}

		fprintf(fp, "%s\n", ssVals.str().c_str());
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
