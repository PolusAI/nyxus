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
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/glrlm.h"
#include "features/zernike.h"
#include "globals.h"

namespace Nyxus
{
	// Macro to make some file i/o calls platform-independent
#ifndef _WIN32
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
	bool save_features_2_csv (std::string intFpath, std::string segFpath, std::string outputDir)
	{
		// Sort the labels
		std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(L.begin(), L.end());

		FILE* fp = nullptr;

		static bool mustRenderHeader = true;	// This can be flipped to 'false' in 'singlecsv' scenario

		if (theEnvironment.separateCsv)
		{
			std::string fullPath = outputDir + "/_INT_" + getPureFname(intFpath) + "_SEG_" + getPureFname(segFpath) + ".csv";
			VERBOSLVL1(std::cout << "\t--> " << fullPath << "\n";)
			fopen_s(&fp, fullPath.c_str(), "w");
		}
		else
		{
			std::string fullPath = outputDir + "/" + "NyxusFeatures.csv";
			VERBOSLVL1(std::cout << "\t--> " << fullPath << "\n";)
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
				auto fn = std::get<0>(enabdF);	// feature name
				auto fc = std::get<1>(enabdF);	// feature code

				// Handle missing feature name (which is a significant issue!) in order to at least be able to trace back to the feature code
				if (fn.empty())
				{
					std::stringstream temp;
					temp << "feature" << fc;
					fn = temp.str();
				}

				// Parameterized feature
				// --Texture family
				bool textureFeature =
					fc == GLCM_ANGULAR2NDMOMENT ||
					fc == GLCM_CONTRAST ||
					fc == GLCM_CORRELATION ||
					fc == GLCM_VARIANCE ||
					fc == GLCM_INVERSEDIFFERENCEMOMENT ||
					fc == GLCM_SUMAVERAGE ||
					fc == GLCM_SUMVARIANCE ||
					fc == GLCM_SUMENTROPY ||
					fc == GLCM_ENTROPY ||
					fc == GLCM_DIFFERENCEVARIANCE ||
					fc == GLCM_DIFFERENCEENTROPY ||
					fc == GLCM_INFOMEAS1 ||
					fc == GLCM_INFOMEAS2;
				if (textureFeature)
				{
					// Polulate with angles
					for (auto ang : theEnvironment.rotAngles)
					{
						// CSV separator
						//if (ang != theEnvironment.rotAngles[0])
						//	ssHead << ",";
						ssHead << "," << fn << "_" << ang;
					}
					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature =
					fc == GLRLM_SRE ||
					fc == GLRLM_LRE ||
					fc == GLRLM_GLN ||
					fc == GLRLM_GLNN ||
					fc == GLRLM_RLN ||
					fc == GLRLM_RLNN ||
					fc == GLRLM_RP ||
					fc == GLRLM_GLV ||
					fc == GLRLM_RV ||
					fc == GLRLM_RE ||
					fc == GLRLM_LGLRE ||
					fc == GLRLM_HGLRE ||
					fc == GLRLM_SRLGLE ||
					fc == GLRLM_SRHGLE ||
					fc == GLRLM_LRLGLE ||
					fc == GLRLM_LRHGLE;
				if (glrlmFeature)
				{
					// Polulate with angles
					for (auto ang : GLRLMFeature::rotAngles)
					{
						ssHead << "," << fn << "_" << ang;
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					// Generate the feature value list
					for (auto i = 0; i < GaborFeature::num_features; i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				if (fc == FRAC_AT_D)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				if (fc == MEAN_FRAC)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				if (fc == RADIAL_CV)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				// --Zernike features header 
				if (fc == ZERNIKE2D)
				{
					// Populate with indices
					for (int i = 0; i < ZernikeFeature::num_feature_values_calculated; i++)
						ssHead << "," << fn << "_Z" << i;						

					// Proceed with other features
					continue;
				}

				// Regular feature
				ssHead << "," << fn;
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

			LR& r = roiData[l];

			// Tear off pure file names from segment and intensity file paths
			std::filesystem::path pseg(r.segFname), pint(r.intFname);
			ssVals << pseg.filename() << "," << pint.filename() << "," << l;

			for (auto& enabdF : F)
			{
				auto fc = std::get<1>(enabdF);
				auto fn = std::get<0>(enabdF);	// debug
				auto vv = r.getFeatureValues(fc);

				// Parameterized feature
				// --Texture family
				bool textureFeature =
					fc == GLCM_ANGULAR2NDMOMENT ||
					fc == GLCM_CONTRAST ||
					fc == GLCM_CORRELATION ||
					fc == GLCM_VARIANCE ||
					fc == GLCM_INVERSEDIFFERENCEMOMENT ||
					fc == GLCM_SUMAVERAGE ||
					fc == GLCM_SUMVARIANCE ||
					fc == GLCM_SUMENTROPY ||
					fc == GLCM_ENTROPY ||
					fc == GLCM_DIFFERENCEVARIANCE ||
					fc == GLCM_DIFFERENCEENTROPY ||
					fc == GLCM_INFOMEAS1 ||
					fc == GLCM_INFOMEAS2;
				if (textureFeature)
				{
					// Polulate with angles
					for (int i = 0; i < theEnvironment.rotAngles.size(); i++)
					{
						ssVals << "," << vv[i];
						//--diagnoze misalignment-- ssVals << "," << fn << "-" << vv[i];	
					}
					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature =
					fc == GLRLM_SRE ||
					fc == GLRLM_LRE ||
					fc == GLRLM_GLN ||
					fc == GLRLM_GLNN ||
					fc == GLRLM_RLN ||
					fc == GLRLM_RLNN ||
					fc == GLRLM_RP ||
					fc == GLRLM_GLV ||
					fc == GLRLM_RV ||
					fc == GLRLM_RE ||
					fc == GLRLM_LGLRE ||
					fc == GLRLM_HGLRE ||
					fc == GLRLM_SRLGLE ||
					fc == GLRLM_SRHGLE ||
					fc == GLRLM_LRLGLE ||
					fc == GLRLM_LRHGLE;
				if (glrlmFeature)
				{
					// Polulate with angles
					auto nAng = 4; // sizeof(GLRLMFeature::rotAngles) / sizeof(GLRLMFeature::rotAngles[0]);
					for (int i = 0; i < nAng; i++)
					{
						ssVals << "," << vv[i];
						//--diagnoze misalignment-- ssVals << "," << fn << "-" << vv[i];	
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					for (auto i = 0; i < GaborFeature::num_features; i++)
					{
						ssVals << "," << vv[i];
						//--diagnoze misalignment-- ssVals << "," << fn << "-" << vv[i];	
					}

					// Proceed with other features
					continue;
				}

				// --Zernike feature values
				if (fc == ZERNIKE2D)
				{
					for (int i=0; i <ZernikeFeature::num_feature_values_calculated; i++)
						ssVals << "," << vv[i]; 

					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					{
						ssVals << "," << vv[i];
						//--diagnoze misalignment-- ssVals << "," << fn << "-" << vv[i];	
					}
					// Proceed with other features
					continue;
				}
				if (fc == MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					{
						ssVals << "," << vv[i];
						//--diagnoze misalignment-- ssVals << "," << fn << "-" << vv[i];	
					}
					// Proceed with other features
					continue;
				}
				if (fc == RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					{
						ssVals << "," << vv[i];
						//--diagnoze misalignment-- ssVals << "," << fn << "-" << vv[i];	
					}
					// Proceed with other features
					continue;
				}

				// Regular feature
				ssVals << "," << vv[0];
				//--diagnoze misalignment-- ssVals << "," << fn << "-" << vv[0];	
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
			LR& lr = roiData[l];
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
				fprintf(fp, "%s\n", ss.str().c_str());
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
			ss.str(std::string()); // Clear ss

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

}