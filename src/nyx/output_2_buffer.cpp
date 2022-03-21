#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "environment.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/glrlm.h"
#include "features/zernike.h"

namespace Nyxus
{
	bool save_features_2_buffer(std::vector<std::string>& headerBuf, std::vector<double>& resultBuf, std::vector<std::string>& stringColBuf)
	{
		std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(L.begin(), L.end());
		std::vector<std::tuple<std::string, AvailableFeatures>> F = theFeatureSet.getEnabledFeatures();

		// We only fill in the header once.
		// We depend on the caller to manage headerBuf contents and clear it appropriately...
		bool fill_header = headerBuf.size() == 0;

		// -- Header
		if (fill_header)
		{

			headerBuf.push_back("mask_image");
			headerBuf.push_back("intensity_image");
			headerBuf.push_back("label");

			for (auto& enabdF : F)
			{
				auto fn = std::get<0>(enabdF);	// feature name
				auto fc = std::get<1>(enabdF);	// feature code

				// Handle missing feature name (which is a significant issue!) in order to at least be able to trace back to the feature code
				if (fn.empty())
					fn = "feature" + std::to_string(fc);

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
						headerBuf.push_back(fn + "_" + std::to_string(ang));
					
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
						headerBuf.push_back(fn + "_" + std::to_string(ang));
					
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					// Generate the feature value list
					for (auto i = 0; i < GaborFeature::num_features; i++)
						headerBuf.push_back(fn + "_" + std::to_string(i));

					// Proceed with other features
					continue;
				}

				if (fc == FRAC_AT_D)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
						headerBuf.push_back(fn + "_" + std::to_string(i));

					// Proceed with other features
					continue;
				}

				if (fc == MEAN_FRAC)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
						headerBuf.push_back(fn + "_" + std::to_string(i));

					// Proceed with other features
					continue;
				}

				if (fc == RADIAL_CV)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
						headerBuf.push_back(fn + "_" + std::to_string(i));

					// Proceed with other features
					continue;
				}

				// --Zernike family
				if (fc == ZERNIKE2D)
				{
					// Populate with indices
					for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
						headerBuf.push_back (fn + "_" + std::to_string(i));

					// Proceed with other features
					continue;
				}

				// Regular feature
				headerBuf.push_back(fn);
			}
		}

		// -- Values
		for (auto l : L)
		{
			LR& r = roiData[l];
			totalNumLabels++;

			// Tear off pure file names from segment and intensity file paths
			std::filesystem::path pseg(r.segFname), pint(r.intFname);
			std::string segfname = pseg.filename().string(),
				intfname = pint.filename().string();

			stringColBuf.push_back(segfname);
			stringColBuf.push_back(intfname);
			resultBuf.push_back(l);
			totalNumFeatures = 1;
			for (auto& enabdF : F)
			{
				auto fc = std::get<1>(enabdF);
				auto fn = std::get<0>(enabdF);	// debug
				auto vv = r.getFeatureValues(fc);

				totalNumFeatures += vv.size();

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
						resultBuf.push_back(vv[i]);
					
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
					auto nAng = 4; // sizeof(GLRLM_features::rotAngles) / sizeof(GLRLM_features::rotAngles[0]);
					for (int i = 0; i < nAng; i++)
						resultBuf.push_back(vv[i]);
					
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					for (auto i = 0; i < GaborFeature::num_features; i++)
						resultBuf.push_back(vv[i]);

					// Proceed with other features
					continue;
				}

				// --Zernike family
				if (fc == ZERNIKE2D)
				{
					for (int i = 0; i < ZernikeFeature::num_feature_values_calculated; i++)
						resultBuf.push_back(vv[i]); 

					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
						resultBuf.push_back(vv[i]);
					
					// Proceed with other features
					continue;
				}
				if (fc == MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
						resultBuf.push_back(vv[i]);
					
					// Proceed with other features
					continue;
				}
				if (fc == RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
						resultBuf.push_back(vv[i]);
					
					// Proceed with other features
					continue;
				}

				// Regular feature
				resultBuf.push_back(vv[0]);
			}
		}

		return true;
	}

}