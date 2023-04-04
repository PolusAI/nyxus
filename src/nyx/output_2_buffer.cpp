#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
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
	/// @brief Copies ROIs' feature values into a ResultsCache structure that will then shape them as a table
	bool save_features_2_buffer (ResultsCache& rescache)
	{
		std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(L.begin(), L.end());
		std::vector<std::tuple<std::string, AvailableFeatures>> F = theFeatureSet.getEnabledFeatures();

		// We only fill in the header once.
		// We depend on the caller to manage headerBuf contents and clear it appropriately...
		bool fill_header = rescache.get_calcResultBuf().size() == 0;

		// -- Header
		if (fill_header)
		{
			rescache.add_to_header({"mask_image", "intensity_image", "label"});

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
					for (auto ang : theEnvironment.glcmAngles)
					{
						std::string col = fn + "_" + std::to_string(ang);
						rescache.add_to_header(col);	
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
						std::string col = fn + "_" + std::to_string(ang);
						rescache.add_to_header(col);	
					}
					
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					// Generate the feature value list
					for (auto i = 0; i < GaborFeature::f0.size(); i++)
					{
						std::string col = fn + "_" + std::to_string(i);
						rescache.add_to_header(col);	
					}

					// Proceed with other features
					continue;
				}

				if (fc == FRAC_AT_D)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					{
						std::string col = fn + "_" + std::to_string(i);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				if (fc == MEAN_FRAC)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					{
						std::string col = fn + "_" + std::to_string(i);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				if (fc == RADIAL_CV)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					{
						std::string col = fn + "_" + std::to_string(i);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				// --Zernike family
				if (fc == ZERNIKE2D)
				{
					// Populate with indices
					for (int i = 0; i < ZernikeFeature::num_feature_values_calculated; i++)
					{
						std::string col = fn + "_" + std::to_string(i);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				// Regular feature
				rescache.add_to_header(fn);	
			}
		}

		// -- Values
		for (auto l : L)
		{
			LR& r = roiData[l];

			// Skip blacklisted ROI
			if (r.blacklisted)
				continue;

			rescache.inc_num_rows();	

			// Tear off pure file names from segment and intensity file paths
			fs::path pseg(r.segFname), pint(r.intFname);
			std::string segfname = pseg.filename().string(),
				intfname = pint.filename().string();

			rescache.add_string(segfname);	
			rescache.add_string(intfname);	
			rescache.add_numeric(l); 
			for (auto& enabdF : F)
			{
				auto fc = std::get<1>(enabdF);
				auto fn = std::get<0>(enabdF);	// debug
				auto vv = r.get_fvals(fc);

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
					for (int i = 0; i < theEnvironment.glcmAngles.size(); i++)
						rescache.add_numeric(vv[i]);		
					
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
						rescache.add_numeric(vv[i]);		
					
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					for (auto i = 0; i < GaborFeature::f0.size(); i++)
						rescache.add_numeric(vv[i]);		

					// Proceed with other features
					continue;
				}

				// --Zernike family
				if (fc == ZERNIKE2D)
				{
					for (int i = 0; i < ZernikeFeature::num_feature_values_calculated; i++)
						rescache.add_numeric(vv[i]);		

					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
						rescache.add_numeric(vv[i]);		
					
					// Proceed with other features
					continue;
				}
				if (fc == MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
						rescache.add_numeric(vv[i]);		
					
					// Proceed with other features
					continue;
				}
				if (fc == RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
						rescache.add_numeric(vv[i]);		
					
					// Proceed with other features
					continue;
				}

				// Regular feature
				rescache.add_numeric(vv[0]);		
			}
		}

		return true;
	}

}
