#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include "constants.h"
#include "globals.h"
#include "environment.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/zernike.h"
#include "helpers/fsystem.h"

namespace Nyxus
{
	static std::mutex mx1;

	bool save_features_2_buffer_wholeslide (
		// out
		ResultsCache & rescache, 
		// in
		Environment & env,
		const LR & r,
		const std::string& ifpath,
		const std::string& mfpath)
	{
		std::lock_guard<std::mutex> lg (mx1);

		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		// We only fill in the header once.
		// We depend on the caller to manage headerBuf contents and clear it appropriately...
		bool fill_header = rescache.get_headerBuf().size() == 0;

		// -- Header

		if (fill_header)
		{
			rescache.add_to_header({ Nyxus::colname_intensity_image, Nyxus::colname_mask_image, Nyxus::colname_roi_label, Nyxus::colname_t_index });

			for (auto& enabdF : F)
			{
				auto fn = std::get<0>(enabdF);	// feature name
				auto fc = std::get<1>(enabdF);	// feature code

				// Handle missing feature name (which is a significant issue!) in order to at least be able to trace back to the feature code
				if (fn.empty())
					fn = "feature" + std::to_string(fc);

				// Parameterized feature
				// --GLCM family
				bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
				bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glcmFeature && nonAngledGlcmFeature == false)
				{
					// Populate with angles
					for (auto ang : env.glcmOptions.glcmAngles)
					{
						std::string col = fn + "_" + std::to_string(ang);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
				bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glrlmFeature && nonAngledGlrlmFeature == false)
				{
					// Populate with angles
					for (auto ang : GLRLMFeature::rotAngles)
					{
						std::string col = fn + "_" + std::to_string(ang);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == (int)Feature2D::GABOR)
				{
					// Generate the feature value list
					for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					{
						std::string col = fn + "_" + std::to_string(i);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				if (fc == (int)Feature2D::FRAC_AT_D)
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

				if (fc == (int)Feature2D::MEAN_FRAC)
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

				if (fc == (int)Feature2D::RADIAL_CV)
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
				if (fc == (int)Feature2D::ZERNIKE2D)
				{
					// Populate with indices
					for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					{
						std::string col = fn + "_Z" + std::to_string(i);
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

		rescache.inc_num_rows();

		// - slide info
		rescache.add_string (ifpath);
		rescache.add_string (mfpath);
		rescache.add_numeric (r.label);
		rescache.add_numeric (DEFAULT_T_INDEX);

		// - features
		for (auto& enabdF : F)
		{
			auto fc = std::get<1>(enabdF);
			auto fn = std::get<0>(enabdF);	// debug

			auto vv = r.get_fvals(fc);

			// Parameterized feature
			// --GLCM family
			bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
			bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
			if (glcmFeature && nonAngledGlcmFeature == false)
			{
				// Mock angled values if they haven't been calculated for some error reason
				if (vv.size() < GLCMFeature::angles.size())
					vv.resize(GLCMFeature::angles.size(), 0.0);

				// Populate with angles
				int nAng = GLCMFeature::angles.size();
				for (int i = 0; i < nAng; i++)
				{
					rescache.add_numeric(Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
				}

				// Proceed with other features
				continue;
			}

			// --GLRLM family
			bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
			bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
			if (glrlmFeature && nonAngledGlrlmFeature == false)
			{
				// Populate with angles
				int nAng = 4; // check GLRLMFeature::rotAngles
				for (int i = 0; i < nAng; i++)
				{
					rescache.add_numeric(Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
				}
				// Proceed with other features
				continue;
			}

			// --Gabor
			if (fc == (int)Feature2D::GABOR)
			{
				for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
				{
					rescache.add_numeric(Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
				}
				// Proceed with other features
				continue;
			}

			// --Zernike family
			if (fc == (int)Feature2D::ZERNIKE2D)
			{
				for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
				{
					rescache.add_numeric(Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
				}
				// Proceed with other features
				continue;
			}

			// --Radial distribution features
			if (fc == (int)Feature2D::FRAC_AT_D)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
				{
					rescache.add_numeric(Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
				}
				// Proceed with other features
				continue;
			}
			if (fc == (int)Feature2D::MEAN_FRAC)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
				{
					rescache.add_numeric(Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
				}
				// Proceed with other features
				continue;
			}
			if (fc == (int)Feature2D::RADIAL_CV)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
				{
					rescache.add_numeric(Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
				}
				// Proceed with other features
				continue;
			}

			// Regular feature
			rescache.add_numeric(Nyxus::force_finite_number(vv[0], env.resultOptions.noval()));
		}

		return true;
	}

	/// @brief Copies ROIs' feature values into a ResultsCache structure that will then shape them as a table
	bool save_features_2_buffer (ResultsCache& rescache, Environment & env, size_t t_index)
	{
		std::vector<int> L{ env.uniqueLabels.begin(), env.uniqueLabels.end() };
		std::sort(L.begin(), L.end());
		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		// We only fill in the header once.
		// We depend on the caller to manage headerBuf contents and clear it appropriately...
		bool fill_header = rescache.get_headerBuf().size() == 0;

		// -- Header
		if (fill_header)
		{
			rescache.add_to_header({ Nyxus::colname_intensity_image, Nyxus::colname_mask_image, Nyxus::colname_roi_label, Nyxus::colname_t_index });

			for (auto& enabdF : F)
			{
				auto fn = std::get<0>(enabdF);	// feature name
				auto fc = std::get<1>(enabdF);	// feature code

				// Handle missing feature name (which is a significant issue!) in order to at least be able to trace back to the feature code
				if (fn.empty())
					fn = "feature" + std::to_string(fc);

				// Parameterized feature
				// --GLCM family
				bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
				bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glcmFeature && nonAngledGlcmFeature == false)
				{
					// Populate with angles
					for (auto ang : env.glcmOptions.glcmAngles)
					{
						std::string col = fn + "_" + std::to_string(ang);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
				bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glrlmFeature && nonAngledGlrlmFeature == false)
				{
					// Populate with angles
					for (auto ang : GLRLMFeature::rotAngles)
					{
						std::string col = fn + "_" + std::to_string(ang);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == (int) Feature2D::GABOR)
				{
					// Generate the feature value list
					for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					{
						std::string col = fn + "_" + std::to_string(i);
						rescache.add_to_header(col);
					}

					// Proceed with other features
					continue;
				}

				if (fc == (int) Feature2D::FRAC_AT_D)
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

				if (fc == (int) Feature2D::MEAN_FRAC)
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

				if (fc == (int) Feature2D::RADIAL_CV)
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
				if (fc == (int) Feature2D::ZERNIKE2D)
				{
					// Populate with indices
					for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					{
						std::string col = fn + "_Z" + std::to_string(i);
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
			LR& r = env.roiData[l];

			// Skip blacklisted ROI
			if (r.blacklisted)
				continue;

			rescache.inc_num_rows();

			// Tear off pure file names from segment and intensity file paths
			const SlideProps & slide = env.dataset.dataset_props[r.slide_idx];
			fs::path pseg(slide.fname_seg), 
				pint(slide.fname_int);
			std::string segfname = pseg.filename().string(),
				intfname = pint.filename().string();

			rescache.add_string (intfname);
			rescache.add_string (segfname);
			rescache.add_numeric (l);
			rescache.add_numeric (t_index);

			for (auto& enabdF : F)
			{
				auto fc = std::get<1>(enabdF);
				auto fn = std::get<0>(enabdF);	// debug

				auto vv = r.get_fvals(fc);

				// Parameterized feature
				// --GLCM family
				bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
				bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glcmFeature && nonAngledGlcmFeature == false)
				{
					// Mock angled values if they haven't been calculated for some error reason
					if (vv.size() < GLCMFeature::angles.size())
						vv.resize(GLCMFeature::angles.size(), 0.0);

					// Populate with angles
					int nAng = GLCMFeature::angles.size();
					for (int i = 0; i < nAng; i++)
					{
						rescache.add_numeric (Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
					}

					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
				bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glrlmFeature && nonAngledGlrlmFeature == false)
				{
					// Populate with angles
					int nAng = 4; // check GLRLMFeature::rotAngles
					for (int i = 0; i < nAng; i++)
					{
						rescache.add_numeric (Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == (int) Feature2D::GABOR)
				{
					for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					{
						rescache.add_numeric (Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
					}
					// Proceed with other features
					continue;
				}

				// --Zernike family
				if (fc == (int) Feature2D::ZERNIKE2D)
				{
					for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					{
						rescache.add_numeric (Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
					}
					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == (int) Feature2D::FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					{
						rescache.add_numeric (Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int) Feature2D::MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					{
						rescache.add_numeric (Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int) Feature2D::RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					{
						rescache.add_numeric (Nyxus::force_finite_number(vv[i], env.resultOptions.noval()));
					}
					// Proceed with other features
					continue;
				}

				// Regular feature
				rescache.add_numeric (Nyxus::force_finite_number(vv[0], env.resultOptions.noval()));
			}
		}

		return true;
	}

}
