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
#include "environment.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/zernike.h"
#include "globals.h"
#include "helpers/fsystem.h"

namespace Nyxus
{
	// Macro to make some file i/o calls platform-independent
#ifndef _WIN32
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

	double auto_precision(std::stringstream& ss, double x)
	{
		if (std::abs(x) >= 1.0)
			ss << std::setprecision(theEnvironment.get_floating_point_precision());
		else
			if (x == 0.0)
				ss << std::setprecision(1);
			else
			{
				// int n = int(std::abs(std::log10(x)) + 0.5);

				double tmp1 = std::abs(x),
					tmp2 = std::log10(tmp1),
					tmp3 = std::abs(tmp2),
					tmp4 = tmp3 + 0.5;
				int n = int(tmp4);

				ss << std::setprecision(theEnvironment.get_floating_point_precision() + n);
			}

		return x;
	}

	std::vector<std::string> get_header(const std::vector<std::tuple<std::string, int>>& F) {
		std::stringstream ssHead;

		std::vector<std::string> head;

		// Mandatory column names
		for (const auto& s : mandatory_output_columns)
		{
			head.emplace_back(s);
		}

		// Optional columns
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
			// --GLCM family
			bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
			bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
			if (glcmFeature && nonAngledGlcmFeature == false)
			{
				// Populate with angles
				for (auto ang : theEnvironment.glcmOptions.glcmAngles)
				{
					// CSV separator
					//if (ang != theEnvironment.rotAngles[0])
					//	ssHead << ",";
					head.emplace_back(fn + "_" + std::to_string(ang));
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
					head.emplace_back(fn + "_" + std::to_string(ang));
				}
				// Proceed with other features
				continue;
			}

			// --Gabor
			if (fc == (int) Feature2D::GABOR)
			{
				// Generate the feature value list
				for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					head.emplace_back(fn + "_" + std::to_string(i));

				// Proceed with other features
				continue;
			}

			if (fc == (int) Feature2D::FRAC_AT_D)
			{
				// Generate the feature value list
				for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					head.emplace_back(fn + "_" + std::to_string(i));

				// Proceed with other features
				continue;
			}

			if (fc == (int) Feature2D::MEAN_FRAC)
			{
				// Generate the feature value list
				for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					head.emplace_back(fn + "_" + std::to_string(i));

				// Proceed with other features
				continue;
			}

			if (fc == (int) Feature2D::RADIAL_CV)
			{
				// Generate the feature value list
				for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					head.emplace_back(fn + "_" + std::to_string(i));

				// Proceed with other features
				continue;
			}

			// --Zernike features header 
			if (fc == (int) Feature2D::ZERNIKE2D)
			{
				// Populate with indices
				for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)	// i < ZernikeFeature::num_feature_values_calculated
					head.emplace_back(fn + "_Z" + std::to_string(i));

				// Proceed with other features
				continue;
			}

			// Regular feature
			head.emplace_back(fn);
		}

		return head;
	}

	std::string get_feature_output_fname(const std::string& intFpath, const std::string& segFpath)
	{
		std::string retval;
		if (theEnvironment.separateCsv)
		{
			retval = theEnvironment.output_dir + "/_INT_" + getPureFname(intFpath) + "_SEG_" + getPureFname(segFpath) + ".csv";
		}
		else
		{
			retval = theEnvironment.output_dir + "/" + theEnvironment.nyxus_result_fname + ".csv";
		}
		return retval;
	}

	const std::vector<std::string> mandatory_output_columns
	{
		Nyxus::colname_intensity_image, 
		Nyxus::colname_mask_image, 
		Nyxus::colname_roi_label 
	};

	static std::mutex mutex1;

	bool save_features_2_csv_wholeslide (
		const LR & r, 
		const std::string & ifpath, 
		const std::string & mfpath, 
		const std::string & outdir)
	{
		std::lock_guard<std::mutex> lg (mutex1); // Lock the mutex

		// Non-exotic formatting for compatibility with the buffer output (Python API, Apache)
		constexpr int VAL_BUF_LEN = 450;
		char rvbuf[VAL_BUF_LEN]; // real value buffer large enough to fit a float64 value in range (2.2250738585072014E-308 to 1.79769313486231570e+308)
		const char rvfmt[] = "%g"; // instead of "%20.12f" which produces a too massive output

		static bool mustRenderHeader = true;	// In 'singlecsv' scenario this flag flips from 'T' to 'F' when necessary (after the header is rendered)

		// Make the file name and write mode
		std::string fullPath = get_feature_output_fname (ifpath, mfpath);
		VERBOSLVL2(std::cout << "\t--> " << fullPath << "\n");

		// Single CSV: create or continue?
		const char* mode = "w";
		if (!theEnvironment.separateCsv)
			mode = mustRenderHeader ? "w" : "a";

		// Open it
		FILE* fp = nullptr;
		fopen_s(&fp, fullPath.c_str(), mode);

		if (!fp)
		{
			std::string errmsg = "Cannot open file " + fullPath + " for writing";
			std::perror(errmsg.c_str());
			return false;
		}

		// configure buffered write
		if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0) 
			std::perror("setvbuf failed");

		// Learn what features need to be displayed
		std::vector<std::tuple<std::string, int>> F = theFeatureSet.getEnabledFeatures();

		// ********** Header

		if (mustRenderHeader)
		{
			std::stringstream ssHead;

			auto head_vector = Nyxus::get_header(F);

			// Make sure that the header is in format "column1","column2",...,"columnN" without spaces
			for (int i = 0; i < head_vector.size(); i++)
			{
				const auto& column = head_vector[i];
				if (i)
					ssHead << ',';
				ssHead << '\"' << column << '\"';
			}

			auto head_string = ssHead.str();
			fprintf(fp, "%s\n", head_string.c_str());

			// Prevent rendering the header again for another image's portion of labels
			if (theEnvironment.separateCsv == false)
				mustRenderHeader = false;
		}

		// ********** Values

		if (! r.blacklisted)
		{
			std::stringstream ssVals;

			// Floating point precision
			ssVals << std::fixed;

			// slide info
			ssVals << ifpath << "," << mfpath << "," << r.label;

			// features
			for (auto& enabdF : F)
			{
				auto fc = std::get<1>(enabdF);
				auto fn = std::get<0>(enabdF);	// feature name, for debugging
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
					// Output the sub-values
					int nAng = (int)GLCMFeature::angles.size();
					for (int i = 0; i < nAng; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
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
					int nAng = 4;
					for (int i = 0; i < nAng; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == (int)Feature2D::GABOR)
				{
					for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
					}

					// Proceed with other features
					continue;
				}

				// --Zernike feature values
				if (fc == (int)Feature2D::ZERNIKE2D)
				{
					for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
					}

					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == (int)Feature2D::FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int)Feature2D::MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int)Feature2D::RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf (rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
					}
					// Proceed with other features
					continue;
				}

				// Regular feature
				snprintf (rvbuf, VAL_BUF_LEN, rvfmt, Nyxus::force_finite_number(vv[0], theEnvironment.nan_substitute));
				ssVals << "," << rvbuf; // Alternatively: auto_precision(ssVals, vv[0]);
			}

			fprintf(fp, "%s\n", ssVals.str().c_str());
		}

		std::fflush(fp);
		std::fclose(fp);

		return true;
	}

	// Saves the result of image scanning and feature calculation. Must be called after the reduction phase.
	bool save_features_2_csv (const std::string& intFpath, const std::string& segFpath, const std::string& outputDir)
	{
		// Non-exotic formatting for compatibility with the buffer output (Python API, Apache)
		constexpr int VAL_BUF_LEN = 450;
		char rvbuf[VAL_BUF_LEN]; // real value buffer large enough to fit a float64 value in range (2.2250738585072014E-308 to 1.79769313486231570e+308)
		const char rvfmt[] = "%g"; // instead of "%20.12f" which produces a too massive output

		// Sort the labels
		std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(L.begin(), L.end());

		static bool mustRenderHeader = true;	// In 'singlecsv' scenario this flag flips from 'T' to 'F' when necessary (after the header is rendered)

		// Make the file name and write mode
		std::string fullPath = get_feature_output_fname(intFpath, segFpath);
		VERBOSLVL2(std::cout << "\t--> " << fullPath << "\n");

		// Single CSV: create or continue?
		const char* mode = "w";
		if (!theEnvironment.separateCsv)
			mode = mustRenderHeader ? "w" : "a";

		// Open it
		FILE* fp = nullptr;
		fopen_s(&fp, fullPath.c_str(), mode);

		if (!fp)
		{
			std::string errmsg = "Cannot open file " + fullPath + " for writing";
			std::perror(errmsg.c_str());
			return false;
		}

		// -- Configure buffered write
		if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0) {
			std::perror("setvbuf failed");
			return false;
		}

		// Learn what features need to be displayed
		std::vector<std::tuple<std::string, int>> F = theFeatureSet.getEnabledFeatures();

		// -- Header
		if (mustRenderHeader)
		{
			std::stringstream ssHead;

			auto head_vector = Nyxus::get_header(F);

			// Make sure that the header is in format "column1","column2",...,"columnN" without spaces
			for (int i = 0; i < head_vector.size(); i++)
			{
				const auto& column = head_vector[i];
				if (i)
					ssHead << ',';
				ssHead << '\"' << column << '\"';
			}

			auto head_string = ssHead.str();
			fprintf(fp, "%s\n", head_string.c_str());

			// Prevent rendering the header again for another image's portion of labels
			if (theEnvironment.separateCsv == false)
				mustRenderHeader = false;
		}

		// -- Values
		for (auto l : L)
		{
			LR& r = roiData[l];

			// Skip blacklisted ROI
			if (r.blacklisted)
				continue;

			std::stringstream ssVals;

			// Floating point precision
			ssVals << std::fixed;

			// Tear off pure file names from segment and intensity file paths
			fs::path pseg(r.segFname), pint(r.intFname);
			ssVals << pint.filename() << "," << pseg.filename() << "," << l;

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
					// Output the sub-values
					int nAng = (int) GLCMFeature::angles.size();
					for (int i = 0; i < nAng; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
#ifndef DIAGNOSE_NYXUS_OUTPUT
						ssVals << "," << rvbuf;
#else
						//--diagnoze misalignment-- 
						ssVals << "," << fn << "-" << rvbuf;
#endif
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
					int nAng = 4;
					for (int i = 0; i < nAng; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
#ifndef DIAGNOSE_NYXUS_OUTPUT
						ssVals << "," << rvbuf;
#else
						//--diagnoze misalignment-- 
						ssVals << "," << fn << "-" << rvbuf;
#endif
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == (int) Feature2D::GABOR)
				{
					for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
#ifndef DIAGNOSE_NYXUS_OUTPUT
						ssVals << "," << rvbuf;
#else	
						//--diagnoze misalignment-- 
						ssVals << "," << fn << "-" << rvbuf;
#endif			
					}

					// Proceed with other features
					continue;
				}

				// --Zernike feature values
				if (fc == (int) Feature2D::ZERNIKE2D)
				{
					for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
#ifndef DIAGNOSE_NYXUS_OUTPUT
						ssVals << "," << rvbuf;
#else
						//--diagnoze misalignment-- 
						ssVals << "," << fn << "-" << rvbuf;
#endif
					}

					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == (int) Feature2D::FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
#ifndef DIAGNOSE_NYXUS_OUTPUT
						ssVals << "," << rvbuf;
#else
						//--diagnoze misalignment-- 
						ssVals << "," << fn << "-" << rvbuf;
#endif
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int) Feature2D::MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
#ifndef DIAGNOSE_NYXUS_OUTPUT
						ssVals << "," << rvbuf;
#else
						//--diagnoze misalignment-- 
						ssVals << "," << fn << "-" << rvbuf;
#endif
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int) Feature2D::RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					{
						double fv = Nyxus::force_finite_number(vv[i], theEnvironment.nan_substitute);	// safe feature value (no NAN, no inf)
						snprintf(rvbuf, VAL_BUF_LEN, rvfmt, fv);
#ifndef DIAGNOSE_NYXUS_OUTPUT
						ssVals << "," << rvbuf;
#else
						//--diagnoze misalignment-- 
						ssVals << "," << fn << "-" << rvbuf;
#endif
					}
					// Proceed with other features
					continue;
				}

				// Regular feature
				snprintf(rvbuf, VAL_BUF_LEN, rvfmt, Nyxus::force_finite_number(vv[0], theEnvironment.nan_substitute));
#ifndef DIAGNOSE_NYXUS_OUTPUT
				ssVals << "," << rvbuf; // Alternatively: auto_precision(ssVals, vv[0]);
#else
				//--diagnoze misalignment-- 
				ssVals << "," << fn << "-" << rvbuf;
#endif
			}

			fprintf(fp, "%s\n", ssVals.str().c_str());
		}

		std::fflush(fp);
		std::fclose(fp);

		return true;
	}

	std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>> get_feature_values_roi (
		const LR& r,
		const std::string& ifpath,
		const std::string& mfpath)
	{
		std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>> features;

		// user's feature selection
		std::vector<std::tuple<std::string, int>> F = theFeatureSet.getEnabledFeatures();

		// numeric columns
		std::vector<double> fvals;

		for (auto& enabdF : F)
		{
			auto fc = std::get<1>(enabdF);
			auto vv = r.get_fvals(std::get<1>(enabdF));

			// Parameterized feature
			// --GLCM family
			bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
			bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
			if (glcmFeature && nonAngledGlcmFeature == false)
			{
				// Mock angled values if they haven't been calculated for some error reason
				if (vv.size() < GLCMFeature::angles.size())
					vv.resize(GLCMFeature::angles.size(), 0.0);
				// Output the sub-values
				int nAng = (int)GLCMFeature::angles.size();
				for (int i = 0; i < nAng; i++)
				{
					fvals.push_back(vv[i]);
				}
				// Proceed with other features
				continue;
			}

			// --GLRLM family
			bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
			bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
			if (glrlmFeature && nonAngledGlrlmFeature == false)
			{
				// Polulate with angles
				int nAng = 4;
				for (int i = 0; i < nAng; i++)
				{
					fvals.push_back(vv[i]);
				}
				// Proceed with other features
				continue;
			}

			// --Gabor
			if (fc == (int)Feature2D::GABOR)
			{
				for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
				{
					fvals.push_back(vv[i]);
				}

				// Proceed with other features
				continue;
			}

			// --Zernike feature values
			if (fc == (int)Feature2D::ZERNIKE2D)
			{
				for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
				{
					fvals.push_back(vv[i]);
				}

				// Proceed with other features
				continue;
			}

			// --Radial distribution features
			if (fc == (int)Feature2D::FRAC_AT_D)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
				{
					fvals.push_back(vv[i]);
				}
				// Proceed with other features
				continue;
			}
			if (fc == (int)Feature2D::MEAN_FRAC)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
				{
					fvals.push_back(vv[i]);
				}
				// Proceed with other features
				continue;
			}
			if (fc == (int)Feature2D::RADIAL_CV)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
				{
					fvals.push_back(vv[i]);
				}
				// Proceed with other features
				continue;
			}

			fvals.push_back(vv[0]);
		}

		// other columns
		std::vector<std::string> textcols;
		textcols.push_back ((fs::path(ifpath)).filename().u8string());
		textcols.push_back ("");
		int roilabl = 1; // whole-slide roi #

		features.push_back (std::make_tuple(textcols, roilabl, fvals));

		return features;
	}

	std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>> get_feature_values() 
	{
		std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>> features;

		// Sort the labels
		std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(L.begin(), L.end());

		// Learn what features need to be displayed
		std::vector<std::tuple<std::string, int>> F = theFeatureSet.getEnabledFeatures();

		// -- Values
		for (auto l : L)
		{
			LR& r = roiData[l];

			std::vector<double> feature_values;

			// Skip blacklisted ROI
			if (r.blacklisted)
				continue;

			// Tear off pure file names from segment and intensity file paths
			fs::path pseg(r.segFname), pint(r.intFname);
			std::vector<std::string> filenames;
			filenames.push_back(pint.filename().u8string());
			filenames.push_back(pseg.filename().u8string());

			for (auto& enabdF : F)
			{
				auto fc = std::get<1>(enabdF);
				auto vv = r.get_fvals(std::get<1>(enabdF));

				// Parameterized feature
				// --GLCM family
				bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
				bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glcmFeature && nonAngledGlcmFeature == false)
				{
					// Mock angled values if they haven't been calculated for some error reason
					if (vv.size() < GLCMFeature::angles.size())
						vv.resize(GLCMFeature::angles.size(), 0.0);
					// Output the sub-values
					int nAng = (int) GLCMFeature::angles.size();
					for (int i = 0; i < nAng; i++)
					{
						feature_values.push_back(vv[i]);
					}
					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
				bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end(); // prevent output of a non-angled feature in an angled way
				if (glrlmFeature && nonAngledGlrlmFeature == false)
				{
					// Polulate with angles
					int nAng = 4;
					for (int i = 0; i < nAng; i++)
					{
						feature_values.push_back(vv[i]);
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == (int) Feature2D::GABOR)
				{
					for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					{
						feature_values.push_back(vv[i]);
					}

					// Proceed with other features
					continue;
				}

				// --Zernike feature values
				if (fc == (int) Feature2D::ZERNIKE2D)
				{
					for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					{
						feature_values.push_back(vv[i]);
					}

					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == (int) Feature2D::FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					{
						feature_values.push_back(vv[i]);
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int) Feature2D::MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					{
						feature_values.push_back(vv[i]);
					}
					// Proceed with other features
					continue;
				}
				if (fc == (int) Feature2D::RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					{
						feature_values.push_back(vv[i]);
					}
					// Proceed with other features
					continue;
				}

				feature_values.push_back(vv[0]);
			}

			features.push_back(std::make_tuple(filenames, l, feature_values));
		}

		return features;
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
