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

	double auto_precision (Environment & env, std::stringstream& ss, double x)
	{
		if (std::abs(x) >= 1.0)
			ss << std::setprecision (env.get_floating_point_precision());
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

				ss << std::setprecision (env.get_floating_point_precision() + n);
			}

		return x;
	}

	std::vector<std::string> get_header (Environment & env) 
	{
		// user's feature selection
		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		std::stringstream ssHead;

		std::vector<std::string> head;

		// Mandatory column names
		head.clear();
		head.push_back (Nyxus::colname_intensity_image);
		head.push_back (Nyxus::colname_mask_image);

		// Annotation columns
		if (env.resultOptions.need_annotation())
		{
			auto slp = env.dataset.dataset_props[0];
			for (auto i = 0; i < slp.annots.size(); i++)
			{
				std::string colnm = "anno" + std::to_string(i);
				head.push_back (colnm);
			}
		}

		// ROI label
		head.push_back (Nyxus::colname_roi_label);

		// time
		head.push_back (Nyxus::colname_t_index);

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
				for (auto ang : env.glcmOptions.glcmAngles)
				{
					// CSV separator
					//if (ang != env.rotAngles[0])
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

	std::string get_feature_output_fname (Environment& env, const std::string& intFpath, const std::string& segFpath)
	{
		std::string retval;
		if (env.separateCsv)
			retval = env.output_dir + "/_INT_" + getPureFname(intFpath) + "_SEG_" + getPureFname(segFpath) + ".csv";
		else
			retval = env.output_dir + "/" + env.nyxus_result_fname + ".csv";
		return retval;
	}

	static std::mutex mutex1;

	bool save_features_2_csv_wholeslide (
		Environment & env,
		const LR & r, 
		const std::string & ifpath, 
		const std::string & mfpath, 
		const std::string & outdir,
		size_t t_index)
	{
		std::lock_guard<std::mutex> lg (mutex1); // Lock the mutex

		// Non-exotic formatting for compatibility with the buffer output (Python API, Apache)
		constexpr int VAL_BUF_LEN = 450;
		char rvbuf[VAL_BUF_LEN]; // real value buffer large enough to fit a float64 value in range (2.2250738585072014E-308 to 1.79769313486231570e+308)
		const char rvfmt[] = "%g"; // instead of "%20.12f" which produces a too massive output

		static bool mustRenderHeader = true;	// In 'singlecsv' scenario this flag flips from 'T' to 'F' when necessary (after the header is rendered)

		// Make the file name and write mode
		std::string fullPath = get_feature_output_fname (env, ifpath, mfpath);
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\t--> " << fullPath << "\n");

		// Single CSV: create or continue?
		const char* mode = "w";
		if (!env.separateCsv)
			mode = mustRenderHeader ? "w" : "a";

		// Open it
		FILE* fp = nullptr;
		auto /*assuming errno_t*/ eno = fopen_s(&fp, fullPath.c_str(), mode);

		if (!fp)
		{
			std::string errmsg = "Cannot open file " + fullPath + " for writing. Errno=" + std::to_string(eno);
			std::cerr << errmsg << "\n";
			return false;
		}

		// configure buffered write
		if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0) 
			std::cerr << "setvbuf failed \n";

		// Learn what features need to be displayed
		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		// ********** Header

		if (mustRenderHeader)
		{
			std::stringstream ssHead;

			auto head_vector = Nyxus::get_header (env);

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
			if (env.separateCsv == false)
				mustRenderHeader = false;
		}

		// ********** Values

		if (! r.blacklisted)
		{
			std::stringstream ssVals;

			// Floating point precision
			ssVals << std::fixed;

			// slide info
			ssVals << ifpath << "," << mfpath;
			
			// ROI and time
			ssVals << "," << r.label << "," << t_index;

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
						double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
						double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
						double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
						double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
						double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
						double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
						double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
						snprintf (rvbuf, VAL_BUF_LEN, rvfmt, fv);
						ssVals << "," << rvbuf;
					}
					// Proceed with other features
					continue;
				}

				// Regular feature
				snprintf (rvbuf, VAL_BUF_LEN, rvfmt, Nyxus::force_finite_number(vv[0], env.resultOptions.noval()));
				ssVals << "," << rvbuf; // Alternatively: auto_precision(ssVals, vv[0]);
			}

			fprintf(fp, "%s\n", ssVals.str().c_str());
		}

		std::fflush(fp);
		std::fclose(fp);

		return true;
	}

	// Saves the result of image scanning and feature calculation. Must be called after the reduction phase.
	bool save_features_2_csv (
		Environment & env,
		const std::string& intFpath, 
		const std::string& segFpath, 
		const std::string& outputDir,
		size_t t_index,
		bool need_aggregation)
	{
		// Non-exotic formatting for compatibility with the buffer output (Python API, Apache)
		constexpr int VAL_BUF_LEN = 450;
		char rvbuf[VAL_BUF_LEN]; // real value buffer large enough to fit a float64 value in range (2.2250738585072014E-308 to 1.79769313486231570e+308)
		const char rvfmt[] = "%g"; // instead of "%20.12f" which produces a too massive output

		// Sort the labels
		std::vector<int> L{ env.uniqueLabels.begin(), env.uniqueLabels.end() };
		std::sort(L.begin(), L.end());

		static bool mustRenderHeader = true;	// In 'singlecsv' scenario this flag flips from 'T' to 'F' when necessary (after the header is rendered)

		// Make the file name and write mode
		std::string fullPath = get_feature_output_fname (env, intFpath, segFpath);
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\t--> " << fullPath << "\n");

		// Single CSV: create or continue?
		const char* mode = "w";
		if (!env.separateCsv)
			mode = mustRenderHeader ? "w" : "a";

		// Open it
		FILE* fp = nullptr;
		auto /*assuming errno_t*/ eno = fopen_s (&fp, fullPath.c_str(), mode);

		if (!fp)
		{
			std::string errmsg = "Cannot open file " + fullPath + " for writing. Errno=" + std::to_string(eno);
			std::cerr << errmsg << "\n";
			return false;
		}

		// -- Configure buffered write
		if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0) {
			std::cerr << "setvbuf failed \n";
			return false;
		}

		// Learn what features need to be displayed
		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		// -- Header
		if (mustRenderHeader)
		{
			std::stringstream ssHead;

			auto head_vector = Nyxus::get_header (env);

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
			if (env.separateCsv == false)
				mustRenderHeader = false;
		}

		if (need_aggregation)
		{
			auto allres = Nyxus::get_feature_values (env.theFeatureSet, env.uniqueLabels, env.roiData, env.dataset);	// shape: td::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>
			if (allres.size())
			{
				// aggregate
				const auto& tup0 = allres[0];
				const auto& v0 = std::get<3>(tup0);
				auto n_feats = v0.size();

				std::vector<double> a (n_feats);
				std::fill(a.begin(), a.end(), 0.0);

				double n_rois = (double) allres.size();
				for (const auto& tup : allres)
				{
					// we ned to add Skipping blacklisted ROI 
					// ...

					const auto& v = std::get<3>(tup);
					for (size_t i = 0; i < n_feats; i++)
					{
						double x = v[i],
							ai = x / n_rois;
						
						// handle likely NAN
						if (ai != ai)
							ai = env.resultOptions.noval();

						a[i] += ai;
					}
				}

				// send the aggregate as ROI #-1 to output 
				const std::vector<std::string> & fnames = std::get<0>(tup0);
				std::stringstream ssVals;
				// ... slide/mask file info
				ssVals << fnames[0] << "," << fnames[1];
				// ... annotation info
				auto lab0 = std::get<1>(tup0);	// annotation info is per slide, so OK to grab it from the 1st ROI
				LR& r0 = env.roiData [lab0];
				auto slp = env.dataset.dataset_props [r0.slide_idx];
				for (const auto& a : slp.annots)
					ssVals << "," << a;
				// ... ROI id
				ssVals << "," << -1;
				// ... aggregated feature values
				for (size_t i = 0; i < n_feats; i++)
					ssVals << "," << a[i];
				fprintf (fp, "%s\n", ssVals.str().c_str());
			}
		}
		else // no aggregation
		{
			// -- Values
			for (auto l : L)
			{
				LR& r = env.roiData[l];

				// Skip blacklisted ROI
				if (r.blacklisted)
					continue;

				std::stringstream ssVals;

				// Floating point precision
				ssVals << std::fixed;

				// Tear off pure file names from segment and intensity file paths
				const SlideProps& sli = env.dataset.dataset_props [r.slide_idx];
				fs::path pseg (sli.fname_seg), 
					pint (sli.fname_int);
				ssVals << pint.filename() << "," << pseg.filename();

				// annotation
				if (env.resultOptions.need_annotation())
				{
					for (const auto & a: sli.annots)
						ssVals << "," << a;
				}

				// ROI label and time
				ssVals << "," << l << "," << t_index;

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
							double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
							double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
							double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
							double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
							double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
							double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
							double fv = Nyxus::force_finite_number(vv[i], env.resultOptions.noval());	// safe feature value (no NAN, no inf)
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
					snprintf(rvbuf, VAL_BUF_LEN, rvfmt, Nyxus::force_finite_number(vv[0], env.resultOptions.noval()));
	#ifndef DIAGNOSE_NYXUS_OUTPUT
					ssVals << "," << rvbuf; // Alternatively: auto_precision(ssVals, vv[0]);
	#else
					//--diagnoze misalignment-- 
					ssVals << "," << fn << "-" << rvbuf;
	#endif
				}

				fprintf(fp, "%s\n", ssVals.str().c_str());
			}
		} //- no aggregation

		std::fflush(fp);
		std::fclose(fp);

		return true;
	}

	std::vector<FTABLE_RECORD> get_feature_values_roi (
		const FeatureSet & fset,
		const LR & r,
		const std::string & ifpath,
		const std::string & mfpath)
	{
		std::vector<FTABLE_RECORD> features;

		// user's feature selection
		std::vector<std::tuple<std::string, int>> F = fset.getEnabledFeatures();

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
		textcols.push_back ((fs::path(ifpath)).filename().string());
		textcols.push_back ("");
		int roilabl = r.label; // whole-slide roi #

		features.push_back (std::make_tuple(textcols, roilabl, -999.88, fvals));

		return features;
	}

	std::vector<FTABLE_RECORD> get_feature_values (
		const FeatureSet & fset, 
		const Uniqueids & uniqueLabels, 
		const Roidata & roiData,
		const Dataset & dataset)
	{
		std::vector<FTABLE_RECORD> features;

		// Sort the labels
		std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(L.begin(), L.end());

		// Learn what features need to be displayed
		std::vector<std::tuple<std::string, int>> F = fset.getEnabledFeatures();

		// -- Values
		for (const auto l : L)
		{
			const LR& r = roiData.at (l);

			std::vector<double> feature_values;

			// Skip blacklisted ROI
			if (r.blacklisted)
				continue;

			// Tear off pure file names from segment and intensity file paths
			const SlideProps& sli = dataset.dataset_props [r.slide_idx];
			fs::path pseg (sli.fname_seg), 
				pint (sli.fname_int);
			std::vector<std::string> filenames;
			filenames.push_back(pint.filename().string());
			filenames.push_back(pseg.filename().string());

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

			features.push_back (std::make_tuple(filenames, l, DEFAULT_T_INDEX, feature_values));
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
