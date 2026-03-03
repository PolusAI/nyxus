#include <atomic>
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

	/// @brief Returns feature column names for all enabled features.
	/// Single source of truth for header generation across CSV, buffer, and Arrow output paths.
	std::vector<std::string> get_feature_column_names (
		Environment & env,
		const std::vector<std::tuple<std::string, int>> & F)
	{
		std::vector<std::string> cols;

		for (const auto& enabdF : F)
		{
			auto fn = std::get<0>(enabdF);
			auto fc = std::get<1>(enabdF);

			if (fn.empty())
				fn = "feature" + std::to_string(fc);

			// GLCM family
			bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
			bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end();
			if (glcmFeature && nonAngledGlcmFeature == false)
			{
				for (auto ang : env.glcmOptions.glcmAngles)
					cols.push_back(fn + "_" + std::to_string(ang));
				continue;
			}

			// GLRLM family
			bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
			bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end();
			if (glrlmFeature && nonAngledGlrlmFeature == false)
			{
				for (auto ang : GLRLMFeature::rotAngles)
					cols.push_back(fn + "_" + std::to_string(ang));
				continue;
			}

			// Gabor
			if (fc == (int)Feature2D::GABOR)
			{
				for (auto i = 0; i < GaborFeature::f0_theta_pairs.size(); i++)
					cols.push_back(fn + "_" + std::to_string(i));
				continue;
			}

			if (fc == (int)Feature2D::FRAC_AT_D)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					cols.push_back(fn + "_" + std::to_string(i));
				continue;
			}

			if (fc == (int)Feature2D::MEAN_FRAC)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					cols.push_back(fn + "_" + std::to_string(i));
				continue;
			}

			if (fc == (int)Feature2D::RADIAL_CV)
			{
				for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					cols.push_back(fn + "_" + std::to_string(i));
				continue;
			}

			// Zernike
			if (fc == (int)Feature2D::ZERNIKE2D)
			{
				for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					cols.push_back(fn + "_Z" + std::to_string(i));
				continue;
			}

			// Regular feature
			cols.push_back(fn);
		}

		return cols;
	}

	/// @brief Collects raw feature values for a single ROI into a flat vector.
	/// Single source of truth for value collection across CSV, buffer, and Python output paths.
	/// Callers apply force_finite_number or other formatting as needed.
	std::vector<double> collect_feature_values (
		const LR & r,
		const std::vector<std::tuple<std::string, int>> & F)
	{
		std::vector<double> vals;

		for (const auto& enabdF : F)
		{
			auto fc = std::get<1>(enabdF);
			auto vv = r.get_fvals(fc);

			// GLCM family
			bool glcmFeature = std::find(GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), (Feature2D)fc) != GLCMFeature::featureset.end();
			bool nonAngledGlcmFeature = std::find(GLCMFeature::nonAngledFeatures.begin(), GLCMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLCMFeature::nonAngledFeatures.end();
			if (glcmFeature && nonAngledGlcmFeature == false)
			{
				if (vv.size() < GLCMFeature::angles.size())
					vv.resize(GLCMFeature::angles.size(), 0.0);
				int nAng = (int)GLCMFeature::angles.size();
				for (int i = 0; i < nAng; i++)
					vals.push_back(vv[i]);
				continue;
			}

			// GLRLM family
			bool glrlmFeature = std::find(GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), (Feature2D)fc) != GLRLMFeature::featureset.end();
			bool nonAngledGlrlmFeature = std::find(GLRLMFeature::nonAngledFeatures.begin(), GLRLMFeature::nonAngledFeatures.end(), (Feature2D)fc) != GLRLMFeature::nonAngledFeatures.end();
			if (glrlmFeature && nonAngledGlrlmFeature == false)
			{
				int nAng = (int)std::size(GLRLMFeature::rotAngles);
				if ((int)vv.size() < nAng)
					vv.resize(nAng, 0.0);
				for (int i = 0; i < nAng; i++)
					vals.push_back(vv[i]);
				continue;
			}

			// Gabor
			if (fc == (int)Feature2D::GABOR)
			{
				auto nGabor = GaborFeature::f0_theta_pairs.size();
				if (vv.size() < nGabor)
					vv.resize(nGabor, 0.0);
				for (size_t i = 0; i < nGabor; i++)
					vals.push_back(vv[i]);
				continue;
			}

			// Zernike
			if (fc == (int)Feature2D::ZERNIKE2D)
			{
				if ((int)vv.size() < ZernikeFeature::NUM_FEATURE_VALS)
					vv.resize(ZernikeFeature::NUM_FEATURE_VALS, 0.0);
				for (int i = 0; i < ZernikeFeature::NUM_FEATURE_VALS; i++)
					vals.push_back(vv[i]);
				continue;
			}

			// Radial distribution
			if (fc == (int)Feature2D::FRAC_AT_D)
			{
				if ((int)vv.size() < RadialDistributionFeature::num_features_FracAtD)
					vv.resize(RadialDistributionFeature::num_features_FracAtD, 0.0);
				for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					vals.push_back(vv[i]);
				continue;
			}
			if (fc == (int)Feature2D::MEAN_FRAC)
			{
				if ((int)vv.size() < RadialDistributionFeature::num_features_MeanFrac)
					vv.resize(RadialDistributionFeature::num_features_MeanFrac, 0.0);
				for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					vals.push_back(vv[i]);
				continue;
			}
			if (fc == (int)Feature2D::RADIAL_CV)
			{
				if ((int)vv.size() < RadialDistributionFeature::num_features_RadialCV)
					vv.resize(RadialDistributionFeature::num_features_RadialCV, 0.0);
				for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					vals.push_back(vv[i]);
				continue;
			}

			// Regular feature
			vals.push_back(vv.empty() ? 0.0 : vv[0]);
		}

		return vals;
	}

	/// @brief Writes all feature values for a single ROI to a stringstream in CSV format.
	/// Uses collect_feature_values() as the single source of truth for value dispatch.
	void write_feature_values_csv (
		std::stringstream & ssVals,
		const LR & r,
		const std::vector<std::tuple<std::string, int>> & F,
		Environment & env,
		char* rvbuf,
		int rvbuf_len,
		const char* rvfmt)
	{
		auto vals = collect_feature_values(r, F);

#ifdef DIAGNOSE_NYXUS_OUTPUT
		auto names = get_feature_column_names(env, F);
		for (size_t i = 0; i < vals.size(); i++)
		{
			double fv = Nyxus::force_finite_number(vals[i], env.resultOptions.noval());
			snprintf(rvbuf, rvbuf_len, rvfmt, fv);
			ssVals << "," << names[i] << "-" << rvbuf;
		}
#else
		for (auto v : vals)
		{
			double fv = Nyxus::force_finite_number(v, env.resultOptions.noval());
			snprintf(rvbuf, rvbuf_len, rvfmt, fv);
			ssVals << "," << rvbuf;
		}
#endif
	}

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

		// Feature columns (single source of truth)
		auto feature_cols = get_feature_column_names(env, F);
		for (const auto& c : feature_cols)
			head.push_back(c);

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

	// Namespace-level header flags (resettable between workflow invocations).
	// std::atomic ensures safe reset-before-use even without mutex protection.
	static std::atomic<bool> mustRenderHeader_wholeslide{true};
	static std::atomic<bool> mustRenderHeader_segmented{true};
	static std::atomic<bool> mustRenderHeader_fmaps{true};

	void reset_csv_header_state()
	{
		mustRenderHeader_wholeslide.store(true);
		mustRenderHeader_segmented.store(true);
		mustRenderHeader_fmaps.store(true);
	}

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

		// Make the file name and write mode
		std::string fullPath = get_feature_output_fname (env, ifpath, mfpath);
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\t--> " << fullPath << "\n");

		// Single CSV: create or continue?
		const char* mode = "w";
		if (!env.separateCsv)
			mode = mustRenderHeader_wholeslide ? "w" : "a";

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

		if (mustRenderHeader_wholeslide)
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
				mustRenderHeader_wholeslide = false;
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
			write_feature_values_csv(ssVals, r, F, env, rvbuf, VAL_BUF_LEN, rvfmt);

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

		// Make the file name and write mode
		std::string fullPath = get_feature_output_fname (env, intFpath, segFpath);
		VERBOSLVL2 (env.get_verbosity_level(), std::cout << "\t--> " << fullPath << "\n");

		// Single CSV: create or continue?
		const char* mode = "w";
		if (!env.separateCsv)
			mode = mustRenderHeader_segmented ? "w" : "a";

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
		if (mustRenderHeader_segmented)
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
				mustRenderHeader_segmented = false;
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

				write_feature_values_csv(ssVals, r, F, env, rvbuf, VAL_BUF_LEN, rvfmt);

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

		std::vector<std::tuple<std::string, int>> F = fset.getEnabledFeatures();
		auto fvals = collect_feature_values(r, F);

		std::vector<std::string> textcols;
		textcols.push_back ((fs::path(ifpath)).filename().string());
		textcols.push_back ("");
		int roilabl = r.label;

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

			auto feature_values = collect_feature_values(r, F);

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

	bool save_features_2_csv_fmaps (
		Environment & env,
		const std::string & intFpath,
		const std::string & segFpath,
		const std::string & outputDir,
		int slide_idx,
		const std::unordered_set<int> & childLabels,
		const std::unordered_map<int, LR> & childRoiData,
		const std::unordered_map<int, FmapChildInfo> & childToParentMap)
	{
		std::lock_guard<std::mutex> lg (mutex1);

		constexpr int VAL_BUF_LEN = 450;
		char rvbuf[VAL_BUF_LEN];
		const char rvfmt[] = "%g";

		// Sort child labels
		std::vector<int> L(childLabels.begin(), childLabels.end());
		std::sort(L.begin(), L.end());

		if (L.empty())
			return true;

		// Output file — fmaps results are written to a separate CSV with "_fmaps.csv" suffix
		// to keep them distinct from standard per-ROI feature results
		std::string fullPath;
		if (env.separateCsv)
			fullPath = outputDir + "/_INT_" + getPureFname(intFpath) + "_SEG_" + getPureFname(segFpath) + "_fmaps.csv";
		else
			fullPath = outputDir + "/" + env.nyxus_result_fname + "_fmaps.csv";

		const char* mode = "w";
		if (!env.separateCsv)
			mode = mustRenderHeader_fmaps ? "w" : "a";

		FILE* fp = nullptr;
		auto eno = fopen_s(&fp, fullPath.c_str(), mode);
		if (!fp)
		{
			std::cerr << "Cannot open file " << fullPath << " for writing. Errno=" << eno << "\n";
			return false;
		}

		if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0)
			std::cerr << "setvbuf failed \n";

		std::vector<std::tuple<std::string, int>> F = env.theFeatureSet.getEnabledFeatures();

		// Header
		if (mustRenderHeader_fmaps)
		{
			std::stringstream ssHead;

			auto head_vector = Nyxus::get_header(env);

			// Build fmaps header: insert fmaps columns after mask_image + annotations, before ROI_label
			// get_header returns: [intensity_image, mask_image, <annots...>, ROI_label, t_index, <features...>]
			// We want: [intensity_image, mask_image, <annots...>, parent_roi_label, kernel_center_x, kernel_center_y, ROI_label, t_index, <features...>]
			std::vector<std::string> fmaps_head;
			fmaps_head.push_back(head_vector[0]); // intensity_image
			fmaps_head.push_back(head_vector[1]); // mask_image

			// Annotation columns (if any)
			int n_annot_cols = 0;
			if (env.resultOptions.need_annotation())
			{
				auto slp = env.dataset.dataset_props[0];
				n_annot_cols = (int) slp.annots.size();
			}
			int roi_label_pos = 2 + n_annot_cols; // position of ROI_label in head_vector
			for (int i = 2; i < roi_label_pos; i++)
				fmaps_head.push_back(head_vector[i]);

			// Fmaps-specific columns
			fmaps_head.push_back("parent_roi_label");
			fmaps_head.push_back("kernel_center_x");
			fmaps_head.push_back("kernel_center_y");

			// Remaining standard columns (ROI_label, t_index, features...)
			for (size_t i = roi_label_pos; i < head_vector.size(); i++)
				fmaps_head.push_back(head_vector[i]);

			for (int i = 0; i < (int)fmaps_head.size(); i++)
			{
				if (i)
					ssHead << ',';
				ssHead << '\"' << fmaps_head[i] << '\"';
			}

			fprintf(fp, "%s\n", ssHead.str().c_str());

			if (env.separateCsv == false)
				mustRenderHeader_fmaps = false;
		}

		// Values
		for (auto l : L)
		{
			const LR& r = childRoiData.at(l);

			if (r.blacklisted)
				continue;

			auto it = childToParentMap.find(l);
			if (it == childToParentMap.end())
				continue;

			const FmapChildInfo& info = it->second;

			std::stringstream ssVals;
			ssVals << std::fixed;

			// File names
			fs::path pint(intFpath), pseg(segFpath);
			ssVals << pint.filename().string() << "," << pseg.filename().string();

			// Annotation columns
			if (env.resultOptions.need_annotation())
			{
				const SlideProps& sli = env.dataset.dataset_props[slide_idx];
				for (const auto& a : sli.annots)
					ssVals << "," << a;
			}

			// Feature maps columns
			ssVals << "," << info.parent_label;
			ssVals << "," << info.center_x;
			ssVals << "," << info.center_y;

			// ROI label (child) and time
			ssVals << "," << l << "," << DEFAULT_T_INDEX;

			// Feature values
			write_feature_values_csv(ssVals, r, F, env, rvbuf, VAL_BUF_LEN, rvfmt);

			fprintf(fp, "%s\n", ssVals.str().c_str());
		}

		std::fflush(fp);
		std::fclose(fp);

		return true;
	}

}
