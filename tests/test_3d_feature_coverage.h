#pragma once

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_gldm.h"
#include "../src/nyx/features/3d_gldzm.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/features/3d_glszm.h"
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/features/3d_ngldm.h"
#include "../src/nyx/features/3d_ngtdm.h"
#include "../src/nyx/features/3d_surface.h"

namespace
{
struct Feature3DCoverageCase
{
	std::string name;
	Nyxus::Feature3D code;
};

struct Computed3DFeatureValues
{
	std::vector<std::vector<double>> values;
	std::string setup_error;
};

static std::size_t setting_index(NyxSetting setting)
{
	return static_cast<std::size_t>(setting);
}

static int feature_code_value(Nyxus::Feature3D feature)
{
	return static_cast<int>(feature);
}

static std::size_t feature_index_from_int(int feature)
{
	return static_cast<std::size_t>(feature);
}

static std::size_t feature_code_index(Nyxus::Feature3D feature)
{
	return feature_index_from_int(feature_code_value(feature));
}

static std::string sanitize_3d_feature_test_name(const testing::TestParamInfo<Feature3DCoverageCase>& info)
{
	std::string s = "F_" + info.param.name;
	for (char& c : s)
		if (!std::isalnum(static_cast<unsigned char>(c)))
			c = '_';
	return s;
}

static Fsettings make_3d_coverage_settings()
{
	Fsettings s;
	s.resize(setting_index(NyxSetting::__COUNT__));
	s[setting_index(NyxSetting::SOFTNAN)].rval = 0.0;
	s[setting_index(NyxSetting::TINY)].rval = 0.0;
	s[setting_index(NyxSetting::SINGLEROI)].bval = false;
	s[setting_index(NyxSetting::GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::PIXELSIZEUM)].rval = 100;
	s[setting_index(NyxSetting::PIXELDISTANCE)].ival = 5;
	s[setting_index(NyxSetting::USEGPU)].bval = false;
	s[setting_index(NyxSetting::VERBOSLVL)].ival = 0;
	s[setting_index(NyxSetting::IBSI)].bval = false;
	s[setting_index(NyxSetting::GLCM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::GLCM_OFFSET)].ival = 1;
	s[setting_index(NyxSetting::GLCM_SPARSEINTENS)].bval = true;
	s[setting_index(NyxSetting::GLDM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::GLRLM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::GLSZM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::NGTDM_GREYDEPTH)].ival = 64;
	s[setting_index(NyxSetting::NGTDM_RADIUS)].ival = 1;
	return s;
}

static std::tuple<std::string, std::string, int> get_3d_coverage_phantom()
{
	fs::path this_fpath(__FILE__);
	fs::path pp = this_fpath.parent_path();
	fs::path f1("/data/nifti/phantoms/ut_inten.nii");
	fs::path f2("/data/nifti/phantoms/ut_mask57.nii");
	return {
		(pp.string() + f1.make_preferred().string()),
		(pp.string() + f2.make_preferred().string()),
		57
	};
}

static void initialize_3d_feature_values_as_unwritten(std::vector<std::vector<double>>& values)
{
	values.resize(static_cast<std::size_t>(Nyxus::FeatureIMQ::_COUNT_));
	const double sentinel = std::numeric_limits<double>::quiet_NaN();
	for (auto& v : values)
		v.assign(1, sentinel);
}

static Computed3DFeatureValues build_computed_3d_feature_values()
{
	Computed3DFeatureValues out;
	initialize_3d_feature_values_as_unwritten(out.values);

	try
	{
		auto [ipath, mpath, label] = get_3d_coverage_phantom();
		if (!fs::exists(ipath) || !fs::exists(mpath))
		{
			std::ostringstream ss;
			ss << "missing 3D coverage phantom: " << ipath << " or " << mpath;
			out.setup_error = ss.str();
			return out;
		}

		Fsettings s = make_3d_coverage_settings();
		auto prepare_roi = [&](Environment& e) -> bool {
			e.dataset.dataset_props.reserve(1);
			SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
			if (!scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()))
			{
				out.setup_error = "scan_slide_props failed for 3D coverage phantom";
				return false;
			}
			e.dataset.update_dataset_props_extrema();

			clear_slide_rois(e.uniqueLabels, e.roiData);
			if (!gatherRoisMetrics_3D(e, 0, ipath, mpath, 0))
			{
				out.setup_error = "gatherRoisMetrics_3D failed for 3D coverage phantom";
				return false;
			}

			std::vector<int> batch = { label };
			if (!scanTrivialRois_3D(e, batch, ipath, mpath, 0))
			{
				out.setup_error = "scanTrivialRois_3D failed for 3D coverage phantom";
				return false;
			}
			allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache);
			return true;
		};

		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_VoxelIntensityFeatures f;
			f.calculate(r, s, e.dataset);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_SurfaceFeature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLCM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLDM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLDZM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLRLM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_GLSZM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_NGLDM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
		{
			Environment e;
			if (!prepare_roi(e)) return out;
			LR& r = e.roiData.at(label);
			D3_NGTDM_feature f;
			f.calculate(r, s);
			f.save_value(out.values);
		}
	}
	catch (const std::exception& e)
	{
		out.setup_error = e.what();
	}

	return out;
}

static const Computed3DFeatureValues& computed_3d_feature_values()
{
	static const Computed3DFeatureValues values = build_computed_3d_feature_values();
	return values;
}

static std::set<Nyxus::Feature3D> implemented_3d_feature_codes()
{
	std::set<Nyxus::Feature3D> out;
	auto add = [&out](std::initializer_list<Nyxus::Feature3D> features) {
		out.insert(features.begin(), features.end());
	};
	add(D3_VoxelIntensityFeatures::featureset);
	add(D3_SurfaceFeature::featureset);
	add(D3_GLCM_feature::featureset);
	add(D3_GLDM_feature::featureset);
	add(D3_GLDZM_feature::featureset);
	add(D3_GLRLM_feature::featureset);
	add(D3_GLSZM_feature::featureset);
	add(D3_NGLDM_feature::featureset);
	add(D3_NGTDM_feature::featureset);
	return out;
}

static const std::set<std::string>& embedded_3p_gt_feature_names()
{
	static const std::set<std::string> names = [] {
		std::set<std::string> out;
		auto add_keys = [&out](const auto& m) {
			for (const auto& kv : m)
				out.insert(kv.first);
		};
		add_keys(compat_d3_fo_radiomics_GT);
		add_keys(compat_d3glcm_GT);
		add_keys(compat_3gldm_GT);
		add_keys(compat_3glrlm_GT);
		add_keys(compat_3glszm_GT);
		add_keys(compat_3ngtdm_GT);
		// MATLAB R2025b regionprops3 built-ins are embedded here as vetted status.
		// 3AREA is intentionally excluded: regionprops3 SurfaceArea differs by >10%.
		// Nyxus currently aliases 3MESH_VOLUME to the convex-hull volume.
		out.insert("3MESH_VOLUME");
		out.insert("3VOXEL_VOLUME");
		out.insert("3VOLUME_CONVEXHULL");
		return out;
	}();
	return names;
}

static const std::map<std::string, double>& matlab_regionprops3_shape_gt()
{
	static const std::map<std::string, double> gt = {
		{ "3MESH_VOLUME", 497824.0 },
		{ "3VOXEL_VOLUME", 274432.0 },
		{ "3VOLUME_CONVEXHULL", 497824.0 }
	};
	return gt;
}

static const std::map<std::string, std::vector<double>>& unvetted_3d_local_regression_gt()
{
	static const std::map<std::string, std::vector<double>> gt = {
		{ "3AREA", { 59992 } },
		{ "3AREA_2_VOLUME", { 0.21860475559470999 } },
		{ "3COMPACTNESS1", { 0.010537043861899255 } },
		{ "3COMPACTNESS2", { 0.039449347281835329 } },
		{ "3COV", { 0.29486207043456802 } },
		{ "3COVERED_IMAGE_INTENSITY_RANGE", { 1.0002043207290587 } },
		{ "3ELONGATION", { 0.80989034437672192 } },
		{ "3EXCESS_KURTOSIS", { -1.2127631603215119 } },
		{ "3FLATNESS", { 1.1857880138204875 } },
		{ "3GLCM_ACOR_AVE", { 896.29490954682888 } },
		{ "3GLCM_ASM_AVE", { 0.21714923037245615 } },
		{ "3GLCM_CLUPROM_AVE", { 5319702.1464416385 } },
		{ "3GLCM_CLUSHADE_AVE", { 25300.544725496744 } },
		{ "3GLCM_CLUTEND_AVE", { 1830.7699876860852 } },
		{ "3GLCM_CONTRAST_AVE", { 190.39105377069197 } },
		{ "3GLCM_CORRELATION_AVE", { 0.81160543314537192 } },
		{ "3GLCM_DIFAVE_AVE", { 4.4826978263468851 } },
		{ "3GLCM_DIFENTRO_AVE", { 2.5804645841819687 } },
		{ "3GLCM_DIFVAR_AVE", { 168.82038645655439 } },
		{ "3GLCM_DIS", { 5.5662607942292563, 5.2865125777406536, 5.5662607942292563, 3.5205463112065347, 3.2058935920047036, 3.5205463112065347, 5.5662607942292563, 5.2865125777406536, 5.5662607942292563, 4.6920801473460996, 4.4260407115349301, 4.6920801473460996, 1.379816189466244 } },
		{ "3GLCM_DIS_AVE", { 4.4826978263468824 } },
		{ "3GLCM_ENERGY", { 0.20619709037207257, 0.21210950671886142, 0.20619709037207257, 0.22347189327232553, 0.229768029906674, 0.22347189327232553, 0.20619709037207257, 0.21210950671886142, 0.20619709037207257, 0.21657166507454639, 0.22249608735556295, 0.21657166507454639, 0.2415813859599367 } },
		{ "3GLCM_ENERGY_AVE", { 0.21714923037245615 } },
		{ "3GLCM_ENTROPY", { -6839097.7643487463, -7009967.2050323486, -6839097.7643487463, -7188531.59987259, -7365630.721654892, -7188531.59987259, -6839097.7643487463, -7009967.2050323486, -6839097.7643487463, -7049922.0657091141, -7222060.7506599426, -7049922.0657091141, -7731459.4186325073 } },
		{ "3GLCM_ENTROPY_AVE", { -7090183.3607361866 } },
		{ "3GLCM_HOM1", { 0.63159286085358357, 0.67511464571418467, 0.63159286085358357, 0.675209797165082, 0.7178512712427948, 0.675209797165082, 0.63159286085358357, 0.67511464571418467, 0.63159286085358357, 0.65283716836970751, 0.69041197263012255, 0.65283716836970751, 0.77822827662266392 } },
		{ "3GLCM_HOM1_AVE", { 0.67070662972368189 } },
		{ "3GLCM_HOM2", { 314305.05501034198, 345990.36653998197, 314305.05501034198, 340739.38462753344, 372698.80542099319, 340739.38462753344, 314305.05501034198, 345990.36653998197, 314305.05501034198, 329550.97066911863, 358264.30287407315, 329550.97066911863, 412615.52956223302 } },
		{ "3GLCM_IDMN_AVE", { 0.97393106857854816 } },
		{ "3GLCM_IDM_AVE", { 0.63463076728371071 } },
		{ "3GLCM_IDN_AVE", { 0.95491658377679667 } },
		{ "3GLCM_ID_AVE", { 0.67070662972368211 } },
		{ "3GLCM_INFOMEAS1_AVE", { -0.44521529228752316 } },
		{ "3GLCM_INFOMEAS2_AVE", { 0.97909293489987548 } },
		{ "3GLCM_IV_AVE", { 0.15630830580149896 } },
		{ "3GLCM_JAVE_AVE", { 22.049501181077645 } },
		{ "3GLCM_JE_AVE", { 5.83580592757035 } },
		{ "3GLCM_JMAX_AVE", { 0.46494916987187934 } },
		{ "3GLCM_JVAR_AVE", { 505.29026036419413 } },
		{ "3GLCM_SUMAVERAGE_AVE", { 44.099002362155289 } },
		{ "3GLCM_SUMENTROPY_AVE", { 4.3772246057690438 } },
		{ "3GLCM_SUMVARIANCE", { 1783.3177119109412, 1796.9249632904919, 1783.3177119109412, 1872.6626392475214, 1887.8346987420691, 1872.6626392475214, 1783.3177119109412, 1796.9249632904919, 1783.3177119109412, 1820.9267907064923, 1833.7216725021183, 1820.9267907064923, 1964.1538345421473 } },
		{ "3GLCM_SUMVARIANCE_AVE", { 1830.7699876860852 } },
		{ "3GLCM_VARIANCE", { 504.75832684863678, 506.44120833399279, 504.75832684863678, 504.89933309869377, 506.49086530867982, 504.89933309869377, 504.75832684863678, 506.44120833399279, 504.75832684863678, 504.73432274406343, 506.30964057188811, 504.73432274406343, 504.78984310590351 } },
		{ "3GLCM_VARIANCE_AVE", { 505.29026036419378 } },
		{ "3GLDZM_GLM", { 47.230300235279401 } },
		{ "3GLDZM_GLNU", { 3435.1800942680934 } },
		{ "3GLDZM_GLNUN", { 0.026851399515903585 } },
		{ "3GLDZM_GLV", { 111.77220626552923 } },
		{ "3GLDZM_HGLZE", { 2342.4734665801629 } },
		{ "3GLDZM_LDE", { 314.01248309662088 } },
		{ "3GLDZM_LDHGLE", { 734618.35720259824 } },
		{ "3GLDZM_LDLGLE", { 0.16729167507144088 } },
		{ "3GLDZM_LGLZE", { 0.0005581993242951194 } },
		{ "3GLDZM_SDE", { 0.022387420258025731 } },
		{ "3GLDZM_SDHGLE", { 61.230746106573264 } },
		{ "3GLDZM_SDLGLE", { 1.8362515436029654e-05 } },
		{ "3GLDZM_ZDE", { 10.230312642315168 } },
		{ "3GLDZM_ZDM", { 15.306504185784746 } },
		{ "3GLDZM_ZDNU", { 4330.2817177741472 } },
		{ "3GLDZM_ZDNUN", { 0.033848043255251946 } },
		{ "3GLDZM_ZDV", { 79.723412707174901 } },
		{ "3GLDZM_ZP", { 0.46617376982276121 } },
		{ "3GLRLM_GLNN_AVE", { 0.030276986604527156 } },
		{ "3GLRLM_GLN_AVE", { 7866.1651094407043 } },
		{ "3GLRLM_GLV_AVE", { 295.15266194934725 } },
		{ "3GLRLM_HGLRE_AVE", { 1843.919606614327 } },
		{ "3GLRLM_LGLRE_AVE", { 0.10312612492203123 } },
		{ "3GLRLM_LRE_AVE", { 26.969695007835824 } },
		{ "3GLRLM_LRHGLE_AVE", { 3615.1714395919889 } },
		{ "3GLRLM_LRLGLE_AVE", { 24.851889404429883 } },
		{ "3GLRLM_RE_AVE", { 6.2830967278091743 } },
		{ "3GLRLM_RLNN_AVE", { 0.6876088115538187 } },
		{ "3GLRLM_RLN_AVE", { 177698.02943612196 } },
		{ "3GLRLM_RP_AVE", { 0.94009893441446613 } },
		{ "3GLRLM_RV_AVE", { 22.305382605370092 } },
		{ "3GLRLM_SRE_AVE", { 0.84624689639030182 } },
		{ "3GLRLM_SRHGLE_AVE", { 1740.5351176831964 } },
		{ "3GLRLM_SRLGLE_AVE", { 0.019094835618041379 } },
		{ "3HYPERFLATNESS", { 3.8027657005973312 } },
		{ "3HYPERSKEWNESS", { 0.32001332615517414 } },
		{ "3INTEGRATED_INTENSITY", { 544286216 } },
		{ "3LEAST_AXIS_LEN", { 104.70681271508683 } },
		{ "3MAJOR_AXIS_LEN", { 88.301459868642283 } },
		{ "3MEDIAN_ABSOLUTE_DEVIATION", { 507.12380480410445 } },
		{ "3MINOR_AXIS_LEN", { 71.514499741981993 } },
		{ "3MODE", { 1279 } },
		{ "3NGLDM_DCENE", { 0.14348407632898436 } },
		{ "3NGLDM_DCENT", { 5.2277449211654039 } },
		{ "3NGLDM_DCM", { 13.485998122653307 } },
		{ "3NGLDM_DCNU", { 115443.18172715895 } },
		{ "3NGLDM_DCNUN", { 0.22575716076180957 } },
		{ "3NGLDM_DCP", { 1 } },
		{ "3NGLDM_DCV", { 86.17064428912758 } },
		{ "3NGLDM_GLM", { 16.955115769712151 } },
		{ "3NGLDM_GLNU", { 115443.18172715895 } },
		{ "3NGLDM_GLNUN", { 0.22575716076180957 } },
		{ "3NGLDM_GLV", { 190.08150972702501 } },
		{ "3NGLDM_HDE", { 261.01822590738425 } },
		{ "3NGLDM_HDHGLE", { 20099.770197121401 } },
		{ "3NGLDM_HDLGLE", { 0.025201544837470152 } },
		{ "3NGLDM_HGLCE", { 740.43602941176471 } },
		{ "3NGLDM_LDE", { 0.10159976999534079 } },
		{ "3NGLDM_LDHGLE", { 73.919882197712482 } },
		{ "3NGLDM_LDLGLE", { 5.8337460459982142e-05 } },
		{ "3NGLDM_LGLCE", { 0.00035968375469422158 } },
		{ "3P01", { 1039.3829596412556 } },
		{ "3P25", { 1469.7943925233644 } },
		{ "3P75", { 2487.9072847682119 } },
		{ "3P99", { 3002.3047021943576 } },
		{ "3QCOD", { 0.25724851827174233 } },
		{ "3ROBUST_MEAN", { 0 } },
		{ "3SPHERICAL_DISPROPORTION", { 2.9375598657539634 } },
		{ "3SPHERICITY", { 0.34041859424142729 } },
		{ "3STANDARD_DEVIATION", { 584.80556406962933 } },
		{ "3STANDARD_DEVIATION_BIASED", { 584.80449858510713 } },
		{ "3STANDARD_ERROR", { 1.116333919044723 } },
		{ "3UNIFORMITY_PIU", { 50.59288537549407 } },
		{ "3VARIANCE_BIASED", { 341996.3015653785 } }
	};
	return gt;
}

static double relative_absdiff_pct(double actual, double expected)
{
	double denom = std::abs(expected);
	if (denom == 0.0)
		return actual == expected ? 0.0 : std::numeric_limits<double>::infinity();
	return 100.0 * std::abs(actual - expected) / denom;
}

static void assert_matlab_regionprops3_shape_agreement(const Feature3DCoverageCase& c)
{
	const auto& gt = matlab_regionprops3_shape_gt();
	auto it = gt.find(c.name);
	ASSERT_TRUE(it != gt.end()) << c.name;

	const auto& computed = computed_3d_feature_values();
	ASSERT_TRUE(computed.setup_error.empty()) << computed.setup_error;
	const std::size_t fcode_index = feature_code_index(c.code);
	ASSERT_LT(fcode_index, computed.values.size()) << c.name;
	ASSERT_FALSE(computed.values[fcode_index].empty()) << c.name;
	const double actual = computed.values[fcode_index][0];
	const double expected = it->second;
	ASSERT_TRUE(std::isfinite(actual)) << c.name;
	ASSERT_LE(relative_absdiff_pct(actual, expected), 10.0) << c.name << " actual=" << actual << " MATLAB regionprops3=" << expected;
}

static void assert_embedded_3p_oracle_agreement(const Feature3DCoverageCase& c)
{
	if (compat_d3_fo_radiomics_GT.find(c.name) != compat_d3_fo_radiomics_GT.end())
		test_compat_radiomics_3fo_feature(c.code, c.name);
	else if (compat_d3glcm_GT.find(c.name) != compat_d3glcm_GT.end())
		test_compat_3glcm_feature(c.code, c.name);
	else if (compat_3gldm_GT.find(c.name) != compat_3gldm_GT.end())
		test_compat_3gldm_feature(c.code, c.name);
	else if (compat_3glrlm_GT.find(c.name) != compat_3glrlm_GT.end())
		test_compat_3glrlm_feature(c.code, c.name);
	else if (compat_3glszm_GT.find(c.name) != compat_3glszm_GT.end())
		test_compat_3glszm_feature(c.code, c.name);
	else if (compat_3ngtdm_GT.find(c.name) != compat_3ngtdm_GT.end())
		test_compat_3ngtdm_feature(c.code, c.name);
	else if (matlab_regionprops3_shape_gt().find(c.name) != matlab_regionprops3_shape_gt().end())
		assert_matlab_regionprops3_shape_agreement(c);
	else
		FAIL() << c.name << " is marked WITH_3P_EMBEDDED_GT but no embedded oracle helper was found";
}

static std::vector<Feature3DCoverageCase> feature_3d_cases(bool require_embedded_3p_gt)
{
	std::vector<Feature3DCoverageCase> out;
	const auto& embedded = embedded_3p_gt_feature_names();
	for (const auto& kv : Nyxus::UserFacing_3D_featureNames)
	{
		const bool has_embedded_gt = embedded.find(kv.first) != embedded.end();
		if (has_embedded_gt == require_embedded_3p_gt)
			out.push_back({ kv.first, kv.second });
	}
	return out;
}

static void assert_3d_feature_is_registered_and_computable(const Feature3DCoverageCase& c)
{
	FeatureSet fs;
	int fcode = -1;
	ASSERT_TRUE(fs.find_3D_FeatureByString(c.name, fcode)) << c.name;
	ASSERT_EQ(feature_code_value(c.code), fcode) << c.name;
	ASSERT_GE(fcode, 0) << c.name;

	const auto implemented = implemented_3d_feature_codes();
	ASSERT_TRUE(implemented.find(c.code) != implemented.end()) << c.name << " is public but not in any 3D feature method featureset";

	const auto& computed = computed_3d_feature_values();
	ASSERT_TRUE(computed.setup_error.empty()) << computed.setup_error;
	const std::size_t fcode_index = feature_index_from_int(fcode);
	ASSERT_LT(fcode_index, computed.values.size()) << c.name;
	const auto& vals = computed.values[fcode_index];
	ASSERT_FALSE(vals.empty()) << c.name;
	EXPECT_TRUE(std::any_of(vals.begin(), vals.end(), [](double v) { return std::isfinite(v); })) << c.name << " was not written by the 3D feature calculators";
}

static double local_regression_tolerance(double expected)
{
	return std::max(1.0e-9, std::abs(expected) * 1.0e-6);
}

static void assert_unvetted_local_regression_agreement(const Feature3DCoverageCase& c)
{
	const auto& gt = unvetted_3d_local_regression_gt();
	auto it = gt.find(c.name);
	ASSERT_TRUE(it != gt.end()) << c.name << " has no embedded local-regression baseline";

	const auto& computed = computed_3d_feature_values();
	ASSERT_TRUE(computed.setup_error.empty()) << computed.setup_error;
	const std::size_t fcode_index = feature_code_index(c.code);
	ASSERT_LT(fcode_index, computed.values.size()) << c.name;

	const auto& actual = computed.values[fcode_index];
	const auto& expected = it->second;
	ASSERT_EQ(expected.size(), actual.size()) << c.name;
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		ASSERT_TRUE(std::isfinite(actual[i])) << c.name << "[" << i << "]";
		const double tolerance = local_regression_tolerance(expected[i]);
		EXPECT_NEAR(actual[i], expected[i], tolerance)
			<< c.name << "[" << i << "] actual=" << actual[i]
			<< " expected=" << expected[i] << " tolerance=" << tolerance;
	}
}

class Test3DFeature_WITH_3P_EMBEDDED_GT : public testing::TestWithParam<Feature3DCoverageCase> {};

TEST_P(Test3DFeature_WITH_3P_EMBEDDED_GT, PublicFeatureIsComputableAndHasEmbeddedOracle)
{
	const auto& c = GetParam();
	ASSERT_TRUE(embedded_3p_gt_feature_names().find(c.name) != embedded_3p_gt_feature_names().end()) << c.name;
	assert_3d_feature_is_registered_and_computable(c);
	assert_embedded_3p_oracle_agreement(c);
}

INSTANTIATE_TEST_SUITE_P(
	WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases(true)),
	sanitize_3d_feature_test_name);

class Test3DFeature_UNVETTED_LOCAL_REGRESSION : public testing::TestWithParam<Feature3DCoverageCase> {};

TEST_P(Test3DFeature_UNVETTED_LOCAL_REGRESSION, PublicFeatureIsComputableButHasNoEmbeddedOracleYet)
{
	const auto& c = GetParam();
	ASSERT_TRUE(embedded_3p_gt_feature_names().find(c.name) == embedded_3p_gt_feature_names().end()) << c.name;
	assert_3d_feature_is_registered_and_computable(c);
	assert_unvetted_local_regression_agreement(c);
}

INSTANTIATE_TEST_SUITE_P(
	UNVETTED_LOCAL_REGRESSION,
	Test3DFeature_UNVETTED_LOCAL_REGRESSION,
	testing::ValuesIn(feature_3d_cases(false)),
	sanitize_3d_feature_test_name);

TEST(TEST_NYXUS, TEST_3D_FEATURE_COVERAGE_COUNTS)
{
	EXPECT_EQ(213u, Nyxus::UserFacing_3D_featureNames.size());
	EXPECT_EQ(94u, feature_3d_cases(true).size());
	EXPECT_EQ(119u, feature_3d_cases(false).size());
	EXPECT_EQ(Nyxus::UserFacing_3D_featureNames.size(), feature_3d_cases(true).size() + feature_3d_cases(false).size());
	EXPECT_EQ(119u, unvetted_3d_local_regression_gt().size());
	for (const auto& c : feature_3d_cases(false))
		EXPECT_TRUE(unvetted_3d_local_regression_gt().find(c.name) != unvetted_3d_local_regression_gt().end()) << c.name;
}
}
