#pragma once

// Mechanics of Environment's feature-settings registry: which settings vector each feature method
// is handed, rather than the values any family computes. Claims no oracle (SPEC 2).

#include <gtest/gtest.h>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "../src/nyx/environment.h"
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/feature_method.h"
#include "../src/nyx/feature_settings.h"

#include "../src/nyx/features/2d_geomoments.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/caliper.h"
#include "../src/nyx/features/chords.h"
#include "../src/nyx/features/circle.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/convex_hull.h"
#include "../src/nyx/features/ellipse_fitting.h"
#include "../src/nyx/features/erosion.h"
#include "../src/nyx/features/euler_number.h"
#include "../src/nyx/features/extrema.h"
#include "../src/nyx/features/fractal_dim.h"
#include "../src/nyx/features/gabor.h"
#include "../src/nyx/features/geodetic_len_thickness.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/gldm.h"
#include "../src/nyx/features/gldzm.h"
#include "../src/nyx/features/glrlm.h"
#include "../src/nyx/features/glszm.h"
#include "../src/nyx/features/hexagonality_polygonality.h"
#include "../src/nyx/features/intensity.h"
#include "../src/nyx/features/intensity_histogram.h"
#include "../src/nyx/features/neighbors.h"
#include "../src/nyx/features/ngldm.h"
#include "../src/nyx/features/ngtdm.h"
#include "../src/nyx/features/radial_distribution.h"
#include "../src/nyx/features/roi_radius.h"
#include "../src/nyx/features/zernike.h"

#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/features/3d_gldm.h"
#include "../src/nyx/features/3d_gldzm.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/features/3d_glszm.h"
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/features/3d_ngldm.h"
#include "../src/nyx/features/3d_ngtdm.h"
#include "../src/nyx/features/3d_surface.h"

#include "../src/nyx/features/focus_score.h"
#include "../src/nyx/features/power_spectrum.h"
#include "../src/nyx/features/saturation.h"
#include "../src/nyx/features/sharpness.h"

// The settings vector each registered feature method is entitled to, keyed by its dynamic type --
// the key Environment::compile_feature_settings() files it under and the key phase3.cpp resolves
// it by. Written out here rather than read back from feature2settings_ so that the assertion is
// against the intended pairing and not against whatever the map happens to hold.
//
// IntensityHistogramFeatures shares fsett_PixelIntensity with the intensity family: that is the
// vector reduce_trivial_rois.cpp hands its in-RAM reduce, so the oversized path resolves to the same
// one. Every other family owns a vector of its own.
inline std::map<std::type_index, const Fsettings*> expected_feature_settings (Environment& e)
{
	return {
		// 2D
		{ typeid(PixelIntensityFeatures), &e.fsett_PixelIntensity },
		{ typeid(IntensityHistogramFeatures), &e.fsett_PixelIntensity },
		{ typeid(BasicMorphologyFeatures), &e.fsett_BasicMorphology },
		{ typeid(NeighborsFeature), &e.fsett_Neighbors },
		{ typeid(ContourFeature), &e.fsett_Contour },
		{ typeid(ConvexHullFeature), &e.fsett_ConvexHull },
		{ typeid(EllipseFittingFeature), &e.fsett_EllipseFitting },
		{ typeid(ExtremaFeature), &e.fsett_Extrema },
		{ typeid(EulerNumberFeature), &e.fsett_EulerNumber },
		{ typeid(CaliperFeretFeature), &e.fsett_CaliperFeret },
		{ typeid(CaliperMartinFeature), &e.fsett_CaliperMartin },
		{ typeid(CaliperNassensteinFeature), &e.fsett_CaliperNassenstein },
		{ typeid(ChordsFeature), &e.fsett_Chords },
		{ typeid(HexagonalityPolygonalityFeature), &e.fsett_HexagonalityPolygonality },
		{ typeid(EnclosingInscribingCircumscribingCircleFeature), &e.fsett_EnclosingInscribingCircumscribingCircle },
		{ typeid(GeodeticLengthThicknessFeature), &e.fsett_GeodeticLengthThickness },
		{ typeid(RoiRadiusFeature), &e.fsett_RoiRadius },
		{ typeid(ErosionPixelsFeature), &e.fsett_ErosionPixels },
		{ typeid(FractalDimensionFeature), &e.fsett_FractalDimension },
		{ typeid(GLCMFeature), &e.fsett_GLCM },
		{ typeid(GLRLMFeature), &e.fsett_GLRLM },
		{ typeid(GLDZMFeature), &e.fsett_GLDZM },
		{ typeid(GLSZMFeature), &e.fsett_GLSZM },
		{ typeid(GLDMFeature), &e.fsett_GLDM },
		{ typeid(NGLDMfeature), &e.fsett_NGLDM },
		{ typeid(NGTDMFeature), &e.fsett_NGTDM },
		{ typeid(Imoms2D_feature), &e.fsett_Imoms2D },
		{ typeid(Smoms2D_feature), &e.fsett_Smoms2D },
		{ typeid(GaborFeature), &e.fsett_Gabor },
		{ typeid(ZernikeFeature), &e.fsett_Zernike },
		{ typeid(RadialDistributionFeature), &e.fsett_RadialDistribution },
		// 3D
		{ typeid(D3_VoxelIntensityFeatures), &e.fsett_D3_VoxelIntensity },
		{ typeid(D3_SurfaceFeature), &e.fsett_D3_Surface },
		{ typeid(D3_GLCM_feature), &e.fsett_D3_GLCM },
		{ typeid(D3_GLDM_feature), &e.fsett_D3_GLDM },
		{ typeid(D3_GLDZM_feature), &e.fsett_D3_GLDZM },
		{ typeid(D3_NGLDM_feature), &e.fsett_D3_NGLDM },
		{ typeid(D3_NGTDM_feature), &e.fsett_D3_NGTDM },
		{ typeid(D3_GLSZM_feature), &e.fsett_D3_GLSZM },
		{ typeid(D3_GLRLM_feature), &e.fsett_D3_GLRLM },
		// 2D image quality
		{ typeid(FocusScoreFeature), &e.fsett_FocusScore },
		{ typeid(PowerSpectrumFeature), &e.fsett_PowerSpectrum },
		{ typeid(SaturationFeature), &e.fsett_Saturation },
		{ typeid(SharpnessFeature), &e.fsett_Sharpness }
	};
}

// Every feature method FeatureManager registers resolves to its own family's settings vector. The
// oversized-ROI path (phase3.cpp) is the registry's only consumer, and what it reads out is a
// family's grey depth, co-occurrence offset and neighbourhood radius: a feature handed another
// family's vector is computed at another family's configuration, silently and with no diagnostic.
//
// The check is on the identity of the vector, not on its contents. Two families configured alike
// today compare equal by value and would stop doing so the moment either default moved.
void test_feature_settings_resolve_to_own_family_mechanics()
{
	Environment e;
	e.compile_feature_settings();

	const std::map<std::type_index, const Fsettings*> expected = expected_feature_settings (e);

	// Ask for everything -- 2D, 3D and image quality alike -- so the requested set is the whole
	// registered set and no family is covered by proxy.
	e.theFeatureSet.enableAll (true);
	ASSERT_TRUE (e.theFeatureMgr.compile());
	e.theFeatureMgr.apply_user_selection (e.theFeatureSet);

	int nrf = e.theFeatureMgr.get_num_requested_features();
	ASSERT_GT (nrf, 0);

	std::set<std::type_index> seen;

	for (int i = 0; i < nrf; i++)
	{
		FeatureMethod* f = e.theFeatureMgr.get_feature_method (i);
		ASSERT_NE (f, nullptr);

		// typeid(*f) is the dynamic type. typeid(f) is FeatureMethod*, one type_info shared by every
		// feature and a key the registry is never given.
		const std::type_info& t = typeid(*f);
		SCOPED_TRACE (std::string("feature method ") + t.name());

		auto exp = expected.find (std::type_index(t));
		ASSERT_NE (exp, expected.end()) << "registered in FeatureManager but absent from this table";

		const Fsettings* got = nullptr;
		ASSERT_NO_THROW (got = &e.get_feature_settings (t));
		ASSERT_EQ (got, exp->second);

		seen.insert (std::type_index(t));
	}

	// and nothing in the table has fallen out of FeatureManager
	for (const auto& kv : expected)
		ASSERT_EQ (seen.count (kv.first), (size_t)1);
}

// The refusal path. A type with no registry entry has no settings vector of its own, and the
// intensity vector at index 0 is not a stand-in for one: a texture family handed it runs at the
// defaults rather than at whatever set_metaparam gave that family, and nothing reports it.
// What these assertions discriminate: a lookup that default-inserts a missing key and returns index
// 0 (the intensity settings) fails them.
//
// FeatureMethod* is the static type of the pointer the oversized-ROI loop (phase3.cpp) holds, the
// one type_info that call site must never resolve by; Environment stands for any other type never
// registered.
void test_feature_settings_unregistered_type_refused_mechanics()
{
	Environment e;
	e.compile_feature_settings();

	ASSERT_THROW (e.get_feature_settings (typeid(FeatureMethod*)), std::runtime_error);
	ASSERT_THROW (e.get_feature_settings (typeid(Environment)), std::runtime_error);
}

// The coarse grey depth is the binning every texture family starts from: the 2D families and 3D
// GLDZM/NGLDM read the common GREYDEPTH, and the other five 3D families read a grey depth of their
// own, which compile_feature_settings() starts at the same value. Asserted at each sign, because the
// sign is the scheme (positive MATLAB levels, 0 none, negative PyRadiomics bins) and a vector left at
// its zero fill would read as a deliberate "no binning".
void test_feature_settings_coarse_gray_depth_reaches_every_texture_family_mechanics()
{
	for (int depth : { -20, 0, 20 })
	{
		SCOPED_TRACE ("coarse grey depth " + std::to_string (depth));
		Environment e;
		e.set_coarse_gray_depth (depth);
		e.compile_feature_settings();

		// 2D
		ASSERT_EQ (STNGS_GLCM_GREYDEPTH (e.fsett_GLCM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_GLRLM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_GLSZM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_GLDM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_GLDZM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_NGLDM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_NGTDM), depth);

		// 3D
		ASSERT_EQ (STNGS_GLCM_GREYDEPTH (e.fsett_D3_GLCM), depth);
		ASSERT_EQ (STNGS_GLRLM_GREYDEPTH (e.fsett_D3_GLRLM), depth);
		ASSERT_EQ (STNGS_GLSZM_GREYDEPTH (e.fsett_D3_GLSZM), depth);
		ASSERT_EQ (STNGS_GLDM_GREYDEPTH (e.fsett_D3_GLDM), depth);
		ASSERT_EQ (STNGS_NGTDM_GREYDEPTH (e.fsett_D3_NGTDM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_D3_GLDZM), depth);
		ASSERT_EQ (STNGS_NGREYS (e.fsett_D3_NGLDM), depth);
	}
}

// The other direction: a family's own grey-depth metaparameter overrides the coarse grey depth for
// that family alone, so starting every family at the common value takes no per-family control away.
void test_feature_settings_family_greydepth_overrides_coarse_mechanics()
{
	Environment e;
	e.set_coarse_gray_depth (64);
	e.compile_feature_settings();

	ASSERT_FALSE (e.set_metaparam ("3glszm/greydepth=-20").has_value());
	ASSERT_FALSE (e.set_metaparam ("glcm/greydepth=16").has_value());

	ASSERT_EQ (STNGS_GLSZM_GREYDEPTH (e.fsett_D3_GLSZM), -20);
	ASSERT_EQ (STNGS_GLCM_GREYDEPTH (e.fsett_GLCM), 16);

	ASSERT_EQ (STNGS_GLRLM_GREYDEPTH (e.fsett_D3_GLRLM), 64);
	ASSERT_EQ (STNGS_GLDM_GREYDEPTH (e.fsett_D3_GLDM), 64);
	ASSERT_EQ (STNGS_NGTDM_GREYDEPTH (e.fsett_D3_NGTDM), 64);
	ASSERT_EQ (STNGS_GLCM_GREYDEPTH (e.fsett_D3_GLCM), 64);
	ASSERT_EQ (STNGS_NGREYS (e.fsett_GLCM), 64);
}

// Parses a command line whose only optional argument is --coarseGrayDepth=<raw>; the directories are
// the temp directory, which exists, so the depth is the one thing that can refuse the line
static bool parse_coarse_gray_depth_cmdline (Environment& e, const std::string& raw, std::string* err = nullptr)
{
	const std::string tmp = fs::temp_directory_path().generic_string() + "/";
	std::vector<std::string> args = { "nyxus", "--intDir=" + tmp, "--segDir=" + tmp, "--outDir=" + tmp,
		"--features=MEAN", "--filePattern=.*\\.tif", "--outputType=singlecsv", "--coarseGrayDepth=" + raw };
	std::vector<char*> argv;
	for (auto& a : args)
		argv.push_back (a.data());
	testing::internal::CaptureStderr();
	bool ok = e.parse_cmdline ((int) argv.size(), argv.data());
	std::string captured = testing::internal::GetCapturedStderr();
	if (err)
		*err = captured;
	return ok;
}

// --coarseGrayDepth takes a depth at every sign
void test_feature_settings_coarse_gray_depth_cli_accepted_mechanics()
{
	for (int depth : { -256, -20, 0, 64, 4096 })
	{
		SCOPED_TRACE ("--coarseGrayDepth=" + std::to_string (depth));
		Environment e;
		ASSERT_TRUE (parse_coarse_gray_depth_cmdline (e, std::to_string (depth)));
		ASSERT_EQ (e.get_coarse_gray_depth(), depth);
	}
}

// The refusal path, and the reason each refusal gives: text that is not one whole integer is no
// integer, while INT_MIN (whose magnitude is no int) and values past the int range are integers out
// of range, refused rather than wrapped into a range they are not in
void test_feature_settings_coarse_gray_depth_cli_refused_mechanics()
{
	for (const char* raw : { "64abc", "abc", "1.5" })
	{
		SCOPED_TRACE (std::string ("--coarseGrayDepth=") + raw);
		Environment e;
		std::string err;
		ASSERT_FALSE (parse_coarse_gray_depth_cmdline (e, raw, &err));
		EXPECT_NE (err.find ("expecting an integer constant"), std::string::npos) << err;
	}
	for (const char* raw : { "-2147483648", "2147483648", "-99999999999", "99999999999" })
	{
		SCOPED_TRACE (std::string ("--coarseGrayDepth=") + raw);
		Environment e;
		std::string err;
		ASSERT_FALSE (parse_coarse_gray_depth_cmdline (e, raw, &err));
		EXPECT_NE (err.find ("out of range"), std::string::npos) << err;
	}
}

// The edges of the range parse_grey_depth() checks: an int whose magnitude is an int too. This is a
// type check only; a depth near the edge parses, and whether the buffers it sizes fit is not asked here.
void test_feature_settings_grey_depth_parse_range_mechanics()
{
	for (const char* raw : { "-2147483647", "2147483647", " -20", "+64", "0" })
	{
		SCOPED_TRACE (raw);
		int v = 12345;
		ASSERT_EQ (Nyxus::parse_grey_depth (raw, v), Nyxus::GreyDepthParse::ok);
		ASSERT_EQ (v, std::stoi (raw));
	}
	for (const char* raw : { "-2147483648", "2147483648", "4294967276", "-99999999999999999999" })
	{
		SCOPED_TRACE (raw);
		int v = 12345;
		ASSERT_EQ (Nyxus::parse_grey_depth (raw, v), Nyxus::GreyDepthParse::out_of_range);
		ASSERT_EQ (v, 12345) << "a refused depth must leave the target alone";
	}
	for (const char* raw : { "", "abc", "64abc", "1.5", "-" })
	{
		SCOPED_TRACE (raw);
		int v = 12345;
		ASSERT_EQ (Nyxus::parse_grey_depth (raw, v), Nyxus::GreyDepthParse::not_an_integer);
		ASSERT_EQ (v, 12345);
	}
}

// Every <family>/greydepth metaparameter applies the CLI's rule. INT_MIN would reach std::abs() as UB,
// and 4294967276 is -20 once wrapped into an int, i.e. 20 PyRadiomics-style bins nobody asked for; both
// are refused and leave the family's depth as it was, and so is text that is no integer.
void test_feature_settings_greydepth_metaparam_refused_mechanics()
{
	Environment e;
	e.set_coarse_gray_depth (64);
	e.compile_feature_settings();

	const std::vector<std::pair<std::string, std::function<int()>>> families = {
		{ "glcm",   [&] { return (int) STNGS_GLCM_GREYDEPTH (e.fsett_GLCM); } },
		{ "3glcm",  [&] { return (int) STNGS_GLCM_GREYDEPTH (e.fsett_D3_GLCM); } },
		{ "3gldm",  [&] { return (int) STNGS_GLDM_GREYDEPTH (e.fsett_D3_GLDM); } },
		{ "3ngtdm", [&] { return (int) STNGS_NGTDM_GREYDEPTH (e.fsett_D3_NGTDM); } },
		{ "3glrlm", [&] { return (int) STNGS_GLRLM_GREYDEPTH (e.fsett_D3_GLRLM); } },
		{ "3glszm", [&] { return (int) STNGS_GLSZM_GREYDEPTH (e.fsett_D3_GLSZM); } } };

	for (const auto& [fam, depth] : families)
	{
		const int before = depth();
		for (const char* raw : { "-2147483648", "4294967276", "2147483648" })
		{
			SCOPED_TRACE (fam + "/greydepth=" + raw);
			auto err = e.set_metaparam (fam + "/greydepth=" + raw);
			ASSERT_TRUE (err.has_value());
			EXPECT_NE (err->find ("out of range"), std::string::npos) << *err;
			ASSERT_EQ (depth(), before);
		}
		{
			SCOPED_TRACE (fam + "/greydepth=abc");
			auto err = e.set_metaparam (fam + "/greydepth=abc");
			ASSERT_TRUE (err.has_value());
			EXPECT_NE (err->find ("expecting an integer"), std::string::npos) << *err;
			ASSERT_EQ (depth(), before);
		}
		// and a depth in range is still taken, at either sign
		for (int d : { -20, 32 })
		{
			ASSERT_FALSE (e.set_metaparam (fam + "/greydepth=" + std::to_string (d)).has_value()) << fam;
			ASSERT_EQ (depth(), d) << fam;
		}
	}
}