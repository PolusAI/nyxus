#pragma once

// Mechanics of Environment's feature-settings registry: which settings vector each feature method
// is handed, rather than the values any family computes. Claims no oracle (SPEC 2).

#include <gtest/gtest.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "../src/nyx/environment.h"
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

// The settings vector each registered feature method is entitled to, keyed by the hash of its
// dynamic type -- the key Environment::compile_feature_settings() files it under and the key
// phase3.cpp resolves it by. Written out here rather than read back from feature2settings_ so that
// the assertion is against the intended pairing and not against whatever the map happens to hold.
//
// IntensityHistogramFeatures shares fsett_PixelIntensity with the intensity family: that is the
// vector reduce_trivial_rois.cpp hands its in-RAM reduce, so the oversized path resolves to the same
// one. Every other family owns a vector of its own.
inline std::map<size_t, const Fsettings*> expected_feature_settings (Environment& e)
{
	return {
		// 2D
		{ typeid(PixelIntensityFeatures).hash_code(), &e.fsett_PixelIntensity },
		{ typeid(IntensityHistogramFeatures).hash_code(), &e.fsett_PixelIntensity },
		{ typeid(BasicMorphologyFeatures).hash_code(), &e.fsett_BasicMorphology },
		{ typeid(NeighborsFeature).hash_code(), &e.fsett_Neighbors },
		{ typeid(ContourFeature).hash_code(), &e.fsett_Contour },
		{ typeid(ConvexHullFeature).hash_code(), &e.fsett_ConvexHull },
		{ typeid(EllipseFittingFeature).hash_code(), &e.fsett_EllipseFitting },
		{ typeid(ExtremaFeature).hash_code(), &e.fsett_Extrema },
		{ typeid(EulerNumberFeature).hash_code(), &e.fsett_EulerNumber },
		{ typeid(CaliperFeretFeature).hash_code(), &e.fsett_CaliperFeret },
		{ typeid(CaliperMartinFeature).hash_code(), &e.fsett_CaliperMartin },
		{ typeid(CaliperNassensteinFeature).hash_code(), &e.fsett_CaliperNassenstein },
		{ typeid(ChordsFeature).hash_code(), &e.fsett_Chords },
		{ typeid(HexagonalityPolygonalityFeature).hash_code(), &e.fsett_HexagonalityPolygonality },
		{ typeid(EnclosingInscribingCircumscribingCircleFeature).hash_code(), &e.fsett_EnclosingInscribingCircumscribingCircle },
		{ typeid(GeodeticLengthThicknessFeature).hash_code(), &e.fsett_GeodeticLengthThickness },
		{ typeid(RoiRadiusFeature).hash_code(), &e.fsett_RoiRadius },
		{ typeid(ErosionPixelsFeature).hash_code(), &e.fsett_ErosionPixels },
		{ typeid(FractalDimensionFeature).hash_code(), &e.fsett_FractalDimension },
		{ typeid(GLCMFeature).hash_code(), &e.fsett_GLCM },
		{ typeid(GLRLMFeature).hash_code(), &e.fsett_GLRLM },
		{ typeid(GLDZMFeature).hash_code(), &e.fsett_GLDZM },
		{ typeid(GLSZMFeature).hash_code(), &e.fsett_GLSZM },
		{ typeid(GLDMFeature).hash_code(), &e.fsett_GLDM },
		{ typeid(NGLDMfeature).hash_code(), &e.fsett_NGLDM },
		{ typeid(NGTDMFeature).hash_code(), &e.fsett_NGTDM },
		{ typeid(Imoms2D_feature).hash_code(), &e.fsett_Imoms2D },
		{ typeid(Smoms2D_feature).hash_code(), &e.fsett_Smoms2D },
		{ typeid(GaborFeature).hash_code(), &e.fsett_Gabor },
		{ typeid(ZernikeFeature).hash_code(), &e.fsett_Zernike },
		{ typeid(RadialDistributionFeature).hash_code(), &e.fsett_RadialDistribution },
		// 3D
		{ typeid(D3_VoxelIntensityFeatures).hash_code(), &e.fsett_D3_VoxelIntensity },
		{ typeid(D3_SurfaceFeature).hash_code(), &e.fsett_D3_Surface },
		{ typeid(D3_GLCM_feature).hash_code(), &e.fsett_D3_GLCM },
		{ typeid(D3_GLDM_feature).hash_code(), &e.fsett_D3_GLDM },
		{ typeid(D3_GLDZM_feature).hash_code(), &e.fsett_D3_GLDZM },
		{ typeid(D3_NGLDM_feature).hash_code(), &e.fsett_D3_NGLDM },
		{ typeid(D3_NGTDM_feature).hash_code(), &e.fsett_D3_NGTDM },
		{ typeid(D3_GLSZM_feature).hash_code(), &e.fsett_D3_GLSZM },
		{ typeid(D3_GLRLM_feature).hash_code(), &e.fsett_D3_GLRLM },
		// 2D image quality
		{ typeid(FocusScoreFeature).hash_code(), &e.fsett_FocusScore },
		{ typeid(PowerSpectrumFeature).hash_code(), &e.fsett_PowerSpectrum },
		{ typeid(SaturationFeature).hash_code(), &e.fsett_Saturation },
		{ typeid(SharpnessFeature).hash_code(), &e.fsett_Sharpness }
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

	const std::map<size_t, const Fsettings*> expected = expected_feature_settings (e);

	// Ask for everything -- 2D, 3D and image quality alike -- so the requested set is the whole
	// registered set and no family is covered by proxy.
	e.theFeatureSet.enableAll (true);
	ASSERT_TRUE (e.theFeatureMgr.compile());
	e.theFeatureMgr.apply_user_selection (e.theFeatureSet);

	int nrf = e.theFeatureMgr.get_num_requested_features();
	ASSERT_GT (nrf, 0);

	std::set<size_t> seen;

	for (int i = 0; i < nrf; i++)
	{
		FeatureMethod* f = e.theFeatureMgr.get_feature_method (i);
		ASSERT_NE (f, nullptr);

		// typeid(*f) is the dynamic type. typeid(f) is FeatureMethod*, one type_info shared by every
		// feature and a key the registry is never given.
		const std::type_info& t = typeid(*f);
		SCOPED_TRACE (std::string("feature method ") + t.name());

		auto exp = expected.find (t.hash_code());
		ASSERT_NE (exp, expected.end()) << "registered in FeatureManager but absent from this table";

		const Fsettings* got = nullptr;
		ASSERT_NO_THROW (got = &e.get_feature_settings (t));
		ASSERT_EQ (got, exp->second);

		seen.insert (t.hash_code());
	}

	// and nothing in the table has fallen out of FeatureManager
	for (const auto& kv : expected)
		ASSERT_EQ (seen.count (kv.first), (size_t)1);
}

// The refusal path. A type with no registry entry has no settings vector of its own, and the
// intensity vector at index 0 is not a stand-in for one: a texture family handed it runs at the
// intensity family's grey depth and at GLCM_OFFSET / NGTDM_RADIUS = 0, both degenerate.
// std::map::operator[] default-inserts exactly that 0 and returns it as a valid answer, which is
// why the lookup is a find() that throws.
//
// FeatureMethod* is the type the oversized-ROI call site resolved by before it dereferenced, so it
// is the concrete miss this pins; Environment stands for any other type never registered.
void test_feature_settings_unregistered_type_refused_mechanics()
{
	Environment e;
	e.compile_feature_settings();

	ASSERT_THROW (e.get_feature_settings (typeid(FeatureMethod*)), std::runtime_error);
	ASSERT_THROW (e.get_feature_settings (typeid(Environment)), std::runtime_error);
}
