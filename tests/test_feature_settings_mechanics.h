#pragma once

// Mechanics of Environment's feature-settings registry: which settings vector each feature method
// is handed, rather than the values any family computes. Claims no oracle (SPEC 2).

#include <gtest/gtest.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <typeindex>
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
