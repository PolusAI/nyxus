#pragma once

#include <algorithm>
#include <iomanip>
#include "test_3d_morphology_common.h"   // the fixture, and the <iostream> test_main_nyxus.h brings with it
#include "test_ref_vals.h"               // ref_vals_map, and the <string> / <vector> it already includes

// ---------------------------------------------------------------------------------------------------
// The 3D shape features whose GT is a self-referential snapshot. The fixture that produces the
// values is shared (test_3d_morphology_common.h); the goldens and their band are this file's own,
// because they establish no vetting and nothing outside a _regression file should be comparing
// against them.
// ---------------------------------------------------------------------------------------------------

// Pinned Nyxus output at the recipe the shared fixture sets, to full precision. D3_SurfaceFeature is
// geometry: calculate() reads exactly one setting, SINGLEROI, and every 3D morphology fixture in the
// tree sets it false, so the recipe's GREYDEPTH/IBSI/PIXELSIZEUM do not reach these numbers.
//
// Regenerate with test_3d_morphology_dump_regression() below.
//
// Note 3VOLUME_CONVEXHULL and 3VOXEL_VOLUME appear here as snapshots AND in
// test_3d_morphology_mirp.h as MIRP-vetted values -- two different claims about the same feature,
// which SPEC 3 treats as two assertions, not a contradiction. The numbers differ (479997.83 here vs
// MIRP's 496958.32) precisely because one is Nyxus and the other is not.
static const ref_vals_map<double> morphology_3d_regression_ref_vals{
    { "3AREA",  59992 },
    { "3AREA_2_VOLUME", 0.21860475559470999 },
    { "3COMPACTNESS1",  0.010537043861899255 },
    { "3COMPACTNESS2",  0.039449347281835329 },
    { "3SPHERICAL_DISPROPORTION",   2.9375598657539634 },
    { "3SPHERICITY",    0.34041859424142729 },
    { "3VOLUME_CONVEXHULL", 479997.83333333186 },
    { "3VOXEL_VOLUME",  274431.35826022143 }
};

// frac_tolerance = 1e9, i.e. rel=1e-9 -- the band the family's other two tables
// (morphology_3d_mirp_pca_ref_vals, morphology_3d_mechanics_*_ref_vals) already use, and what the
// arithmetic supports: double-precision geometry with no float-precision approximation anywhere in
// the path, pinned to 17 digits. What a looser band lets through here, measured:
// tests/vetting/audit/morphology_3d_golden_regen.md, "Regression drift guards".
static void assert_3d_morphology_feature_regression (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("REGRESSION__") + fname);
    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);
    ASSERT_TRUE(morphology_3d_regression_ref_vals.count(fname) > 0) << fname;
    ASSERT_TRUE(agrees_gt(actual, morphology_3d_regression_ref_vals.at(fname), 1.e9))
        << fname << " actual=" << std::setprecision(17) << actual;
}

// Regenerates every golden in morphology_3d_regression_ref_vals at full precision, in the exact
// shape the table wants. Run it with
//     runAllTests --gtest_filter=*3D_MORPHOLOGY_DUMP_REGRESSION*
// and paste the output over the table above. These are Nyxus' own values on the ut_ phantom, so the
// only honest way to refresh them is to read them out of the same code path the assertions use --
// which is what this does, through the shared fixture.
void test_3d_morphology_dump_regression()
{
    FeatureSet fs;

    std::vector<std::string> names;
    for (const auto& nv : morphology_3d_regression_ref_vals)
        names.push_back (nv.first);
    std::sort (names.begin(), names.end());

    std::cout << "[3DMORPH-REGEN]\n";
    for (const auto& fname : names)
    {
        int fcode = -1;
        ASSERT_TRUE(fs.find_3D_FeatureByString(fname, fcode)) << fname;

        double actual = 0.0;
        calculate_3d_morphology_feature_value (fname, (Nyxus::Feature3D)fcode, actual);
        if (::testing::Test::HasFatalFailure())
            return;

        std::cout << "[3DMORPH-REGEN]    { \"" << fname << "\", "
                  << std::setprecision(17) << actual << " },\n";
    }
}

void test_3d_morphology_area_regression() {
    assert_3d_morphology_feature_regression ("3AREA", Feature3D::AREA);
}

void test_3d_morphology_area_2_volume_regression() {
    assert_3d_morphology_feature_regression ("3AREA_2_VOLUME", Feature3D::AREA_2_VOLUME);
}

void test_3d_morphology_compactness1_regression() {
    assert_3d_morphology_feature_regression ("3COMPACTNESS1", Feature3D::COMPACTNESS1);
}

void test_3d_morphology_compactness2_regression() {
    assert_3d_morphology_feature_regression ("3COMPACTNESS2", Feature3D::COMPACTNESS2);
}

void test_3d_morphology_spherical_disproportion_regression() {
    assert_3d_morphology_feature_regression ("3SPHERICAL_DISPROPORTION", Feature3D::SPHERICAL_DISPROPORTION);
}

void test_3d_morphology_sphericity_regression() {
    assert_3d_morphology_feature_regression ("3SPHERICITY", Feature3D::SPHERICITY);
}

void test_3d_morphology_volume_convex_hull_regression() {
    assert_3d_morphology_feature_regression ("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
}

void test_3d_morphology_voxel_volume_regression() {
    assert_3d_morphology_feature_regression ("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}
