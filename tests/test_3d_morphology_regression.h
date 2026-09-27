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
// Every feature in this table also carries an oracle row: six of them against MIRP in
// test_3d_morphology_mirp.h, 3AREA additionally against closed-form solids in
// test_3d_morphology_analytic.h, and 3VOXEL_VOLUME and 3VOLUME_CONVEXHULL against MIRP and MATLAB.
// SPEC 3 treats a snapshot and an oracle row as distinct claims, not a contradiction: what these
// pins add is that the number has not moved.
static const ref_vals_map<double> morphology_3d_regression_ref_vals{
    { "3AREA",  46739.022534087213 },
    { "3AREA_2_VOLUME", 0.17037000249358961 },
    { "3COMPACTNESS1",  0.015317649265301225 },
    { "3COMPACTNESS2",  0.083365524768728425 },
    { "3SPHERICAL_DISPROPORTION",   2.2891337610474518 },
    { "3SPHERICITY",    0.43684646874563782 },
    { "3VOLUME_CONVEXHULL", 480655.16666666372 },
    { "3VOXEL_VOLUME",  274431.35826022143 }
};

// frac_tolerance = 1e9, i.e. rel=1e-9, for all eight -- the band the family's other tables
// (morphology_3d_mirp_pca_ref_vals, morphology_3d_mechanics_*_ref_vals) already use, and what the
// arithmetic supports: double-precision geometry with no approximation left in the path, pinned to
// 17 digits.
//
// 3VOLUME_CONVEXHULL held rel=1e-3 for as long as the hull was built in float. It is built in double
// now, and the eps the facet predicate compares against is derived from the cloud's own coordinate
// extent, so the decision has a margin no compiler can round away: the contour voxels are lattice
// points, so a point that is not exactly coplanar with a facet stands at least |det|/|normal| off
// it, and with integer coordinates under 1e3 that is upwards of 1e-5 -- eight orders of magnitude
// above both the eps (~4e-13 here) and the rounding error of the distance itself (~2e-14). Nothing
// between those scales exists for a toolchain to disagree about, which is why the pin can hold at
// rel=1e-9 where the float hull spread 6.5e-04 across platforms.
static constexpr double MORPHOLOGY_3D_REGRESSION_FRAC_TOLERANCE = 1.e9;

static void assert_3d_morphology_feature_regression (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("REGRESSION__") + fname);
    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);
    ASSERT_TRUE(morphology_3d_regression_ref_vals.count(fname) > 0) << fname;
    ASSERT_TRUE(agrees_gt(actual, morphology_3d_regression_ref_vals.at(fname),
                          MORPHOLOGY_3D_REGRESSION_FRAC_TOLERANCE))
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
