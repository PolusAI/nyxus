#pragma once

#include <string>
#include "test_3d_morphology_common.h"
#include "test_ref_vals.h"

// ---------------------------------------------------------------------------------------------------
// The eight 3D shape features whose GT is a self-referential snapshot. The fixture that produces the
// values is shared (test_3d_morphology_common.h); the goldens and the 10% band are this file's own,
// because they establish no vetting and nothing outside a _regression file should be comparing
// against them.
// ---------------------------------------------------------------------------------------------------

// Pinned Nyxus output at the IBSI-path recipe the fixture sets (GREYDEPTH=128, ibsi=true). Note
// 3VOLUME_CONVEXHULL appears here as a snapshot AND in test_3d_morphology_matlab.h as a
// MATLAB-vetted value -- two different claims about the same feature, which SPEC 3 treats as two
// assertions, not a contradiction. The numbers differ (478516 here vs MATLAB's 497824) precisely
// because one is Nyxus and the other is not.
//
// 3MESH_VOLUME had a golden here that no function read: this file defines eight regression tests and
// the table carried nine entries. A golden nothing asserts is not a drift guard, it is a number that
// looks like one, so it is gone; the feature's only executable assertion is the MATLAB one.
static ref_vals_map<double> morphology_3d_regression_ref_vals{
    { "3AREA",  58457 },
    { "3AREA_2_VOLUME", 0.21 },
    { "3COMPACTNESS1",  0.011 },
    { "3COMPACTNESS2",  0.043 },
    { "3SPHERICAL_DISPROPORTION",   2.9 },
    { "3SPHERICITY",    0.35 },
    { "3VOLUME_CONVEXHULL", 478516 },
    { "3VOXEL_VOLUME",  274431 }
};

static void assert_3d_morphology_feature_regression (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("REGRESSION__") + fname);
    double actual = 0.0;
    calculate_3d_morphology_feature_value (fname, expecting_fcode, actual);
    ASSERT_TRUE(morphology_3d_regression_ref_vals.count(fname) > 0) << fname;
    ASSERT_TRUE(agrees_gt(actual, morphology_3d_regression_ref_vals[fname], 10.));
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
