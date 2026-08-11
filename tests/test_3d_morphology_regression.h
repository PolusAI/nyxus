#pragma once

#include "test_3d_morphology_common.h"

// ---------------------------------------------------------------------------------------------------
// Migrated from test_3d_shape.h (Wave 8). The eight 3D shape features whose GT is a self-referential
// snapshot (agrees_gt vs morphology_3d_regression_ref_vals at 10% tolerance) -> test_3d_morphology_regression.h per registry
// target_test. Shared fixture (morphology_3d_regression_ref_vals, assert_3d_morphology_feature) lives in test_3d_morphology_common.h.
// ---------------------------------------------------------------------------------------------------

void test_3d_morphology_area_regression() {
    assert_3d_morphology_feature ("3AREA", Feature3D::AREA);
}

void test_3d_morphology_area_2_volume_regression() {
    assert_3d_morphology_feature ("3AREA_2_VOLUME", Feature3D::AREA_2_VOLUME);
}

void test_3d_morphology_compactness1_regression() {
    assert_3d_morphology_feature ("3COMPACTNESS1", Feature3D::COMPACTNESS1);
}

void test_3d_morphology_compactness2_regression() {
    assert_3d_morphology_feature ("3COMPACTNESS2", Feature3D::COMPACTNESS2);
}

void test_3d_morphology_spherical_disproportion_regression() {
    assert_3d_morphology_feature ("3SPHERICAL_DISPROPORTION", Feature3D::SPHERICAL_DISPROPORTION);
}

void test_3d_morphology_sphericity_regression() {
    assert_3d_morphology_feature ("3SPHERICITY", Feature3D::SPHERICITY);
}

void test_3d_morphology_volume_convex_hull_regression() {
    assert_3d_morphology_feature ("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
}

void test_3d_morphology_voxel_volume_regression() {
    assert_3d_morphology_feature ("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}
