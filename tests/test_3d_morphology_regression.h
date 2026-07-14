#pragma once

#include "test_3d_morphology_common.h"

// ---------------------------------------------------------------------------------------------------
// Migrated from test_3d_shape.h (Wave 8). The eight 3D shape features whose GT is a self-referential
// snapshot (agrees_gt vs d3shape_GT at 10% tolerance) -> test_3d_morphology_regression.h per registry
// target_test. Shared fixture (d3shape_GT, test_3shape_feature) lives in test_3d_morphology_common.h.
// ---------------------------------------------------------------------------------------------------

void test_3shape_area() {
    test_3shape_feature ("3AREA", Feature3D::AREA);
}

void test_3shape_area2volume() {
    test_3shape_feature ("3AREA_2_VOLUME", Feature3D::AREA_2_VOLUME);
}

void test_3shape_compactness1() {
    test_3shape_feature ("3COMPACTNESS1", Feature3D::COMPACTNESS1);
}

void test_3shape_compactness2() {
    test_3shape_feature ("3COMPACTNESS2", Feature3D::COMPACTNESS2);
}

void test_3shape_sprericaldisproportion() {
    test_3shape_feature ("3SPHERICAL_DISPROPORTION", Feature3D::SPHERICAL_DISPROPORTION);
}

void test_3shape_sphericity() {
    test_3shape_feature ("3SPHERICITY", Feature3D::SPHERICITY);
}

void test_3shape_volumeconvhull() {
    test_3shape_feature ("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
}

void test_3shape_voxelvolume() {
    test_3shape_feature ("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}
