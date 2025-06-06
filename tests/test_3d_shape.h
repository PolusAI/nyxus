#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_surface.h"

static std::unordered_map<std::string, float> d3shape_GT{
    { "3AREA",  58457 },
    { "3AREA_2_VOLUME", 0.21 },
    { "3COMPACTNESS1",  0.011 },
    { "3COMPACTNESS2",  0.043 },
    { "3MESH_VOLUME",   478516 },
    { "3SPHERICAL_DISPROPORTION",   2.9 },
    { "3SPHERICITY",    0.35 },
    { "3VOLUME_CONVEXHULL", 478516 },
    { "3VOXEL_VOLUME",  274431 }
};

std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

void test_3shape_feature (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    clear_slide_rois();
    ASSERT_TRUE(gatherRoisMetrics_3D(ipath, mpath));
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(batch, ipath, mpath));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch));

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // set feature's state
    Environment::ibsi_compliance = false;

    // extract the feature
    LR& r = Nyxus::roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_SurfaceFeature f;
    ASSERT_NO_THROW(f.calculate(r));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3shape_GT[fname], 10.));
}

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

void test_3shape_meshvolume() {
    test_3shape_feature ("3MESH_VOLUME", Feature3D::MESH_VOLUME);
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


