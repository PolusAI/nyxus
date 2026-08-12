#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_surface.h"
#include "test_ref_vals.h"

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();

// Fixture only (SPEC 6.3.1): loads the segmented phantom, runs D3_SurfaceFeature and hands the value
// back. Judging it belongs to whichever file owns the reference table, so the goldens and their
// tolerances now live in test_3d_morphology_regression.h and test_3d_morphology_matlab.h. This header
// used to hold one snapshot table that both of those files asserted against, which is how
// test_3d_morphology_mesh_volume_matlab ended up pinned to Nyxus output instead of to MATLAB's.
void calculate_3d_morphology_feature_value (const std::string& fname, const Nyxus::Feature3D& expecting_fcode, double& out)
{
#if 0

    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    Environment e;
    clear_slide_rois (e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // set feature's state
    Environment::ibsi_compliance = false;

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_SurfaceFeature f;
    Fsettings s;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0];

    // verdict (the caller's table supplies the golden)
    out = atot;

    *********************************
#endif

    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    Environment e;
    // (1) slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    // (2) properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
    // (3) voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
    // (4) buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (5) feature settings
    Fsettings s;    
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 128;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = true;
    //

    // (6) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_SurfaceFeature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) saving values

    f.save_value(r.fvals);

    // we don't expect subfeatures so using subfeature [0]
    double atot = r.fvals[fcode][0];

    out = atot;
}

// The ten test functions that used to live here have been distributed to their taxonomy homes
// (registry target_test): the eight regression shape features -> test_3d_morphology_regression.h;
// 3MESH_VOLUME (matlab/vetted) and the MATLAB-derived covariance/eigenvalue math test ->
// test_3d_morphology_matlab.h. This header now carries only the shared fixture they include.


