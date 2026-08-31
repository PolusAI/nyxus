#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/environment.h"
#include "../src/nyx/feature_settings.h"
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/slideprops.h"

static std::tuple<std::string, std::string, int> get_3d_firstorder_phantom()
{
    const fs::path tests_dir = fs::path(__FILE__).parent_path();
    return {
        (tests_dir / "data/nifti/phantoms/ut_inten.nii").string(),
        (tests_dir / "data/nifti/phantoms/ut_mask57.nii").string(),
        57
    };
}

// Fixture only: oracle values and regression snapshots stay in the files that assert them.
static void calculate_3d_firstorder_values(std::vector<std::vector<double>>& values)
{
    auto [ipath, mpath, label] = get_3d_firstorder_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.fpimageOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    LR& r = e.roiData.at(label);
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_VoxelIntensityFeatures f;
    Fsettings s;
    ASSERT_NO_THROW(f.calculate(r, s, e.dataset));
    f.save_value(r.fvals);
    values = r.fvals;
}
