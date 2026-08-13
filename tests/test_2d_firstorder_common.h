#pragma once

// Shared fixture for the 2D first-order families; holds no assertions of its own.

#include <gtest/gtest.h>

#include "../src/nyx/dataset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity.h"
#include "test_data.h"
#include "test_main_nyxus.h"

using namespace Nyxus;

static void calculate_pixel_intensity_feature_values(
    std::vector<std::vector<double>>& fvals,
    Fsettings s = Fsettings(),
    int slide_idx = -1,
    double slide_min = -1.0,
    double slide_max = -1.0)
{
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("", ""));
    if (slide_idx >= 0)
    {
        ds.dataset_props[slide_idx].min_preroi_inten = slide_min;
        ds.dataset_props[slide_idx].max_preroi_inten = slide_max;
    }

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = slide_idx;
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    roidata.make_nonanisotropic_aabb();

    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    roidata.initialize_fvals();
    f.save_value(roidata.fvals);
    fvals = roidata.fvals;
}

