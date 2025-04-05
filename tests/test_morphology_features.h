#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

//
// The value of PERIMETER feature should match the pixel count of contour calculated 
// in Matlab as:
//      BW_noholes = imfill(imread('circles.png'), "holes");
//      nnz(bwperim(BW_noholes))    % returns 846
//
void test_morphology_perimeter()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data (roidata, roiDataForPerimeterTest, sizeof(roiDataForPerimeterTest)/sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    ContourFeature f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::PERIMETER][0], 846));
}
