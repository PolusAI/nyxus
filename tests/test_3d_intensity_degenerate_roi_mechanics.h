#pragma once

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "../src/nyx/dataset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_intensity.h"
#include "test_main_nyxus.h"

// ---------------------------------------------------------------------------
// Degenerate-ROI mechanics for the 3D intensity family (SPEC.md 2: plumbing, no
// correctness claim). The 3D twin of test_2d_intensity_degenerate_roi_mechanics.h.
//
// QCOD, UNIFORMITY_PIU and COV are quotients whose denominators -- p75 + p25,
// aux_max + aux_min, and the mean -- a populated ROI sitting entirely at grey level 0
// leaves at 0. The offset map puts a volume's own minimum on grey level 0, so an all-air
// CT ROI reaches the 3D family in exactly that state.
//
// As in 2D this is asserted on the feature object, not through the Python API: the output
// sanitizer replaces a NaN with ResultOptions::noval(), and at the default noval of 0.0 a
// guarded quotient and an unguarded one read the same. save_value()'s output is read
// directly so the difference stays observable.
//
// The ROI is built by hand rather than loaded from a volume. The 3D grey-level pass reads
// only raw_pixels_3D, aux_min, aux_max, aux_area and slide_idx, and slide_idx -1 names no
// scanned slide, so report_in_source_domain() takes the identity map and leaves the
// grey-level values as they are.
// ---------------------------------------------------------------------------

// Fills `r` as a 3x3x2 ROI, every voxel carrying intensity `grey`. LR is not copyable, so the
// caller owns it and this populates it in place.
static void fill_3d_constant_roi (LR& r, PixIntens grey)
{
    r.slide_idx = -1;
    for (int z = 0; z < 2; z++)
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                r.raw_pixels_3D.push_back (Pixel3 (x, y, z, grey));
    r.aux_area = (decltype(r.aux_area)) r.raw_pixels_3D.size();
    r.aux_min = grey;
    r.aux_max = grey;
}

static std::vector<std::vector<double>> calculate_3d_intensity (LR& r)
{
    Dataset ds;
    ds.dataset_props.push_back (SlideProps ("", ""));
    Fsettings s;
    D3_VoxelIntensityFeatures f;
    EXPECT_NO_THROW (f.calculate (r, s, ds));
    r.initialize_fvals();
    f.save_value (r.fvals);
    return r.fvals;
}

// All three quotients come back as 0 rather than NaN on an ROI of one grey level at 0.
void test_3d_intensity_zero_valued_roi_ratios_mechanics()
{
    LR r (100);
    fill_3d_constant_roi (r, 0);
    ASSERT_EQ (r.raw_pixels_3D.size(), 18u);

    auto v = calculate_3d_intensity (r);
    double qcod = v[(int)Feature3D::QCOD][0],
        piu = v[(int)Feature3D::UNIFORMITY_PIU][0],
        cov = v[(int)Feature3D::COV][0];

    ASSERT_FALSE (std::isnan(qcod)) << "3QCOD is NaN before the output sanitizer";
    ASSERT_FALSE (std::isnan(piu)) << "3UNIFORMITY_PIU is NaN before the output sanitizer";
    ASSERT_FALSE (std::isnan(cov)) << "3COV is NaN before the output sanitizer";
    ASSERT_DOUBLE_EQ (qcod, 0.0);
    ASSERT_DOUBLE_EQ (piu, 0.0);
    ASSERT_DOUBLE_EQ (cov, 0.0);

    // the distribution is still described
    ASSERT_DOUBLE_EQ (v[(int)Feature3D::MIN][0], 0.0);
    ASSERT_DOUBLE_EQ (v[(int)Feature3D::MAX][0], 0.0);
    ASSERT_DOUBLE_EQ (v[(int)Feature3D::MEAN][0], 0.0);
}

// A non-degenerate ROI is unaffected: every quotient keeps its ordinary definition.
void test_3d_intensity_nonzero_roi_ratios_unaffected_mechanics()
{
    // not constant, and not at 0: two grey levels, so the mean, the quartiles and the range
    // are all non-zero and every guard sees a live denominator
    LR r (100);
    fill_3d_constant_roi (r, 100);
    for (size_t i = 0; i < r.raw_pixels_3D.size(); i += 2)
        r.raw_pixels_3D[i].inten = 200;
    r.aux_max = 200;

    auto v = calculate_3d_intensity (r);
    double qcod = v[(int)Feature3D::QCOD][0],
        piu = v[(int)Feature3D::UNIFORMITY_PIU][0],
        cov = v[(int)Feature3D::COV][0];

    ASSERT_FALSE (std::isnan(qcod));
    ASSERT_FALSE (std::isnan(piu));
    ASSERT_FALSE (std::isnan(cov));
    // the guards must not have collapsed a live denominator to the degenerate answer
    ASSERT_NE (piu, 0.0);
    ASSERT_NE (cov, 0.0);
}
