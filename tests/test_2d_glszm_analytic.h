#pragma once

// 2D GLSZM size emphases against their closed form, on a ROI whose zones are large enough that the
// square of a zone size no longer fits in an int (SPEC 4, oracle `analytic`).

// Nothing of its own: gtest, <vector>, GLSZMFeature, the settings recipe and the ROI loader all
// arrive with the common header.
#include "test_2d_glszm_common.h"

// bench_glszm_2d_two_large_zones -- a 216 x 432 rectangle, the top half at intensity 1 and the
// bottom half at intensity 2. Each half is one zone of L = 216 * 216 = 46,656 pixels, and L^2 =
// 2,176,782,336 is past INT_MAX, so a size emphasis is right here only if j^2 is taken in double.
// With two zones, both of size L, the size-zone matrix has one column that matters and:
//
//   SAE = (1/L^2 + 1/L^2) / 2 = 1 / L^2        LAE = (L^2 + L^2) / 2 = L^2
//
// Read at IBSI=true, which keeps the two levels as they are.
static void make_glszm_2d_two_large_zones (std::vector<NyxusPixel>& intensity, std::vector<NyxusPixel>& mask)
{
    const size_t side = 216;
    intensity.clear();
    mask.clear();
    for (size_t y = 0; y < 2 * side; y++)
        for (size_t x = 0; x < side; x++)
        {
            intensity.push_back ({ x, y, y < side ? 1u : 2u });
            mask.push_back ({ x, y, 1u });
        }
}

void test_2d_glszm_large_zone_size_emphasis_analytic()
{
    std::vector<NyxusPixel> intensity, mask;
    make_glszm_2d_two_large_zones (intensity, mask);

    LR roidata;
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());

    GLSZMFeature f;
    f.calculate (roidata, make_glszm2d_settings (true/*ibsi*/, 0));
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    const double L = 216.0 * 216.0,
        L2 = L * L;
    ASSERT_NEAR (roidata.fvals[(int)Feature2D::GLSZM_SAE][0], 1.0 / L2, 1e-12 / L2);
    ASSERT_NEAR (roidata.fvals[(int)Feature2D::GLSZM_LAE][0], L2, 1e-12 * L2);
}
