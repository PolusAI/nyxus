#pragma once

// 3D GLSZM size emphases against their closed form, on a volume whose zones are large enough that
// the square of a zone size no longer fits in an int (SPEC 4, oracle `analytic`).

// Nothing of its own: gtest, <vector>, D3_GLSZM_feature, the settings recipe and the in-memory
// volume runner all arrive with the common header.
#include "test_3d_glszm_common.h"

// bench_glszm_3d_two_large_zones -- a 64 x 64 x 32 volume, slices 0-15 at intensity 1 and slices
// 16-31 at intensity 2. Each half is one 26-connected zone of L = 64 * 64 * 16 = 65,536 voxels, and
// L^2 = 2^32 wraps to exactly 0 in 32 bits whether the product is read signed or unsigned, so a size
// emphasis is right here only if j^2 is taken in double. With two zones, both of size L:
//
//   SAE = (1/L^2 + 1/L^2) / 2 = 1 / L^2        LAE = (L^2 + L^2) / 2 = L^2
//
// Read with no binning, which keeps the two levels as they are.
void test_3d_glszm_large_zone_size_emphasis_analytic()
{
    const int w = 64, h = 64, d = 32;
    std::vector<PixIntens> voxels (size_t(w) * h * d);
    for (int z = 0; z < d; z++)
        for (int i = 0; i < w * h; i++)
            voxels[size_t(z) * w * h + i] = z < d / 2 ? 1 : 2;

    std::vector<std::vector<double>> fvals;
    D3_GLSZM_feature f;
    ASSERT_NO_FATAL_FAILURE (run_3d_glszm_on_volume (
        fvals, voxels, w, h, d, make_glszm3d_settings (64/*greydepth*/, 0/*no binning*/), f));

    const double L = 64.0 * 64.0 * 16.0,
        L2 = L * L;
    // rel=1e-12 rather than SPEC 7's exact tier (abs 1e-9): these values run from below 1e-9 to past 1e9,
    // so an absolute 1e-9 would accept a small emphasis of 0 and lies below a large one's last bit.
    ASSERT_NEAR (fvals[(int)Feature3D::GLSZM_SAE][0], 1.0 / L2, 1e-12 / L2);
    ASSERT_NEAR (fvals[(int)Feature3D::GLSZM_LAE][0], L2, 1e-12 * L2);
}
