#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/cli_fpimage_options.h"
#include "../src/nyx/slideprops.h"

// ---------------------------------------------------------------------------
// Hounsfield-Unit (HU) / CT intensity handling -- ANALYTIC unit tests.
//
// Oracle: closed form (SPEC.md section 4 token `analytic`). These pin the load-time
// intensity map against hand-computed ground truth:
//
//     Nyxus::record_intensity_domain_map (SlideProps&, const FpImageOptions&)
//     SlideProps::to_grey_level(x) / SlideProps::to_source_intensity(u)
//         (nyxus-src/src/nyx/slideprops.h, slideprops.cpp)
//
// Three maps, one per branch of the recorder:
//   * offset     : u = x - floor(min) when the slide holds a negative intensity, so
//                  1 HU == 1 grey level and negative CT no longer wraps on the unsigned
//                  cast. The identity when nothing in the slide is negative. `min` is
//                  the all-pixel minimum, not the within-mask one.
//   * quantized  : a real-valued slide clamped to [min, max] then mapped onto
//                  [0, target dynamic range] (keeps shape, not absolute intensities).
//   * native     : carried as it is (OME-Zarr, in-memory montage input).
//
// to_source_intensity() is the inverse the intensity families report through, so each
// case also asserts the round trip that makes a reported feature absolute again.
//
// (Mechanics of the CLI/option plumbing live in test_2d_hu_mechanics.h.)
// ---------------------------------------------------------------------------

// A CT volume: minimum -1024 => offset map with offset -1024. Air/min -> 0,
// water 0 -> 1024, bone 3071 -> 4095, and sub-minimum outliers clamp to 0.
void test_hu_domain_map_offset_negative_min_analytic()
{
    SlideProps p ("ct.nii", "");
    p.min_allpix_inten = -1024.0;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, -1024.0);

    EXPECT_EQ(p.to_grey_level(-1024.0), 0u);        // global min / air -> 0
    EXPECT_EQ(p.to_grey_level(0.0), 1024u);         // water -> +offset
    EXPECT_EQ(p.to_grey_level(3071.0), 4095u);      // bone -> full span
    EXPECT_EQ(p.to_grey_level(100.0), 1124u);       // 1 HU == 1 grey level
    EXPECT_EQ(p.to_grey_level(-2000.0), 0u);        // sub-min outlier clamps to 0

    // What the intensity families report: grey levels read back as absolute HU.
    EXPECT_DOUBLE_EQ(p.to_source_intensity(0.0), -1024.0);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(1024.0), 0.0);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(4095.0), 3071.0);
}

// A non-negative integer slide needs no shift, so the map stays the identity and the
// grey levels the pipeline stores are the slide's own values.
void test_hu_domain_map_identity_nonnegative_analytic()
{
    SlideProps p ("plain.tif", "");
    p.min_allpix_inten = 0.0;
    p.min_preroi_inten = 3.0;
    p.max_preroi_inten = 255.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);

    EXPECT_EQ(p.to_grey_level(0.0), 0u);
    EXPECT_EQ(p.to_grey_level(42.0), 42u);
    EXPECT_EQ(p.to_grey_level(255.0), 255u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(42.0), 42.0);
}

// A real-valued slide left in its default mode: clamp to [min, max], then map onto
// [0, DR]. min=-1024, max=3071, DR=10000 -> u = 10000*(x+1024)/4095.
void test_hu_domain_map_quantized_float_analytic()
{
    SlideProps p ("real.tif", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_DOUBLE_EQ(p.inten_scale, 4095.0 / 10000.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, -1024.0);

    EXPECT_EQ(p.to_grey_level(-1024.0), 0u);        // min endpoint -> 0
    EXPECT_EQ(p.to_grey_level(3071.0), 10000u);     // max endpoint -> DR
    EXPECT_EQ(p.to_grey_level(1023.5), 5000u);      // exact midpoint -> DR/2
    EXPECT_EQ(p.to_grey_level(-5000.0), 0u);        // below min -> clamped to 0
    EXPECT_EQ(p.to_grey_level(-1023.9), 0u);        // the cast truncates, as the loaders do

    EXPECT_DOUBLE_EQ(p.to_source_intensity(0.0), -1024.0);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(5000.0), 1023.5);
}

// preserve_hu is what a real-valued slide sets to take the offset map instead of the
// quantization, so absolute intensities survive the load.
void test_hu_domain_map_preserve_hu_float_analytic()
{
    SlideProps p ("real.tif", "");
    p.fp_phys_pivoxels = true;
    p.preserve_hu = true;
    p.min_allpix_inten = -1024.0;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, -1024.0);
    EXPECT_EQ(p.to_grey_level(0.0), 1024u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(1024.0), 0.0);
}

// OME-Zarr is copied into the pipeline untouched, so no inverse may be applied to it
// however negative or real-valued its voxels are.
void test_hu_domain_map_zarr_native_analytic()
{
    SlideProps p ("vol.zarr", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::native);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(42.0), 42.0);
}
