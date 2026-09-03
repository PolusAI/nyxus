#pragma once

#include <gtest/gtest.h>
#include <limits>
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
//   * native     : carried as it is (in-memory montage input, which never went through
//                  a tile loader at all).
//
// The map is chosen from what the slide holds, not from the file format it arrived in:
// every backend applies the recorded map at load time, so OME-Zarr takes the same
// branches TIFF, NIfTI and DICOM do.
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

// --fpimgmin / --fpimgmax may name a window narrower than the slide's own range, and the loaders
// hard-clamp to it at BOTH ends. The forward map has to clamp above as well, or an intensity past
// the window maps beyond the top grey level the loader can store: the whole-slide workflows read
// to_grey_level(max_preroi_inten) straight into the vROI's aux_max, which sets the binning range
// for the intensity and every texture family downstream.
void test_hu_domain_map_quantized_window_clamps_above_analytic()
{
    SlideProps p ("real.tif", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = -1024.0;       // the slide holds more than the window asks for
    p.max_preroi_inten = 3071.0;

    FpImageOptions fpo;
    fpo.raw_min_intensity = "0";        // the window: [0, 1000], well inside the slide
    fpo.raw_max_intensity = "1000";
    ASSERT_TRUE(fpo.parse_input());
    ASSERT_FALSE(fpo.empty());

    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1000.0 / 10000.0);
    EXPECT_DOUBLE_EQ(p.inten_top_grey, 10000.0);

    EXPECT_EQ(p.to_grey_level(0.0), 0u);            // window minimum -> 0
    EXPECT_EQ(p.to_grey_level(500.0), 5000u);       // window midpoint -> DR/2
    EXPECT_EQ(p.to_grey_level(1000.0), 10000u);     // window maximum -> exactly DR

    // Above the window the loader still stores DR, so the forward map must not run past it
    EXPECT_EQ(p.to_grey_level(3071.0), 10000u);     // the slide's own maximum
    EXPECT_EQ(p.to_grey_level(1e9), 10000u);

    // Below the window, unchanged: clamped to 0 rather than wrapping
    EXPECT_EQ(p.to_grey_level(-1024.0), 0u);
}

// Malformed --fpimg* input is rejected at parse time rather than reaching the recorder, where
// an inverted or unparseable window would put the quantization endpoints the wrong way round.
void test_hu_fpimage_options_reject_malformed_analytic()
{
    // an inverted window
    FpImageOptions inverted;
    inverted.raw_min_intensity = "1000";
    inverted.raw_max_intensity = "10";
    EXPECT_FALSE(inverted.parse_input());
    EXPECT_FALSE(inverted.get_last_er_msg().empty());

    // a window of zero width -- the minimum must be strictly below the maximum
    FpImageOptions degenerate;
    degenerate.raw_min_intensity = "5";
    degenerate.raw_max_intensity = "5";
    EXPECT_FALSE(degenerate.parse_input());

    // a non-numeric endpoint
    FpImageOptions notanumber;
    notanumber.raw_min_intensity = "abc";
    notanumber.raw_max_intensity = "10";
    EXPECT_FALSE(notanumber.parse_input());

    // a target dynamic range that spans nothing
    FpImageOptions zerodr;
    zerodr.raw_min_intensity = "0";
    zerodr.raw_max_intensity = "10";
    zerodr.raw_target_dyn_range = "0";
    EXPECT_FALSE(zerodr.parse_input());

    FpImageOptions negdr;
    negdr.raw_min_intensity = "0";
    negdr.raw_max_intensity = "10";
    negdr.raw_target_dyn_range = "-100";
    EXPECT_FALSE(negdr.parse_input());

    // the well-formed control, so the rejections above are not vacuous
    FpImageOptions ok;
    ok.raw_min_intensity = "0";
    ok.raw_max_intensity = "10";
    ok.raw_target_dyn_range = "1000";
    EXPECT_TRUE(ok.parse_input());
    EXPECT_FLOAT_EQ(ok.min_intensity(), 0.0f);
    EXPECT_FLOAT_EQ(ok.max_intensity(), 10.0f);
    EXPECT_FLOAT_EQ(ok.target_dyn_range(), 1000.0f);
}

// The recorder handed the extrema of a scan that measured nothing. Nyxus::
// record_scanned_intensity_range() settles those to a flat zero range before they get here
// (see the two cases below it), so this is the recorder's own floor: the seeds are the
// sentinels the scan starts from -- numeric_limits max / lowest, not the infinities an
// earlier version of this case used, which happen to be substituted at output and so hid
// what an unsettled range actually reports. The map must stay usable either way.
void test_hu_domain_map_nonfinite_slide_range_analytic()
{
    SlideProps p ("real.tif", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = (std::numeric_limits<double>::max)();
    p.max_preroi_inten = (std::numeric_limits<double>::lowest)();
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    // max <= min, so there is no range to quantize into and the offset map carries it
    EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset);

    // whatever the recorded map, a non-finite intensity takes grey level 0 rather than an
    // undefined conversion -- the same convention the loaders use
    EXPECT_EQ(p.to_grey_level(std::numeric_limits<double>::quiet_NaN()), 0u);
    EXPECT_EQ(p.to_grey_level(std::numeric_limits<double>::infinity()), 0u);
    EXPECT_EQ(p.to_grey_level(-std::numeric_limits<double>::infinity()), 0u);
}

// The same convention on a well-formed quantized map: a non-finite sample does not saturate
// to the top grey level, it maps to 0.
void test_hu_domain_map_nonfinite_pixel_quantized_analytic()
{
    SlideProps p ("real.tif", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = 0.0;
    p.max_preroi_inten = 10.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_EQ(p.to_grey_level(10.0), 10000u);       // the finite maximum still saturates
    EXPECT_EQ(p.to_grey_level(std::numeric_limits<double>::quiet_NaN()), 0u);
    EXPECT_EQ(p.to_grey_level(std::numeric_limits<double>::infinity()), 0u);
}

// The offset map has no upper clamp -- its loaders have none either -- so the forward map must
// keep mapping above the slide maximum rather than saturating with the quantized branch.
void test_hu_domain_map_offset_has_no_upper_clamp_analytic()
{
    SlideProps p ("ct.nii", "");
    p.min_allpix_inten = -1024.0;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset);
    EXPECT_DOUBLE_EQ(p.inten_top_grey, 0.0);        // unused on this branch
    EXPECT_EQ(p.to_grey_level(3071.0), 4095u);
    EXPECT_EQ(p.to_grey_level(9000.0), 10024u);     // past the slide maximum, still 1:1
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

// A real-valued OME-Zarr takes the quantization every other real-valued slide takes.
// It used to be excused from the recorder on the grounds that its tile loader copied
// voxels untouched, but that loader narrowed each sample into the unsigned destination
// type -- dropping the fraction and wrapping the negatives -- so "untouched" was never
// true and the identity inverse reported the converted grey levels as source values.
void test_hu_domain_map_zarr_float_quantized_analytic()
{
    SlideProps p ("vol.zarr", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_DOUBLE_EQ(p.inten_scale, 4095.0 / 10000.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, -1024.0);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(0.0), -1024.0);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(10000.0), 3071.0);
}

// A signed-integer OME-Zarr takes the offset map, so its negatives survive the load
// instead of wrapping, and are reported back as the values the file states.
void test_hu_domain_map_zarr_signed_offset_analytic()
{
    SlideProps p ("vol.zarr", "");
    p.min_allpix_inten = -1024.0;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, -1024.0);
    EXPECT_EQ(p.to_grey_level(-1024.0), 0u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(0.0), -1024.0);
}

// An unsigned OME-Zarr holds no negative value, so it takes the identity and its grey
// levels are its own values -- the case the previous blanket exemption did get right.
void test_hu_domain_map_zarr_unsigned_identity_analytic()
{
    SlideProps p ("vol.zarr", "");
    p.min_allpix_inten = 0.0;
    p.min_preroi_inten = 3.0;
    p.max_preroi_inten = 255.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);
    EXPECT_EQ(p.to_grey_level(42.0), 42u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(42.0), 42.0);
}

// A constant real-valued slide has no range to quantize into. The offset map carries it
// instead, at the exact minimum rather than its floor, so the loader's truncation cannot
// eat the fraction: a constant 0.5 slide is stored as grey level 0 and read back as 0.5.
// Recording the identity here instead reported that 0 as the source value.
void test_hu_domain_map_constant_float_analytic()
{
    SlideProps p ("flat.tif", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = 0.5;
    p.max_preroi_inten = 0.5;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.5);

    EXPECT_EQ(p.to_grey_level(0.5), 0u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(0.0), 0.5);
}

// The same, on a negative constant: the offset is the value itself, so it neither wraps
// on the unsigned cast nor loses its sign.
void test_hu_domain_map_constant_negative_float_analytic()
{
    SlideProps p ("flat.tif", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = -0.5;
    p.max_preroi_inten = -0.5;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset);
    EXPECT_DOUBLE_EQ(p.inten_offset, -0.5);
    EXPECT_EQ(p.to_grey_level(-0.5), 0u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(0.0), -0.5);
}

// The scan's own guard, ahead of the recorder. A slide with no finite sample in it -- an
// all-NaN real-valued TIFF, or a montage, which never scans a tile at all -- leaves both
// extrema on the sentinels they started from, which is the only way the maximum can end up
// below the minimum. Left there they reach the recorded offset, and the intensity families
// add that offset back: MIN, MAX and MEAN come out as DBL_MAX, which is finite, so the
// output sanitizer passes 1.797e308 straight into the dataframe. They also reach
// COVERED_IMAGE_INTENSITY_RANGE's divisor, which becomes -Inf, and to_grey_level() in both
// whole-slide workflows, which sets the vROI's grey range for every family -- benign where
// the recorded offset is the sentinel too and cancels, an undefined conversion of DBL_MAX
// where it is not (--preserve-hu leaves the offset at 0, the sentinel not being negative).
// A flat zero range is what such a slide has, and it settles all three consumers.
void test_hu_scanned_range_degenerate_slide_settles_analytic()
{
    SlideProps p ("allnan.tif", "");
    p.fp_phys_pivoxels = true;
    Nyxus::record_scanned_intensity_range (p,
        (std::numeric_limits<double>::max)(),       // slide_I_min, never assigned
        (std::numeric_limits<double>::lowest)(),    // slide_I_max, never assigned
        (std::numeric_limits<double>::max)());      // allpix_I_min, never assigned

    EXPECT_DOUBLE_EQ(p.min_preroi_inten, 0.0);
    EXPECT_DOUBLE_EQ(p.max_preroi_inten, 0.0);
    EXPECT_DOUBLE_EQ(p.min_allpix_inten, 0.0);

    // and the map that comes out of it is the identity, so nothing is added back on the way out
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);
    EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);
    EXPECT_EQ(p.to_grey_level(0.0), 0u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(0.0), 0.0);
}

// A measured range passes through untouched -- the guard tests for max below min, which a
// real scan cannot produce. The all-pixel minimum is measured off the whole buffer rather
// than off the mask, so it survives an empty mask on its own and is settled separately:
// here the mask covered nothing while the buffer held a CT air value, and that value is what
// the load-time offset has to be derived from.
void test_hu_scanned_range_passthrough_analytic()
{
    SlideProps p ("ct.nii", "");
    Nyxus::record_scanned_intensity_range (p, -1024.0, 3071.0, -1024.0);
    EXPECT_DOUBLE_EQ(p.min_preroi_inten, -1024.0);
    EXPECT_DOUBLE_EQ(p.max_preroi_inten, 3071.0);
    EXPECT_DOUBLE_EQ(p.min_allpix_inten, -1024.0);

    SlideProps q ("emptymask.nii", "");
    Nyxus::record_scanned_intensity_range (q,
        (std::numeric_limits<double>::max)(),       // no masked voxel reached the extrema
        (std::numeric_limits<double>::lowest)(),
        -1024.0);                                   // but every voxel reached this one
    EXPECT_DOUBLE_EQ(q.min_preroi_inten, 0.0);
    EXPECT_DOUBLE_EQ(q.max_preroi_inten, 0.0);
    EXPECT_DOUBLE_EQ(q.min_allpix_inten, -1024.0);
}
