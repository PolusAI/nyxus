#pragma once

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include "../src/nyx/cli_fpimage_options.h"
#include "../src/nyx/grey_level_cast.h"
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
// Four maps, one per branch of the recorder:
//   * offset     : u = x - floor(min) when the slide holds a negative intensity, so
//                  1 HU == 1 grey level and negative CT no longer wraps on the unsigned
//                  cast. The identity when nothing in the slide is negative. `min` is
//                  the all-pixel minimum, not the within-mask one.
//   * quantized  : a real-valued slide clamped to [min, max] then mapped onto
//                  [0, target dynamic range] (keeps shape, not absolute intensities).
//   * stored     : a slide of integer samples with a header rescale (DICOM, integer
//                  NIfTI) keeps u = stored - shift, and the rescale moves into the
//                  inverse, so a small slope loses no stored level.
//   * native     : carried as it is (in-memory montage input, which never went through
//                  a tile loader at all).
//
// The map is chosen from what the slide holds, not from the file format it arrived in:
// every backend applies the recorded map at load time, so OME-Zarr takes the same
// branches TIFF does. Only a header rescale, which DICOM and NIfTI alone carry, selects
// the stored map.
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

// The one narrowing every load-time map now shares. Converting a double to an unsigned integer
// is undefined -- not a wrapping cast -- when the value is non-finite, negative, or above the
// destination's maximum, so each of those has a stated answer here, and the two narrowings
// differ only in their last step.
void test_hu_grey_level_cast_analytic()
{
    const double nan = std::numeric_limits<double>::quiet_NaN(),
        inf = std::numeric_limits<double>::infinity();
    const uint32_t top = (std::numeric_limits<uint32_t>::max)();

    // the one step they differ in
    EXPECT_EQ(Nyxus::grey_level_rounded<uint32_t>(2.4), 2u);
    EXPECT_EQ(Nyxus::grey_level_rounded<uint32_t>(2.5), 3u);         // llround: half away from zero
    EXPECT_EQ(Nyxus::grey_level_rounded<uint32_t>(0.5), 1u);
    EXPECT_EQ(Nyxus::grey_level_truncated<uint32_t>(2.9), 2u);
    EXPECT_EQ(Nyxus::grey_level_truncated<uint32_t>(3.75), 3u);

    for (bool round_to_nearest : { false, true })
    {
        auto g = [round_to_nearest] (double y) { return Nyxus::grey_level<uint32_t> (y, round_to_nearest); };

        // a non-finite value carries no intensity
        EXPECT_EQ(g(nan), 0u);
        EXPECT_EQ(g(inf), 0u);
        EXPECT_EQ(g(-inf), 0u);

        // below the destination
        EXPECT_EQ(g(-7.5), 0u);
        EXPECT_EQ(g(-0.0), 0u);

        // above it: saturate rather than convert out of range
        EXPECT_EQ(g(5.0e9), top);
        EXPECT_EQ(g(1.0e300), top);
        EXPECT_EQ(g((double) top), top);                             // exactly representable
    }

    // just under the top the two narrowings part: rounding lands on it, truncation stays below
    EXPECT_EQ(Nyxus::grey_level_rounded<uint32_t>(4294967294.6), top);
    EXPECT_EQ(Nyxus::grey_level_truncated<uint32_t>(4294967294.6), top - 1u);

    // grey_level() dispatches on the flag and nothing else
    EXPECT_EQ(Nyxus::grey_level<uint32_t>(2.5, true), Nyxus::grey_level_rounded<uint32_t>(2.5));
    EXPECT_EQ(Nyxus::grey_level<uint32_t>(2.5, false), Nyxus::grey_level_truncated<uint32_t>(2.5));
}

// A slide whose range exceeds the grey type's reaches the offset map's upper end through the
// forward map, which sets the whole-slide vROI's grey range. It used to convert out of range there;
// it saturates now, on both narrowings -- rounding under preserve_hu and truncating without it.
void test_hu_domain_map_offset_saturates_above_grey_range_analytic()
{
    const uint32_t top = (std::numeric_limits<uint32_t>::max)();

    // a real-valued slide carried on the offset map by preserve_hu, spanning past UINT32_MAX
    SlideProps p ("wide.tif", "");
    p.fp_phys_pivoxels = true;
    p.preserve_hu = true;
    p.min_allpix_inten = 0.0;
    p.min_preroi_inten = 0.0;
    p.max_preroi_inten = 1.0e10;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::offset);
    EXPECT_EQ(p.to_grey_level(1.0e10), top);                          // saturates
    EXPECT_EQ(p.to_grey_level(100.5), 101u);                          // and still rounds in range

    // an integer slide takes the offset map without the flag, and truncates
    SlideProps q ("wide.nii", "");
    q.min_allpix_inten = 0.0;
    q.min_preroi_inten = 0.0;
    q.max_preroi_inten = 1.0e10;
    Nyxus::record_intensity_domain_map (q, fpo);

    ASSERT_EQ((int)q.inten_map, (int)IntenMap::offset);
    EXPECT_EQ(q.to_grey_level(1.0e10), top);
    EXPECT_EQ(q.to_grey_level(100.5), 100u);
}

// The quantized map clamps its input to [fpmin, fpmax], so its output is bounded by the target
// dynamic range -- which --fpimgdr can set above UINT32_MAX. The narrowing saturates there too, and
// the non-finite check has to stay ahead of the clamp: left to the narrowing, +Inf would clamp to
// fpmax first and come back as the top grey level instead of 0.
void test_hu_domain_map_quantized_saturates_above_grey_range_analytic()
{
    const uint32_t top = (std::numeric_limits<uint32_t>::max)();

    SlideProps p ("real.tif", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = 0.0;
    p.max_preroi_inten = 10.0;
    FpImageOptions fpo;
    fpo.set_target_dyn_range (1.0e10f);                               // exact in float: 1024 * 9765625
    Nyxus::record_intensity_domain_map (p, fpo);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::quantized);
    EXPECT_DOUBLE_EQ(p.inten_top_grey, 1.0e10);

    EXPECT_EQ(p.to_grey_level(10.0), top);                            // the window maximum saturates
    EXPECT_NEAR((double) p.to_grey_level(1.0), 1.0e9, 2.0);           // a value inside still maps
    EXPECT_EQ(p.to_grey_level(std::numeric_limits<double>::infinity()), 0u);
}

// A PET-like series: stored 0..30000 with RescaleSlope 0.0005 and no intercept, so the physical
// range is [0, 15]. The stored map keeps every stored level as a grey level and carries the slope
// in the inverse. The offset map, which rescales before it narrows, would leave 16 grey levels for
// the same slide -- the same slide under preserve_hu shows that side by side.
void test_hu_domain_map_stored_small_slope_keeps_every_level_analytic()
{
    SlideProps p ("pet.dcm", "");
    p.integer_rescale = true;
    p.rescale_slope = 0.0005;
    p.rescale_intercept = 0.0;
    p.min_allpix_inten = 0.0;
    p.min_preroi_inten = 0.0;
    p.max_preroi_inten = 15.0;
    FpImageOptions fpo;
    Nyxus::record_intensity_domain_map (p, fpo);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::stored);
    EXPECT_DOUBLE_EQ(p.inten_scale, 0.0005);
    EXPECT_DOUBLE_EQ(p.inten_stored_shift, 0.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, 0.0);

    EXPECT_EQ(p.to_grey_level(15.0), 30000u);                         // the top stored level
    EXPECT_EQ(p.to_grey_level(0.0005 * 12345), 12345u);               // the forward map rounds float error away
    EXPECT_EQ(p.to_grey_level(0.0005 * 12346), 12346u);               // and adjacent stored levels stay apart
    EXPECT_DOUBLE_EQ(p.to_source_intensity(30000.0), 15.0);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(12345.0), 6.1725);

    // the same slide under preserve_hu takes the offset map, 1 grey level == 1 intensity unit
    SlideProps q = p;
    q.preserve_hu = true;
    Nyxus::record_intensity_domain_map (q, fpo);
    ASSERT_EQ((int)q.inten_map, (int)IntenMap::offset);
    EXPECT_EQ(q.to_grey_level(15.0), 15u);
    EXPECT_EQ(q.to_grey_level(0.0005 * 12345), q.to_grey_level(0.0005 * 12346));
}

// A textbook CT, slope 1 with an integer intercept, lands on exactly the grey levels and inverse
// the offset map gives the same slide -- with a negative value present (the minimum on grey level
// 0) and without one (physical 0 on grey level 0). So a stored map changes nothing for such a CT.
void test_hu_domain_map_stored_unit_slope_matches_offset_map_analytic()
{
    FpImageOptions fpo;
    for (double min : { -1000.0, 0.0 })
    {
        SlideProps s ("ct.dcm", "");
        s.integer_rescale = true;
        s.rescale_slope = 1.0;
        s.rescale_intercept = -1024.0;
        s.min_allpix_inten = min;
        s.min_preroi_inten = min;
        s.max_preroi_inten = 3071.0;
        Nyxus::record_intensity_domain_map (s, fpo);

        SlideProps o = s;
        o.integer_rescale = false;
        Nyxus::record_intensity_domain_map (o, fpo);

        ASSERT_EQ((int)s.inten_map, (int)IntenMap::stored);
        ASSERT_EQ((int)o.inten_map, (int)IntenMap::offset);
        EXPECT_DOUBLE_EQ(s.inten_scale, o.inten_scale);
        EXPECT_DOUBLE_EQ(s.inten_offset, o.inten_offset);
        // the stored shift is the offset map's shift carried back through the intercept
        EXPECT_DOUBLE_EQ(s.inten_stored_shift, o.inten_offset + 1024.0);
        for (double x : { min, -1.0, 0.0, 1.0, 3071.0 })
            EXPECT_EQ(s.to_grey_level(x), o.to_grey_level(x)) << "min " << min << ", x " << x;
    }
}

// Signed stored integers and a non-unit slope. ct3d_int16.nii's header: stored -200..311, slope 2,
// intercept -1024, physical -1424..-402. The shift is the stored minimum, -200, so grey levels run
// 0..511 and the inverse carries the slope. ct3d_frac.nii's: stored 0..511, slope 0.5, the same
// intercept, physical -1024..-768.5 -- a fractional value the inverse reports exactly. And a
// SlideProps built for a later pass inherits the shift along with the map.
void test_hu_domain_map_stored_signed_and_fractional_analytic()
{
    FpImageOptions fpo;

    SlideProps p ("ct3d_int16.nii", "");
    p.integer_rescale = true;
    p.rescale_slope = 2.0;
    p.rescale_intercept = -1024.0;
    p.min_allpix_inten = -1424.0;
    p.min_preroi_inten = -1424.0;
    p.max_preroi_inten = -402.0;
    Nyxus::record_intensity_domain_map (p, fpo);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::stored);
    EXPECT_DOUBLE_EQ(p.inten_stored_shift, -200.0);
    EXPECT_DOUBLE_EQ(p.inten_scale, 2.0);
    EXPECT_DOUBLE_EQ(p.inten_offset, -1424.0);
    EXPECT_EQ(p.to_grey_level(-1424.0), 0u);
    EXPECT_EQ(p.to_grey_level(-402.0), 511u);
    EXPECT_DOUBLE_EQ(p.to_source_intensity(511.0), -402.0);

    SlideProps f ("ct3d_frac.nii", "");
    f.integer_rescale = true;
    f.rescale_slope = 0.5;
    f.rescale_intercept = -1024.0;
    f.min_allpix_inten = -1024.0;
    f.min_preroi_inten = -1024.0;
    f.max_preroi_inten = -768.5;
    Nyxus::record_intensity_domain_map (f, fpo);

    ASSERT_EQ((int)f.inten_map, (int)IntenMap::stored);
    EXPECT_DOUBLE_EQ(f.inten_stored_shift, 0.0);
    EXPECT_DOUBLE_EQ(f.inten_offset, -1024.0);
    EXPECT_EQ(f.to_grey_level(-768.5), 511u);
    EXPECT_EQ(f.to_grey_level(-1023.5), 1u);
    EXPECT_DOUBLE_EQ(f.to_source_intensity(511.0), -768.5);
    EXPECT_DOUBLE_EQ(f.to_source_intensity(1.0), -1023.5);

    SlideProps pass ("ct3d_int16.nii", "");
    pass.inherit_intensity_domain (p);
    EXPECT_EQ((int)pass.inten_map, (int)IntenMap::stored);
    EXPECT_DOUBLE_EQ(pass.inten_stored_shift, -200.0);
    EXPECT_EQ(pass.to_grey_level(-402.0), 511u);
}

// With no negative value the shift is the stored value at physical 0, rounded down so no sample
// goes below it. Two quotients show both halves of that: -1/0.3 is not an integer and floors to
// -4, while 0.3/0.1 is exactly 3 but computes as 2.9999999999999996, which must not floor to 2.
void test_hu_domain_map_stored_nonnegative_shift_analytic()
{
    FpImageOptions fpo;

    SlideProps p ("a.nii", "");
    p.integer_rescale = true;
    p.rescale_slope = 0.3;
    p.rescale_intercept = 1.0;
    p.min_allpix_inten = 1.0;           // stored 0
    p.min_preroi_inten = 1.0;
    p.max_preroi_inten = 4.0;           // stored 10
    Nyxus::record_intensity_domain_map (p, fpo);

    ASSERT_EQ((int)p.inten_map, (int)IntenMap::stored);
    EXPECT_DOUBLE_EQ(p.inten_stored_shift, -4.0);
    EXPECT_EQ(p.to_grey_level(1.0), 4u);                              // stored 0 -> 0 - (-4)
    EXPECT_NEAR(p.to_source_intensity(4.0), 1.0, 1e-12);

    SlideProps q ("b.nii", "");
    q.integer_rescale = true;
    q.rescale_slope = 0.1;
    q.rescale_intercept = -0.3;
    q.min_allpix_inten = 0.0;
    q.min_preroi_inten = 0.0;
    q.max_preroi_inten = 1.0;
    ASSERT_LT(0.3 / 0.1, 3.0);                                        // the quotient really is short of 3
    Nyxus::record_intensity_domain_map (q, fpo);

    ASSERT_EQ((int)q.inten_map, (int)IntenMap::stored);
    EXPECT_DOUBLE_EQ(q.inten_stored_shift, 3.0);
    EXPECT_EQ(q.to_grey_level(0.0), 0u);                              // physical 0 stays on grey level 0
}

// The stored map needs a positive, finite slope to carry in the inverse, and a header rescale to
// carry at all. Anything else takes the offset map it would otherwise have taken.
void test_hu_domain_map_stored_falls_back_to_offset_analytic()
{
    FpImageOptions fpo;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    struct Case { bool integer_rescale; double slope, intercept; };
    for (Case c : { Case{ true, 0.0, 0.0 }, Case{ true, -1.0, 0.0 }, Case{ true, nan, 0.0 },
                    Case{ true, 1.0, nan }, Case{ false, 0.5, -1024.0 } })
    {
        SlideProps p ("x.nii", "");
        p.integer_rescale = c.integer_rescale;
        p.rescale_slope = c.slope;
        p.rescale_intercept = c.intercept;
        p.min_allpix_inten = -1024.0;
        p.min_preroi_inten = -1024.0;
        p.max_preroi_inten = 3071.0;
        Nyxus::record_intensity_domain_map (p, fpo);

        EXPECT_EQ((int)p.inten_map, (int)IntenMap::offset) << "slope " << c.slope << ", intercept " << c.intercept;
        EXPECT_DOUBLE_EQ(p.inten_scale, 1.0);
        EXPECT_DOUBLE_EQ(p.inten_stored_shift, 0.0);
    }
}
