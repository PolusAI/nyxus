#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/slideprops.h"
#include "../src/nyx/cli_fpimage_options.h"

// ---------------------------------------------------------------------------
// Hounsfield-Unit (HU) / CT intensity handling — baseline unit tests.
//
// These pin down the CURRENT behavior of the single load-time quantization
// primitive that HU preservation will modify:
//
//     SlideProps::uint_friendly_inten(double x, double uint_dynrange)
//         (nyxus-src/src/nyx/slideprops.h)
//
// Today it has two branches:
//   * fp_phys_pivoxels == true : clamp x to [min_preroi, max_preroi] then
//     min-max map onto [0, uint_dynrange]  (destroys absolute HU, keeps shape).
//   * fp_phys_pivoxels == false: raw (unsigned int) cast — which WRAPS for
//     negative CT stored values (e.g. -1024 HU) and is the bug HU mode fixes.
//
// A future "preserve HU" path will add an offset-preserving branch
// (u = x - floor(global_min)). Until then these tests are the regression net
// for the function's existing, well-defined behavior.
// ---------------------------------------------------------------------------

// fp_phys_pivoxels==true: min-max normalization over a CT-like HU span.
// min=-1024, max=3071, DR=10000  ->  u = 10000*(x+1024)/4095, clamped.
void test_hu_uint_friendly_normalization_ct_range()
{
    SlideProps p("", "");
    p.fp_phys_pivoxels = true;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;
    const double DR = 10000.0;

    EXPECT_EQ(p.uint_friendly_inten(-1024.0, DR), 0u);      // min endpoint -> 0
    EXPECT_EQ(p.uint_friendly_inten(3071.0, DR), 10000u);   // max endpoint -> DR
    EXPECT_EQ(p.uint_friendly_inten(1023.5, DR), 5000u);    // exact midpoint -> DR/2
    EXPECT_EQ(p.uint_friendly_inten(-5000.0, DR), 0u);      // below min -> clamped to 0
    EXPECT_EQ(p.uint_friendly_inten(10000.0, DR), 10000u);  // above max -> clamped to DR
}

// fp_phys_pivoxels==false: raw truncating cast. Only NON-negative inputs are
// well-defined; assert those. (Negative HU through this branch is undefined /
// wraps — that broken CT case is deliberately NOT asserted here; it is the
// defect HU preservation addresses.)
void test_hu_uint_friendly_rawcast_nonnegative()
{
    SlideProps p("", "");
    p.fp_phys_pivoxels = false;   // min/max unused on this branch

    EXPECT_EQ(p.uint_friendly_inten(0.0, 10000.0), 0u);
    EXPECT_EQ(p.uint_friendly_inten(42.0, 10000.0), 42u);
    EXPECT_EQ(p.uint_friendly_inten(99.9, 10000.0), 99u);     // truncates toward zero
    EXPECT_EQ(p.uint_friendly_inten(10000.0, 10000.0), 10000u);
}

// preserve_hu==true: slope-1 offset map u = round(x - floor(min)), 1 HU == 1 grey
// level. min=-1024 => offset -1024: air/min -> 0, water 0 -> 1024, bone 3071 -> 4095.
// dynrange is intentionally ignored on this branch. Negative HU no longer wraps.
void test_hu_uint_friendly_preserve_offset()
{
    SlideProps p("", "");
    p.preserve_hu = true;
    p.min_preroi_inten = -1024.0;
    p.max_preroi_inten = 3071.0;      // unused by the offset branch, set for realism
    const double DR = 10000.0;        // ignored in HU mode

    EXPECT_EQ(p.uint_friendly_inten(-1024.0, DR), 0u);      // global min / air -> 0
    EXPECT_EQ(p.uint_friendly_inten(0.0, DR), 1024u);       // water -> +offset
    EXPECT_EQ(p.uint_friendly_inten(3071.0, DR), 4095u);    // bone -> full span
    EXPECT_EQ(p.uint_friendly_inten(100.0, DR), 1124u);     // 1 HU == 1 grey level
    EXPECT_EQ(p.uint_friendly_inten(-2000.0, DR), 0u);      // sub-min outlier clamps to 0
}

// FpImageOptions parses the --preserve-hu raw string into the preserve_hu() flag
// via Nyxus::parse_as_bool (accepts TRUE/FALSE/T/F, case-insensitive).
// Default is off; empty leaves it off; a non-boolean is a parse error.
void test_hu_fpimage_options_parse()
{
    FpImageOptions off;
    ASSERT_TRUE(off.parse_input());
    EXPECT_FALSE(off.preserve_hu());             // absent -> default off

    FpImageOptions on;
    on.raw_preserve_hu = "true";
    ASSERT_TRUE(on.parse_input());
    EXPECT_TRUE(on.preserve_hu());

    FpImageOptions onF;
    onF.raw_preserve_hu = "False";               // case-insensitive
    ASSERT_TRUE(onF.parse_input());
    EXPECT_FALSE(onF.preserve_hu());

    FpImageOptions bad;
    bad.raw_preserve_hu = "banana";
    EXPECT_FALSE(bad.parse_input());             // non-boolean -> parse error
}
