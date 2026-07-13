#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/cli_fpimage_options.h"

// ---------------------------------------------------------------------------
// Hounsfield-Unit (HU) — MECHANICS tests (SPEC.md §2: plumbing, no correctness
// claim). Verifies the --preserve-hu option parses into the preserve_hu() flag;
// the numeric behavior it switches on is vetted in test_hu_analytic.h.
// ---------------------------------------------------------------------------

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
