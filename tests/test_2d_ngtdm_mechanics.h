#pragma once

#include <utility>                  // std::make_pair

#include "test_2d_ngtdm_common.h"   // gtest, <string>, <vector>, the fixture

// Machinery checks for the 2D NGTDM feature class. SPEC 2 mechanics tier: these assert how the code
// behaves, not what a feature value should be, so they establish no vetting and carry no oracle
// token and no reference table.

// NGTDMFeature::n_levels is a STATIC, shared by every test in the binary and by every caller in a
// process. In IBSI mode it must not matter: ngtdm.cpp forces the grey-binning info to 0 whenever
// IBSI compliance is on, so the phantom's own levels are used whatever the static says.
//
// That immunity is the only reason the IBSI and mirp assertions are safe to run in a binary where
// another test sets the static to 100 -- so it is checked rather than assumed. Outside IBSI mode the
// static is decisive: the same fixture gives NGTDM_CONTRAST 3169.93 at 100 levels and 6634.50 at the
// default 0.
void test_2d_ngtdm_ibsi_mode_ignores_n_levels_mechanics()
{
    const Fsettings s = make_ngtdm2d_settings (true);

    // the feature is named in the loop's own list rather than hidden in a parallel array: the
    // coverage scanners read feature coverage off assertion and loop lines, so an array of enum
    // values with the names kept somewhere else makes this file look like it tests nothing
    for (const auto& f : { std::make_pair (Nyxus::Feature2D::NGTDM_COARSENESS, "NGTDM_COARSENESS"),
                           std::make_pair (Nyxus::Feature2D::NGTDM_CONTRAST, "NGTDM_CONTRAST"),
                           std::make_pair (Nyxus::Feature2D::NGTDM_BUSYNESS, "NGTDM_BUSYNESS"),
                           std::make_pair (Nyxus::Feature2D::NGTDM_COMPLEXITY, "NGTDM_COMPLEXITY"),
                           std::make_pair (Nyxus::Feature2D::NGTDM_STRENGTH, "NGTDM_STRENGTH") })
    {
        const std::vector<double> unset = ngtdm_2d_phantom_slice_values (f.first, s, 0);
        const std::vector<double> polluted = ngtdm_2d_phantom_slice_values (f.first, s, 100);

        ASSERT_EQ (unset.size(), polluted.size()) << f.second;
        for (size_t z = 0; z < unset.size(); z++)
        {
            SCOPED_TRACE (std::string ("MECHANICS__n_levels__") + f.second + "__z"
                          + std::to_string (z + 1));
            // bit-exact, not merely close: the setting is either read or it is not
            ASSERT_DOUBLE_EQ (unset[z], polluted[z]) << f.second;
        }
    }
}

// The fixture helper restores the static it borrows. Without this, the order tests happen to run in
// would decide what a later non-IBSI NGTDM test computes.
void test_2d_ngtdm_slice_helper_restores_n_levels_mechanics()
{
    const int before = NGTDMFeature::n_levels;
    ngtdm_2d_phantom_slice_values (Nyxus::Feature2D::NGTDM_CONTRAST,
                                   make_ngtdm2d_settings (false), 100);
    ASSERT_EQ (NGTDMFeature::n_levels, before);
}
