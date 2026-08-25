#pragma once

// Shared fixture for the 2D NGTDM tests: the settings recipes and the IBSI-phantom scaffolding that
// turns a feature id into the four slice values, and their average, ready to compare against a
// golden.
//
// Fixtures only, no reference data (SPEC 6.3.1) -- each file keeps its own table beside the
// assertions that read it, and this header claims nothing about any of them.

// Every include below is named for the symbol this file uses directly rather than left to arrive
// transitively: gtest for the assertion macros, <string>, <unordered_map> and <vector> for the
// helpers' parameters and return types.
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/nyx/feature_settings.h"     // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"           // Feature2D
#include "../src/nyx/roi_cache.h"            // LR
#include "../src/nyx/features/ngtdm.h"       // NGTDMFeature
#include "test_data.h"                       // the IBSI phantom slices, NyxusPixel
#include "test_main_nyxus.h"                 // load_masked_test_roi_data, agrees_gt

// The two configurations the 2D NGTDM tests run on. IBSI mode is the discretisation that makes the
// published consensus values, a mirp run and a PyRadiomics run comparable; Nyxus' default mode bins
// to a fixed grey count instead and no reference reproduces it.
//
// PIXELDISTANCE is deliberately absent: NGTDM never reads it. The neighbourhood is fixed at the
// d=1 8-neighbourhood the IBSI definition uses, which is why setting a pixel distance in a NGTDM
// test would look meaningful and change nothing.
static Fsettings make_ngtdm2d_settings (bool ibsi_mode)
{
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 128;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = ibsi_mode;
    return s;
}

// Featurises the four IBSI digital-phantom slices one at a time, in z order. Each slice is its own
// ROI and yields its own scalar; what the IBSI "2D, averaged" aggregation publishes is their mean.
//
// n_levels is a STATIC on NGTDMFeature, shared by every test in the binary, so it is set and
// restored around the calculation rather than assigned and left. Assigning it and walking away
// makes every later non-IBSI NGTDM test depend on which tests ran before it: at 100 levels this
// fixture's NGTDM_CONTRAST is 3169.93 and at the default 0 it is 6634.50, so the leak would be
// silent and large. In IBSI mode the value is ignored entirely -- ngtdm.cpp forces the grey-binning
// info to 0 -- which the tests confirm rather than assume.
static std::vector<double> ngtdm_2d_phantom_slice_values (const Feature2D& feature_,
                                                          const Fsettings& s,
                                                          int n_levels)
{
    const int saved_n_levels = NGTDMFeature::n_levels;
    NGTDMFeature::n_levels = n_levels;

    const NyxusPixel* intensities[] = { ibsi_phantom_z1_intensity, ibsi_phantom_z2_intensity,
                                        ibsi_phantom_z3_intensity, ibsi_phantom_z4_intensity };
    const NyxusPixel* masks[] = { ibsi_phantom_z1_mask, ibsi_phantom_z2_mask,
                                  ibsi_phantom_z3_mask, ibsi_phantom_z4_mask };
    const size_t counts[] = { sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel),
                              sizeof(ibsi_phantom_z2_mask) / sizeof(NyxusPixel),
                              sizeof(ibsi_phantom_z3_mask) / sizeof(NyxusPixel),
                              sizeof(ibsi_phantom_z4_mask) / sizeof(NyxusPixel) };

    std::vector<double> out;
    for (int z = 0; z < 4; z++)
    {
        LR roidata;
        Nyxus::load_masked_test_roi_data (roidata, intensities[z], masks[z], counts[z]);

        NGTDMFeature f;
        f.calculate (roidata, s);

        roidata.initialize_fvals();
        f.save_value (roidata.fvals);

        out.push_back (roidata.fvals[(int)feature_][0]);
    }

    NGTDMFeature::n_levels = saved_n_levels;
    return out;
}

// The mean of slice values already computed. Callers that assert the per-slice values first pass
// their own vector in, so the mean they check is the mean of the numbers they just checked rather
// than of a second, independent run of the same four featurisations.
static double ngtdm_2d_phantom_slice_mean (const std::vector<double>& per_slice)
{
    double total = 0;
    for (double v : per_slice)
        total += v;
    return total / double (per_slice.size());
}

// Compares the four-slice mean against the caller's table. The table, the settings, the tolerance
// and the trace prefix come from the caller, so each file asserts against its own goldens at its own
// tier and on its own recipe.
//
// A mean is all this can check, and a mean is weaker than the four values behind it: errors in two
// slices that cancel leave it unmoved, and a defect confined to one slice reaches it diluted by
// four. Where the reference exposes per-slice values, as mirp does, the caller asserts those
// separately and derives the mean from that same vector.
void assert_ngtdm_feature_against_golden_values (
    const Feature2D& feature_,
    const std::string& feature_name,
    const std::unordered_map<std::string, double>& feature_reference_values,
    const std::string& trace_prefix,
    double frac_tolerance,
    const Fsettings& s,
    int n_levels)
{
    SCOPED_TRACE (trace_prefix + feature_name);

    // a missing key would otherwise be compared against a default-inserted zero
    ASSERT_TRUE (feature_reference_values.count(feature_name) > 0) << feature_name;

    // Verdict
    const double aveTotal =
        ngtdm_2d_phantom_slice_mean (ngtdm_2d_phantom_slice_values (feature_, s, n_levels));
    ASSERT_TRUE (agrees_gt (aveTotal, feature_reference_values.at(feature_name), frac_tolerance))
        << feature_name;
}
