#pragma once

// Shared fixture for the 2D GLDZM tests: the settings recipe and the IBSI-phantom scaffolding that
// turns a feature id into the four slice values, and their average, ready to compare against a
// golden.
//
// Fixtures only, no reference data (SPEC 6.3.1) -- each file keeps its own table beside the
// assertions that read it, and this header claims nothing about any of them.

// What this header includes is what it spells: gtest for the assertion macros, <string>,
// <unordered_map> and <vector> for the helpers' parameters and return types, environment.h for
// Fsettings, gldzm.h for GLDZMFeature, test_data.h for the phantom arrays and test_main_nyxus.h for
// the ROI loader and agrees_gt (which is also where roi_cache.h arrives). The files including this
// one repeat none of them.
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/nyx/environment.h"
#include "../src/nyx/features/gldzm.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// The settings every 2D GLDZM test runs on. IBSI mode is the discretisation that makes the
// published consensus values and a mirp run comparable; the matrix check exercises the same
// settings so the matrix and the features it feeds are built the same way.
static Fsettings make_gldzm2d_settings (bool ibsi_mode)
{
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 128;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = ibsi_mode;
    return s;
}

// Featurises the four IBSI digital-phantom slices one at a time, in z order. Each slice is its own
// ROI and yields its own scalar; what the IBSI "2D, averaged" aggregation publishes is their mean.
static std::vector<double> gldzm_2d_phantom_slice_values (const Feature2D& feature_)
{
    const Fsettings s = make_gldzm2d_settings (true);

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

        GLDZMFeature f;
        f.calculate (roidata, s);

        roidata.initialize_fvals();
        f.save_value (roidata.fvals);

        out.push_back (roidata.fvals[(int)feature_][0]);
    }
    return out;
}

// Compares the four-slice mean against the caller's table. The table, the tolerance and the trace
// prefix come from the caller, so each file asserts against its own goldens at its own tier.
//
// A mean is all this can check, and a mean is weaker than the four values behind it: errors in two
// slices that cancel leave it unmoved, and a defect confined to one slice reaches it diluted by
// four. Where the reference exposes per-slice values, as mirp does, the caller asserts those
// separately.
void assert_gldzm_feature_against_golden_values (
    const Feature2D& feature_,
    const std::string& feature_name,
    const std::unordered_map<std::string, double>& feature_reference_values,
    const std::string& trace_prefix,
    double frac_tolerance)
{
    SCOPED_TRACE (trace_prefix + feature_name);

    // a missing key would otherwise be compared against a default-inserted zero
    ASSERT_TRUE (feature_reference_values.count(feature_name) > 0) << feature_name;

    const std::vector<double> per_slice = gldzm_2d_phantom_slice_values (feature_);

    double total = 0;
    for (double v : per_slice)
        total += v;

    // Verdict
    const double aveTotal = total / double(per_slice.size());
    ASSERT_TRUE (agrees_gt (aveTotal, feature_reference_values.at(feature_name), frac_tolerance))
        << feature_name;
}
