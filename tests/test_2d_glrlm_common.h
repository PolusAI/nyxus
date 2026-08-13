#pragma once

// The fixture the 2D GLRLM oracle tests share. gtest is deliberately absent: this file builds ROIs
// and a settings bundle and asserts nothing - the files that include it bring gtest in themselves,
// and SPEC 6.3.1 keeps every golden table with the assertions that read it.

#include <string>

#include "../src/nyx/feature_settings.h"   // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"         // Feature2D, UserFacingFeatureNames
#include "../src/nyx/features/glrlm.h"     // GLRLMFeature
#include "../src/nyx/roi_cache.h"          // LR
#include "test_data.h"                     // the IBSI phantom slices
#include "test_main_nyxus.h"               // load_masked_test_roi_data

// Recipe glrlm.ibsi_ng128: the IBSI path, which is where both reference tools were run
// (PyRadiomics binWidth=1 on this integer phantom, MIRP base_discretisation_method="none").
static Fsettings make_2d_glrlm_ibsi_settings()
{
    Fsettings s;
    s.resize ((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 128;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = true;
    return s;
}

// An _AVE feature holds the mean over the 4 directions in slot [0]; its base feature keeps the 4
// directions in slots [0..3]. Both reach the same quantity, so both are checked against the same
// golden - the base by averaging the 4 directions, the _AVE by reading the value it aggregated.
static bool is_2d_glrlm_ave_feature (const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    return feature_name.size() > ave_suffix.size() &&
        feature_name.compare (feature_name.size() - ave_suffix.size(), ave_suffix.size(), ave_suffix) == 0;
}

// Computes one 2D GLRLM feature over the 4 IBSI phantom slices, aggregated the way the IBSI
// consensus and both reference tools report it: mean over the 4 in-slice directions, then over the
// slices. Returns false if the name is not a 2D feature, so a typo in a golden table fails the test
// instead of silently checking nothing.
static bool calc_2d_glrlm_phantom_feature (const std::string& feature_name, double& value)
{
    auto it_code = Nyxus::UserFacingFeatureNames.find (feature_name);
    if (it_code == Nyxus::UserFacingFeatureNames.end())
        return false;
    const int code = (int) it_code->second;
    const bool is_ave = is_2d_glrlm_ave_feature (feature_name);

    const NyxusPixel* intensities[4] = { ibsi_phantom_z1_intensity, ibsi_phantom_z2_intensity,
                                         ibsi_phantom_z3_intensity, ibsi_phantom_z4_intensity };
    const NyxusPixel* masks[4] = { ibsi_phantom_z1_mask, ibsi_phantom_z2_mask,
                                   ibsi_phantom_z3_mask, ibsi_phantom_z4_mask };
    const size_t counts[4] = { sizeof(ibsi_phantom_z1_intensity) / sizeof(NyxusPixel),
                               sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel),
                               sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel),
                               sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel) };

    Fsettings s = make_2d_glrlm_ibsi_settings();
    double total = 0;
    for (int i = 0; i < 4; i++)
    {
        LR roidata;
        GLRLMFeature f;
        load_masked_test_roi_data (roidata, intensities[i], masks[i], counts[i]);
        f.calculate (roidata, s);
        roidata.initialize_fvals();
        f.save_value (roidata.fvals);

        const auto& fv = roidata.fvals[code];
        total += is_ave ? fv[0] : (fv[0] + fv[1] + fv[2] + fv[3]) / 4.;
    }

    value = total / 4.;
    return true;
}
