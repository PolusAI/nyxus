#pragma once

// The fixture the 2D GLCM third-party oracle tests share. gtest is deliberately absent: this file
// builds a ROI and a settings bundle and asserts nothing - the files that include it bring gtest
// in themselves, and SPEC 6.3.1 keeps every golden table with the assertions that read it.

#include <string>
#include <vector>

#include "../src/nyx/feature_settings.h"   // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"         // UserFacingFeatureNames
#include "../src/nyx/features/glcm.h"      // GLCMFeature
#include "../src/nyx/roi_cache.h"          // LR
#include "test_data.h"                     // NyxusPixel
#include "test_main_nyxus.h"               // load_masked_test_roi_data

// img[y,x] = ((y + 2x) % 8) + 1 over 8x8 with a one-pixel background border. Every grey level 1..8
// occurs and every level pair is populated, so the matrix this builds is denser and larger (8x8)
// than the IBSI phantom's, whose in-mask levels are {1,3,4,6}; the IBSI phantom is the fixture of
// test_2d_glcm_ibsi.h and this one is the second, independent configuration.
// tests/vetting/oracles/gen_glcm_{pyradiomics,mirp}.py generate the goldens on this same array.
// Agreement on both fixtures is recorded in tests/vetting/audit/glcm_2d_*_vetting_report.md.
static void make_2d_glcm_dense_phantom (std::vector<NyxusPixel>& intensity, std::vector<NyxusPixel>& mask)
{
    const size_t side = 10, lo = 1, hi = 8;   // 8x8 fixture inside a 10x10 frame

    intensity.clear();
    mask.clear();
    for (size_t y = 0; y < side; y++)
        for (size_t x = 0; x < side; x++)
        {
            const bool in_roi = x >= lo && x <= hi && y >= lo && y <= hi;
            const unsigned int level = in_roi
                ? (unsigned int) (((y - lo) + 2 * (x - lo)) % 8) + 1
                : 0;
            intensity.push_back (NyxusPixel {x, y, level});
            mask.push_back (NyxusPixel {x, y, in_roi ? 1u : 0u});
        }
}

// Recipe glcm.ibsi_identity: the IBSI path, so the matrix is symmetric and the grey levels are the
// image's own - the configuration both reference tools were run at (PyRadiomics binWidth=1 on an
// integer image is the same identity binning; MIRP base_discretisation_method="none").
static Fsettings make_2d_glcm_dense_settings()
{
    Fsettings s;
    s.resize ((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 0;     // needs to be ==0 in the IBSI mode
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = true;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 0;   // needs to be ==0 in the IBSI mode
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;      // distance 1, as both tools were run at
    return s;
}

// The reference tools report one value per feature for the whole angle set. That is what an _AVE
// feature holds, and what averaging the 4 angled values of its base feature produces, so a base
// feature and its _AVE twin are checked against the same golden - the base through the 4 angles,
// the _AVE through the single value the feature aggregated.
static bool is_2d_glcm_ave_feature (const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    return feature_name.size() > ave_suffix.size() &&
        feature_name.compare (feature_name.size() - ave_suffix.size(), ave_suffix.size(), ave_suffix) == 0;
}

// Computes one 2D GLCM feature on the dense phantom, aggregated the way the reference tools
// aggregate. Returns false if the name is not a 2D feature, so a typo in a golden table fails the
// test instead of silently checking nothing.
static bool calc_2d_glcm_dense_feature (const std::string& feature_name, double& value)
{
    auto it_code = Nyxus::UserFacingFeatureNames.find (feature_name);
    if (it_code == Nyxus::UserFacingFeatureNames.end())
        return false;
    const int code = (int) it_code->second;

    std::vector<NyxusPixel> intensity, mask;
    make_2d_glcm_dense_phantom (intensity, mask);

    Fsettings s = make_2d_glcm_dense_settings();
    LR roidata;
    GLCMFeature f;
    GLCMFeature::angles = {0, 45, 90, 135};
    load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
    f.calculate (roidata, s);
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    const auto& fv = roidata.fvals[code];
    value = is_2d_glcm_ave_feature (feature_name)
        ? fv[0]
        : (fv[0] + fv[1] + fv[2] + fv[3]) / 4.;
    return true;
}
