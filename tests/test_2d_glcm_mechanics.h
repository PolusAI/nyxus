#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <typeinfo>

#include "../src/nyx/environment.h"
#include "../src/nyx/feature_settings.h"
#include "../src/nyx/features/glcm.h"
#include "test_2d_glcm_common.h"	// the dense phantom and its settings, load_masked_test_roi_data

// The production path (CLI and Python featurize) builds GLCM settings via
// Environment::compile_feature_settings(), which must initialise the GLCM-specific settings
// (GLCM_OFFSET / GLCM_GREYDEPTH / GLCM_NUMANG) and not just the COMMON ones. A zero GLCM_OFFSET
// makes the co-occurrence shift dx = dy = 0, so every pixel co-occurs with itself -> a purely
// diagonal matrix -> CONTRAST = 0 and CORRELATION = 1 for any image. The oracle tests cannot catch
// that because they set GLCM_OFFSET = 1 explicitly; this one exercises the real default.
inline void test_2d_glcm_bug_offset_default_is_one_mechanics()
{
    Environment e;
    e.set_coarse_gray_depth(64);
    e.compile_feature_settings();

    const Fsettings& s = e.get_feature_settings(typeid(GLCMFeature));

    // co-occurrence distance must default to 1 (IBSI delta = 1), NOT 0
    ASSERT_EQ(STNGS_GLCM_OFFSET(s), 1);
    // the GLCM-specific grey depth and angle count must be initialised, not left at 0
    ASSERT_GT(STNGS_GLCM_GREYDEPTH(s), 0);
    ASSERT_GT(STNGS_GLCM_NUMANG(s), 0);
}

// 2D GLCM bins at its own grey depth, GLCM_GREYDEPTH -- what glcm/greydepth sets -- and not at the
// common GREYDEPTH the other 2D families read. Measured on the dense phantom's eight levels with
// MATLAB binning, where 4 levels merge them and 64 keep them apart: a run at GLCM_GREYDEPTH=4 under a
// common depth of 64 has to equal the run with both at 4, value for value on every feature and
// angle, and differ from the run with both at 64.
static std::vector<std::vector<double>> run_2d_glcm_dense (int greydepth, int glcm_greydepth)
{
    std::vector<NyxusPixel> intensity, mask;
    make_2d_glcm_dense_phantom (intensity, mask);

    Fsettings s = make_2d_glcm_dense_settings();
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = greydepth;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = glcm_greydepth;

    LR roidata;
    GLCMFeature f;
    GLCMFeature::angles = {0, 45, 90, 135};
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
    f.calculate (roidata, s);
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);
    return roidata.fvals;
}

inline void test_2d_glcm_own_greydepth_bins_the_matrix_mechanics()
{
    const auto own = run_2d_glcm_dense (64, 4),
        both4 = run_2d_glcm_dense (4, 4),
        both64 = run_2d_glcm_dense (64, 64);

    bool any_differs = false;
    for (auto fc : GLCMFeature::featureset)
    {
        SCOPED_TRACE ("2D feature code " + std::to_string ((int)fc));
        const auto& a = own[(int)fc];
        ASSERT_EQ (a, both4[(int)fc]);
        any_differs = any_differs || a != both64[(int)fc];
    }
    ASSERT_TRUE (any_differs);
}

// Runs 2D GLCM through extract(), the entry point the trivial-ROI pass calls, which is calculate()
// followed by save_value(), on a w x h ROI that fills its frame, at the default grey depth and the
// given soft-NaN substitute.
template <class Inten>
static std::vector<std::vector<double>> run_2d_glcm_extract (unsigned int w, unsigned int h, Inten inten, double softnan)
{
    std::vector<NyxusPixel> intensity, mask;
    for (unsigned int y = 0; y < h; y++)
        for (unsigned int x = 0; x < w; x++)
        {
            intensity.push_back (NyxusPixel {x, y, (unsigned int) inten (x, y)});
            mask.push_back (NyxusPixel {x, y, 1u});
        }

    Fsettings s = make_2d_glcm_dense_settings();
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 64;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 64;
    s[(int)NyxSetting::SOFTNAN].rval = softnan;

    LR roidata;
    GLCMFeature::symmetric_glcm = false;
    GLCMFeature::angles = {0, 45, 90, 135};
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
    roidata.initialize_fvals();
    GLCMFeature::extract (roidata, s);
    return roidata.fvals;
}

// A ROI of one grey level has no co-occurrence statistics, and every GLCM feature reports the soft-NaN
// substitute (--noval) there: one value per angle for an angled feature, one for an _AVE feature. The
// substitute is -999 so that it cannot be mistaken for a computed 0.
inline void test_2d_glcm_constant_roi_reports_noval_mechanics()
{
    const double noval = -999.;
    const auto fv = run_2d_glcm_extract (4, 4, [](unsigned int, unsigned int) { return 7; }, noval);

    const std::vector<Nyxus::Feature2D> non_angled (GLCMFeature::nonAngledFeatures);
    for (auto fc : GLCMFeature::featureset)
    {
        SCOPED_TRACE ("2D feature code " + std::to_string ((int)fc));
        const bool ave = std::find (non_angled.begin(), non_angled.end(), fc) != non_angled.end();
        const auto& v = fv[(int)fc];
        if (ave)
            ASSERT_FALSE (v.empty());
        else
            ASSERT_EQ (v.size(), GLCMFeature::angles.size());
        for (double x : v)
            ASSERT_EQ (x, noval);
        if (ave)
            ASSERT_EQ (v[0], noval);
    }
}

// The same pass on a ROI of two grey levels computes the features: no angle of any GLCM feature
// reports the substitute, so it is the constant ROI, not the substitute setting, that selects it.
inline void test_2d_glcm_two_level_roi_computes_mechanics()
{
    const double noval = -999.;
    const auto fv = run_2d_glcm_extract (4, 4, [](unsigned int x, unsigned int y) { return (x + y) % 2 ? 7 : 3; }, noval);

    for (auto fc : GLCMFeature::featureset)
    {
        SCOPED_TRACE ("2D feature code " + std::to_string ((int)fc));
        ASSERT_FALSE (fv[(int)fc].empty());
        for (double x : fv[(int)fc])
            ASSERT_NE (x, noval);
    }
}

// GLCM_HOM2_AVE is the mean of the per-angle GLCM_HOM2 values, like every _AVE feature; homogeneity 2
// and the inverse difference moment are the same sum, so it equals GLCM_IDM_AVE as well.
inline void test_2d_glcm_hom2_ave_mechanics()
{
    // looked up by its user-facing name, which is what --features and the output columns go by
    auto it = Nyxus::UserFacingFeatureNames.find ("GLCM_HOM2_AVE");
    ASSERT_TRUE (it != Nyxus::UserFacingFeatureNames.end());
    const int hom2_ave = (int) it->second;
    ASSERT_TRUE (std::find (GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), it->second) != GLCMFeature::featureset.end());

    const auto fv = run_2d_glcm_dense (64, 64);
    const auto& hom2 = fv[(int)Nyxus::Feature2D::GLCM_HOM2];
    ASSERT_EQ (hom2.size(), 4u);
    const double mean = (hom2[0] + hom2[1] + hom2[2] + hom2[3]) / 4.;
    ASSERT_GT (mean, 0.);
    ASSERT_NEAR (fv[hom2_ave][0], mean, mean * 1e-12);
    ASSERT_NEAR (fv[hom2_ave][0], fv[(int)Nyxus::Feature2D::GLCM_IDM_AVE][0], mean * 1e-12);
}

// Information correlation 1 is undefined where both marginals hold one grey level (HX = HY = 0). On
// this ROI the horizontal pairs all join level 7 to level 7, while the ROI itself has two levels: the
// third row is a single pixel of level 3 with no horizontal neighbour. The 0-degree value is the
// soft-NaN substitute rather than NaN.
inline void test_2d_glcm_infomeas1_single_level_reports_noval_mechanics()
{
    const double noval = -999.;
    std::vector<NyxusPixel> intensity, mask;
    for (unsigned int y = 0; y < 3; y++)
        for (unsigned int x = 0; x < 3; x++)
        {
            const bool in_roi = y == 0 || (y == 2 && x == 0);
            intensity.push_back (NyxusPixel {x, y, in_roi ? (y == 0 ? 7u : 3u) : 0u});
            mask.push_back (NyxusPixel {x, y, in_roi ? 1u : 0u});
        }

    Fsettings s = make_2d_glcm_dense_settings();
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 64;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 64;
    s[(int)NyxSetting::SOFTNAN].rval = noval;

    LR roidata;
    GLCMFeature::symmetric_glcm = false;
    GLCMFeature::angles = {0, 45, 90, 135};
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
    roidata.initialize_fvals();
    GLCMFeature::extract (roidata, s);

    // the 0-degree matrix has pairs, so the value comes from the quotient's own guard
    ASSERT_GT (roidata.fvals[(int)Nyxus::Feature2D::GLCM_ASM][0], 0.);
    ASSERT_EQ (roidata.fvals[(int)Nyxus::Feature2D::GLCM_INFOMEAS1][0], noval);
}