#pragma once

// The fixture the 2D intensity-histogram oracle tests share. gtest is deliberately absent: this
// file builds a ROI and a settings bundle and asserts nothing - the files that include it bring
// gtest in themselves, and SPEC 6.3.1 keeps every golden table with the assertions that read it.

#include <string>
#include <vector>

#include "../src/nyx/dataset.h"                            // Dataset, SlideProps
#include "../src/nyx/feature_settings.h"                   // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"                         // UserFacingFeatureNames
#include "../src/nyx/features/intensity_histogram.h"       // IntensityHistogramFeatures
#include "../src/nyx/roi_cache.h"                          // LR
#include "test_data.h"                                     // the IBSI phantom slices
#include "test_main_nyxus.h"                               // load_masked_test_roi_data

// The IBSI intensity-histogram consensus is quoted at fixed bin number with 6 bins, and both the
// published goldens and the reference tools are read at that discretisation.
static const int IH_PHANTOM_NBINS = 6;

// Recipe ih.ibsi_fbn. Kept here rather than taken from the regression header so that an oracle file
// never has to include a snapshot file to reach its fixture.
static Fsettings make_2d_intensity_histogram_ibsi_settings (int nbins = IH_PHANTOM_NBINS)
{
    Fsettings s;
    s.resize ((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = -7777.0;
    s[(int)NyxSetting::GREYDEPTH].ival = nbins;
    s[(int)NyxSetting::IBSI].bval = true;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::FPIMG_ACTIVE].bval = false;
    s[(int)NyxSetting::FPIMG_MIN].rval = 0.0;
    s[(int)NyxSetting::FPIMG_MAX].rval = 1.0;
    s[(int)NyxSetting::FPIMG_TARGET_DR].rval = 1e4;
    return s;
}

// All 4 phantom slices as ONE ROI: the intensity histogram is over the whole masked voxel set
// (74 of the 80 pixels), which is what the IBSI consensus and both reference tools are computed on.
static void make_2d_intensity_histogram_phantom (std::vector<NyxusPixel>& intensity,
                                                 std::vector<NyxusPixel>& mask)
{
    intensity.clear();
    mask.clear();
    for (auto* z : {ibsi_phantom_z1_intensity, ibsi_phantom_z2_intensity,
                    ibsi_phantom_z3_intensity, ibsi_phantom_z4_intensity})
        for (size_t i = 0; i < 20; i++)
            intensity.push_back (z[i]);
    for (auto* z : {ibsi_phantom_z1_mask, ibsi_phantom_z2_mask,
                    ibsi_phantom_z3_mask, ibsi_phantom_z4_mask})
        for (size_t i = 0; i < 20; i++)
            mask.push_back (z[i]);
}

// Computes the whole intensity-histogram feature buffer on the phantom once.
static void calc_2d_intensity_histogram_phantom (std::vector<std::vector<double>>& fvals,
                                                 int nbins = IH_PHANTOM_NBINS)
{
    std::vector<NyxusPixel> intensity, mask;
    make_2d_intensity_histogram_phantom (intensity, mask);

    Dataset ds;
    ds.dataset_props.push_back (SlideProps ("", ""));
    LR roidata (1);
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
    roidata.make_nonanisotropic_aabb();

    Fsettings s = make_2d_intensity_histogram_ibsi_settings (nbins);
    IntensityHistogramFeatures f;
    f.calculate (roidata, s, ds);
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);
    fvals = roidata.fvals;
}

// Reads one feature out of that buffer by name. Returns false if the name is not a 2D feature, so
// a typo in a golden table fails the test instead of silently checking nothing.
static bool read_2d_intensity_histogram_feature (const std::vector<std::vector<double>>& fvals,
                                                 const std::string& feature_name, double& value)
{
    auto it_code = Nyxus::UserFacingFeatureNames.find (feature_name);
    if (it_code == Nyxus::UserFacingFeatureNames.end())
        return false;

    const auto& fv = fvals[(int) it_code->second];
    if (fv.empty())
        return false;

    value = fv[0];
    return true;
}
