#pragma once

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "../src/nyx/environment.h"
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/slideprops.h"
#include "test_ref_vals.h"

// Provenance (SPEC 6.4):
//   tool      = MATLAB R2026a
//   functions = sum, iqr, kurtosis, max, mean, mad, median, min, mode, prctile,
//               range, rms, skewness, std, var
//   fixture   = tests/data/nifti/phantoms/ut_{inten,mask57}.nii, label 57
//   config    = default Nyxus 3D first-order settings and float-NIfTI loader domain
//   recipe    = firstorder3d.matlab_native
//   generator = tests/vetting/oracles/gen_firstorder3d_matlab.m
static const ref_vals_map<double> firstorder_3d_matlab_ref_vals
{
    { "3INTEGRATED_INTENSITY", 544286216.0 },
    { "3INTERQUARTILE_RANGE", 1018.5 },
    { "3KURTOSIS", 1.7872368396784151 },
    { "3MAX", 3024.0 },
    { "3MEAN", 1983.3190590018658 },
    { "3MEAN_ABSOLUTE_DEVIATION", 507.28947581807233 },
    { "3MEDIAN", 1964.5 },
    { "3MIN", 1024.0 },
    { "3MODE", 1279.0 },
    { "3P01", 1037.0 },
    { "3P10", 1188.0 },
    { "3P25", 1469.0 },
    { "3P75", 2487.5 },
    { "3P90", 2808.0 },
    { "3P99", 3002.0 },
    { "3RANGE", 2000.0 },
    { "3ROOT_MEAN_SQUARED", 2067.740503875048 },
    { "3SKEWNESS", 0.074690529125406482 },
    { "3STANDARD_DEVIATION", 584.80556406964115 },
    { "3STANDARD_DEVIATION_BIASED", 584.80449858511895 },
    { "3VARIANCE", 341997.54776681121 },
    { "3VARIANCE_BIASED", 341996.30156539241 }
};

static double firstorder_3d_matlab_rel_tol(const std::string& fname)
{
    // 1%: MATLAB sample percentiles vs Nyxus' 100-bin CDF; worst measured residual is 2.30e-3.
    if (fname == "3INTERQUARTILE_RANGE" || fname == "3P01" || fname == "3P10" ||
        fname == "3P25" || fname == "3P75" || fname == "3P90" || fname == "3P99")
        return 1.0e-2;

    // 0.1%: same native statistic on the same integer voxel vector (SPEC 7).
    return 1.0e-3;
}

static std::tuple<std::string, std::string, int> get_3d_firstorder_matlab_phantom()
{
    const fs::path tests_dir = fs::path(__FILE__).parent_path();
    return {
        (tests_dir / "data/nifti/phantoms/ut_inten.nii").string(),
        (tests_dir / "data/nifti/phantoms/ut_mask57.nii").string(),
        57
    };
}

static void calculate_3d_firstorder_values_matlab(std::vector<std::vector<double>>& values)
{
    auto [ipath, mpath, label] = get_3d_firstorder_matlab_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    LR& r = e.roiData.at(label);
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_VoxelIntensityFeatures f;
    Fsettings s;
    ASSERT_NO_THROW(f.calculate(r, s, e.dataset));
    f.save_value(r.fvals);
    values = r.fvals;
}

static void assert_3d_firstorder_value_matlab(
    const std::vector<std::vector<double>>& values,
    const std::string& fname,
    int expected_fcode = -1)
{
    SCOPED_TRACE(std::string("MATLAB_ORACLE__") + fname);
    ASSERT_TRUE(firstorder_3d_matlab_ref_vals.count(fname) > 0) << fname;

    FeatureSet features;
    int fcode = -1;
    ASSERT_TRUE(features.find_3D_FeatureByString(fname, fcode)) << fname;
    if (expected_fcode >= 0)
        ASSERT_EQ(expected_fcode, fcode) << fname;
    ASSERT_LT(static_cast<std::size_t>(fcode), values.size()) << fname;
    ASSERT_FALSE(values[fcode].empty()) << fname;

    const double actual = values[fcode][0];
    const double expected = firstorder_3d_matlab_ref_vals.at(fname);
    const double relative_error = std::abs(actual - expected) / std::abs(expected);
    ASSERT_LE(relative_error, firstorder_3d_matlab_rel_tol(fname))
        << fname << " actual=" << actual << " MATLAB=" << expected;
}

static void assert_3d_firstorder_feature_matlab(
    const Nyxus::Feature3D& expected_fcode,
    const std::string& fname)
{
    std::vector<std::vector<double>> values;
    calculate_3d_firstorder_values_matlab(values);
    assert_3d_firstorder_value_matlab(values, fname, static_cast<int>(expected_fcode));
}

void test_3d_firstorder_matlab()
{
    std::vector<std::vector<double>> values;
    calculate_3d_firstorder_values_matlab(values);
    for (const auto& entry : firstorder_3d_matlab_ref_vals)
        assert_3d_firstorder_value_matlab(values, entry.first);
}
