#pragma once

#include "../src/nyx/featureset.h"
#include "test_3d_firstorder_common.h"
#include "test_ref_vals.h"

// Provenance (SPEC 6.4):
//   tool      = MATLAB R2026a
//   functions = sum, iqr, kurtosis, max, mean, mad, median, min, mode, moment,
//               prctile, range, rms, skewness, std, var
//   fixture   = tests/data/nifti/phantoms/ut_{inten,mask57}.nii, label 57
//   config    = default Nyxus 3D first-order settings and float-NIfTI loader domain
//   recipe    = firstorder3d.matlab_native
//   generator = tests/vetting/oracles/gen_firstorder3d_matlab.m
//   report    = tests/vetting/audit/firstorder_3d_matlab_vetting_report.md
static const ref_vals_map<double> firstorder_3d_matlab_ref_vals
{
    { "3COV", 0.29486207043457396 },
    { "3EXCESS_KURTOSIS", -1.2127631603215849 },
    { "3HYPERFLATNESS", 3.8027657005971736 },
    { "3HYPERSKEWNESS", 0.32001332615504319 },
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
    { "3QCOD", 0.25742449134335904 },
    { "3RANGE", 2000.0 },
    { "3ROOT_MEAN_SQUARED", 2067.740503875048 },
    { "3SKEWNESS", 0.074690529125406482 },
    { "3STANDARD_DEVIATION", 584.80556406964115 },
    { "3STANDARD_DEVIATION_BIASED", 584.80449858511895 },
    { "3STANDARD_ERROR", 1.1163339190447454 },
    { "3UNIFORMITY_PIU", 50.59288537549407 },
    { "3VARIANCE", 341997.54776681121 },
    { "3VARIANCE_BIASED", 341996.30156539241 }
};

static double firstorder_3d_matlab_rel_tol(Nyxus::Feature3D feature)
{
    switch (feature)
    {
    // 1%: MATLAB sample percentiles vs Nyxus' 100-bin CDF; worst measured residual is 2.30e-3.
    case Nyxus::Feature3D::INTERQUARTILE_RANGE:
    case Nyxus::Feature3D::QCOD:
    case Nyxus::Feature3D::P01:
    case Nyxus::Feature3D::P10:
    case Nyxus::Feature3D::P25:
    case Nyxus::Feature3D::P75:
    case Nyxus::Feature3D::P90:
    case Nyxus::Feature3D::P99:
        return 1.0e-2;
    default:
        // 0.1%: same native statistic on the same integer voxel vector (SPEC 7).
        return 1.0e-3;
    }
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
    ASSERT_LE(relative_error, firstorder_3d_matlab_rel_tol(static_cast<Nyxus::Feature3D>(fcode)))
        << fname << " actual=" << actual << " MATLAB=" << expected;
}

void test_3d_firstorder_matlab()
{
    std::vector<std::vector<double>> values;
    calculate_3d_firstorder_values(values);
    for (const auto& entry : firstorder_3d_matlab_ref_vals)
        assert_3d_firstorder_value_matlab(values, entry.first);
}
