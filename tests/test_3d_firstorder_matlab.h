#pragma once

#include "../src/nyx/featureset.h"
#include "test_3d_firstorder_common.h"
#include "test_ref_vals.h"

// Provenance (SPEC 6.4):
//   tool      = MATLAB R2026a for the definitions; the values below were re-derived on GNU
//               Octave 11.3.0 + statistics when Nyxus began reporting in the volume's own
//               domain, with the previous MATLAB values reproduced to 13 significant digits
//               on the old domain as a control. The percentile values (3P01 to 3P99,
//               3INTERQUARTILE_RANGE, 3QCOD) are computed on Octave as quantile(x, p, 1, 7),
//               the rule prctile Method="inclusive" takes, and match numpy.percentile to the
//               last digit. MATLAB run pending: the generator's prctile Method="inclusive"
//               calls have not yet been run on MATLAB
//   functions = sum, kurtosis, max, mean, mad, median, min, mode, moment,
//               prctile, range, rms, skewness, std, var
//   fixture   = tests/data/nifti/phantoms/ut_{inten,mask57}.nii, label 57
//   config    = default Nyxus 3D first-order settings, values reported in the volume's own
//               intensity domain (the load-time offset undone, its integer truncation kept)
//   recipe    = firstorder3d.matlab_native
//   generator = tests/vetting/oracles/gen_firstorder3d_matlab.m
//   report    = tests/vetting/audit/firstorder_3d_matlab_vetting_report.md
static const ref_vals_map<double> firstorder_3d_matlab_ref_vals
{
    { "3COV", 0.60960486355613208 },
    { "3EXCESS_KURTOSIS", -1.2127631603215918 },
    { "3HYPERFLATNESS", 3.8027657005973312 },
    { "3HYPERSKEWNESS", 0.3200133261551748 },
    { "3INTEGRATED_INTENSITY", 263267848.0 },
    { "3INTERQUARTILE_RANGE", 1018.25 },
    { "3KURTOSIS", 1.7872368396784082 },
    { "3MAX", 2000.0 },
    { "3MEAN", 959.31905900186564 },
    { "3MEAN_ABSOLUTE_DEVIATION", 507.28947581807233 },
    { "3MEDIAN", 940.5 },
    { "3MIN", 0.0 },
    { "3MODE", 255.0 },
    { "3P01", 13.0 },
    { "3P10", 164.0 },
    { "3P25", 445.0 },
    { "3P75", 1463.25 },
    { "3P90", 1784.0 },
    { "3P99", 1978.0 },
    { "3QCOD", 0.53360408751473865 },
    { "3RANGE", 2000.0 },
    { "3ROOT_MEAN_SQUARED", 1123.5165145780536 },
    { "3SKEWNESS", 0.07469052912540243 },
    { "3STANDARD_DEVIATION", 584.80556406962933 },
    { "3STANDARD_DEVIATION_BIASED", 584.80449858510713 },
    { "3STANDARD_ERROR", 1.116333919044723 },
    { "3UNIFORMITY_PIU", 0.0 },
    { "3VARIANCE", 341997.54776679736 },
    { "3VARIANCE_BIASED", 341996.30156537856 }
};

// The voxel range a zero reference's residual is measured against, itself one of the goldens above.
static constexpr double firstorder_3d_matlab_span = 2000.0;

static double firstorder_3d_matlab_rel_tol(Nyxus::Feature3D feature)
{
    switch (feature)
    {
    // prctile Method="inclusive" takes numpy's default 'linear' ranks i/(n-1), the rule Nyxus takes,
    // so both sides interpolate between the same two voxels by the same fraction and agree to
    // rounding. A percentile read off a 100-bin histogram misses these goldens by 1.3e-4 to 1.8e-1.
    case Nyxus::Feature3D::INTERQUARTILE_RANGE:
    case Nyxus::Feature3D::P01:
    case Nyxus::Feature3D::P10:
    case Nyxus::Feature3D::P25:
    case Nyxus::Feature3D::P75:
    case Nyxus::Feature3D::P90:
    case Nyxus::Feature3D::P99:
    case Nyxus::Feature3D::QCOD:
        return 1.0e-9;
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
    const Nyxus::Feature3D feature = static_cast<Nyxus::Feature3D>(fcode);

    // Two scales: a zero reference (3MIN, and 3UNIFORMITY_PIU built from it) has no relative error
    // to take, so its residual lives on the voxel range; everything else is a native statistic over
    // the same voxel vector and keeps its relative comparison.
    double residual, tol;
    if (expected == 0.0)
    {
        residual = std::abs(actual) / firstorder_3d_matlab_span;
        tol = 1.0e-3;
    }
    else
    {
        residual = std::abs(actual - expected) / std::abs(expected);
        tol = firstorder_3d_matlab_rel_tol(feature);
    }
    ASSERT_LE(residual, tol)
        << fname << " actual=" << actual << " MATLAB=" << expected;
}

void test_3d_firstorder_matlab()
{
    std::vector<std::vector<double>> values;
    calculate_3d_firstorder_values(values);
    for (const auto& entry : firstorder_3d_matlab_ref_vals)
        assert_3d_firstorder_value_matlab(values, entry.first);
}
