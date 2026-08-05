#pragma once

#include <gtest/gtest.h>
#include <unordered_map>
#include <string>

// TAXONOMY: oracle=pyradiomics (SPEC §2/§6).
// Split out of test_glcm_regression.h: keys that were third-party vetted vs PyRadiomics v3.0.1
// on the matlab-binned / difference-invariant subset (see provenance comments below).

// ORACLE-VETTED 2026-07 (corrected 2026-07-09): the 10 keys below were run against a third-party
// symmetric-matrix oracle (PyRadiomics v3.0.1) on the *same* per-slice matlab-binned phantom images,
// aggregated the same way (mean over 4 slices x 4 angles). Nine of them depend only on the grey-level
// DIFFERENCE p_{x-y} / |i-j| (CONTRAST/DIFAVE/DIS/DIFENTRO/DIFVAR/ID/HOM1/IDM/IV); SUMENTROPY is the
// dimensionless entropy of the SUM distribution p_{x+y}. Both kinds are invariant to matrix
// symmetrization and to a level RELABELING (origin shift) -- and this phantom's binning relabels
// levels without RESCALING them, so they coincide with the symmetric-matrix oracle -> PyRadiomics
// agrees within the test's 1% tolerance. (Difference-based features are NOT invariant to level
// *scaling*; that only holds here because the binning does not rescale.) These are VETTED against an
// external definition, not merely pinned.
// CORRECTION (2026-07-09, PR #356 review): ACOR/IDN/IDMN/SUMAVERAGE were previously listed here as
// oracle-vetted -- that was WRONG for this ibsi=False / matlab-binning config. They depend on the
// absolute grey-level values / Ng, which matlab binning re-maps, so under this config they diverge
// from PyRadiomics by up to ~43% (ACOR; measured on a dense 8-level phantom: Nyxus ibsi=False ACOR
// 29.25 vs oracle 20.51). They were therefore MOVED to the unvetted snapshot below. They ARE
// genuinely third-party-vetted on the IBSI path (symmetric matrix, identity binning), where Nyxus
// ibsi=True == PyRadiomics exactly (ACOR 20.512755, SUMAVERAGE 9.020408, IDN 0.779479, IDMN
// 0.887342) -- covered by the dense-phantom oracle test in tests/python/test_glcm_oracle.py.
static std::unordered_map<std::string, double> oracle_pyradiomics_glcm_feature_golden_values
{
    {"GLCM_CONTRAST", 1.4448130208333334e+03},
    {"GLCM_DIFAVE", 23.6493},
    {"GLCM_DIFENTRO", 1.44004},
    {"GLCM_DIFVAR", 801.208},
    {"GLCM_DIS", 23.6493},
    {"GLCM_HOM1", 0.580526},
    {"GLCM_ID", 0.580526},
    {"GLCM_IDM", 0.572168},
    {"GLCM_IV", 0.000206466},
    {"GLCM_SUMENTROPY", 1.61957}
};

// True iff this key is a PyRadiomics-vetted golden (not a regression pin).
static inline bool glcm_pyradiomics_has(const std::string& key)
{
	return oracle_pyradiomics_glcm_feature_golden_values.count(key) > 0;
}

static inline double glcm_pyradiomics_golden(const std::string& key)
{
	return oracle_pyradiomics_glcm_feature_golden_values.at(key);
}
