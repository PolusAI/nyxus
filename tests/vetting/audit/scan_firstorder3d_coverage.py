"""Scan the 3D first-order tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_firstorder3d_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_3d_firstorder_matlab.h",
    "test_3d_firstorder_pyradiomics.h",
    "test_3d_firstorder_regression.h",
]

MATLAB = "MATLAB R2026a "

NOTE = {
    "3COV": MATLAB + "std/mean",
    "3COVERED_IMAGE_INTENSITY_RANGE": "Nyxus-specific slide/ROI ratio; snapshot only",
    "3ENERGY": "native PyRadiomics Energy",
    "3ENTROPY": "native PyRadiomics Entropy at binCount=20",
    "3EXCESS_KURTOSIS": MATLAB + "kurtosis-3",
    "3HYPERFLATNESS": MATLAB + "sixth central moment standardized by std",
    "3HYPERSKEWNESS": MATLAB + "fifth central moment standardized by std",
    "3INTEGRATED_INTENSITY": MATLAB + "sum",
    "3INTERQUARTILE_RANGE": "two oracle recipes; percentile-estimator residuals use rel=1e-2",
    "3KURTOSIS": "native kurtosis in both tools",
    "3MAX": "native maximum in both tools",
    "3MEAN": "native mean in both tools",
    "3MEAN_ABSOLUTE_DEVIATION": "native mean absolute deviation in both tools",
    "3MEDIAN": "native median in both tools",
    "3MEDIAN_ABSOLUTE_DEVIATION": "MATLAB median deviation differs from Nyxus mean deviation about "
                                  "the median",
    "3MIN": "native minimum in both tools",
    "3MODE": MATLAB + "mode",
    "3P01": MATLAB + "prctile midpoint; rel=1e-2 covers the Nyxus CDF estimator",
    "3P10": "two native percentile oracles at separate recipes",
    "3P25": MATLAB + "prctile midpoint",
    "3P75": MATLAB + "prctile midpoint",
    "3P90": "two native percentile oracles at separate recipes",
    "3P99": MATLAB + "prctile midpoint",
    "3QCOD": "defining ratio over MATLAB R2026a percentile built-ins",
    "3RANGE": "native range in both tools",
    "3ROBUST_MEAN": "MATLAB trimmean semantics differ from Nyxus histogram-threshold selection",
    "3ROBUST_MEAN_ABSOLUTE_DEVIATION": "native PyRadiomics RobustMeanAbsoluteDeviation",
    "3ROOT_MEAN_SQUARED": "native RMS in both tools",
    "3SKEWNESS": "native skewness in both tools",
    "3STANDARD_DEVIATION": MATLAB + "std with N-1 normalization",
    "3STANDARD_DEVIATION_BIASED": MATLAB + "std with N normalization",
    "3STANDARD_ERROR": "defining std/sqrt(n) over MATLAB R2026a built-in",
    "3UNIFORMITY": "native PyRadiomics Uniformity at binCount=20",
    "3UNIFORMITY_PIU": "defining PIU expression over MATLAB R2026a min/max",
    "3VARIANCE": "MATLAB matches sample variance; PyRadiomics population variance agrees within "
                 "rel=1e-3",
    "3VARIANCE_BIASED": MATLAB + "var with N normalization",
}

FAMILY = scanlib.Family(
    dim="3D", family="firstorder",
    sources=SOURCES,
    oracle_suffix={"matlab": "matlab", "pyradiomics": "pyradiomics"},
    notes=NOTE,
    enum_dim_prefix=True,
    # test_3d_firstorder_matlab loops its golden table, whose entries are written `{ "3COV", ... }`
    table_owner={"firstorder_3d_matlab_ref_vals": "test_3d_firstorder_matlab"},
    # a row names the one case its assertion runs in, so the identity tier applies
    checks=scanlib.CORE_CHECKS | scanlib.ORACLE_FILE_CHECK | scanlib.IDENTITY_CHECKS,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
