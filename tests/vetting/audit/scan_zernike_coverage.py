"""Scan the 2D Zernike tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_zernike_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `test_output.csv`. This file is
the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_2d_zernike_analytic.h",
    "test_2d_zernike_regression.h",
    "test_2d_zernike_invariant.h",
    "test_2d_zernike_mechanics.h",
]

NOTE = {
    "ZERNIKE2D": "30 magnitudes, one per (n,m) with n<=9 and n-m even; every entry asserted "
                 "separately. Vetted against the closed form (factorial series for R_nm) at "
                 "ZernikeFeature's own geometry, which tests the Singh & Walia recurrence rather "
                 "than restating it; centrosome's polynomials corroborate to 4.9e-15. NOT "
                 "comparable to CellProfiler's Zernikes -- different disk and normalisation.",
}

FAMILY = scanlib.Family(
    dim="2D", family="zernike",
    sources=SOURCES,
    # The closed form is the family's only oracle; `analytic` is the SPEC 4 token for it.
    oracle_suffix={"analytic": "analytic"},
    notes=NOTE,
    # the invariant and mechanics guards are neither oracle nor regression; this family names them
    # in a column of their own rather than folding them into the notes
    # this family carries rows that claim no oracle, so the reverse of oracle_mismatch
    # applies: an oracle-suffixed test asserting one of them is a claim gone stale
    checks=scanlib.DEFAULT_CHECKS | scanlib.NO_ORACLE_CLAIMED_CHECK,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
