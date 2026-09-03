"""Regenerate zernike_2d_coverage.csv by scanning the 2D Zernike tests. Stdlib only.

    python tests/vetting/audit/scan_zernike_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
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
    dim="2D", family="zernike", out="zernike_2d_coverage.csv",
    sources=SOURCES,
    # The closed form is the family's only oracle; `analytic` is the SPEC 4 token for it.
    oracle_suffix={"analytic": "analytic"},
    notes=NOTE,
    # the invariant and mechanics guards are neither oracle nor regression; this family names them
    # in a column of their own rather than folding them into the notes
    extra_column="Invariant_Mechanics",
    # this family carries rows that claim no oracle, so the reverse of oracle_mismatch
    # applies: an oracle-suffixed test asserting one of them is a claim gone stale
    checks=scanlib.DEFAULT_CHECKS | scanlib.NO_ORACLE_CLAIMED_CHECK,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
