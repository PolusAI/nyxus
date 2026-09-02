"""Regenerate gldm_2d_coverage.csv by scanning the 2D GLDM tests. Stdlib only.

    python tests/vetting/audit/scan_gldm_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_2d_gldm_ibsi.h",
    "test_2d_gldm_pyradiomics.h",
    "test_2d_gldm_regression.h",
    "test_2d_gldm_mechanics.h",
    os.path.join("python", "test_2d_gldm_mechanics.py"),
]

# A golden table whose keys are never named in the asserting function's body: the pytest mechanics
# guard loops over GLDM_BACKGROUND_EXCLUDED_REF_VALS, so the features it pins appear only there.
TABLE_OWNER = {
    "GLDM_BACKGROUND_EXCLUDED_REF_VALS": "test_2d_gldm_background_not_counted_mechanics",
}

# All 14 features are deliberately asserted twice, by test_2d_gldm_ibsi.h and
# test_2d_gldm_pyradiomics.h. That is not redundancy: the IBSI consensus values are published to
# three significant figures and fix the DEFINITION (rel=1e-2), while PyRadiomics reproduces Nyxus to
# 1.6e-16 and fixes the DIGITS (rel=1e-9). Dropping either weakens the family - see
# audit/gldm_2d_pyradiomics_vetting_report.md.
DUAL_ORACLE = ("asserted against both oracles by design: IBSI fixes the definition at its published "
               "3-significant-figure precision, pyradiomics fixes the digits at 1.6e-16")

# GLDM_DE is the one feature that does not reach the family's exact tier: calc_DE() takes its
# logarithm through Nyxus::fast_log10, the shared float log2 approximation the whole texture set
# reads, so it lands a measured 1.3e-3 from PyRadiomics and asserts at twice that.
NOTE = {
    "GLDM_DE": ("asserted against both oracles by design: IBSI fixes the definition at its "
                "published 3-significant-figure precision, pyradiomics fixes the digits, at "
                "rel=2.5e-3 rather than the family's 1e-9 because calc_DE() takes its logarithm "
                "through the shared float fast_log10() approximation (measured residual 1.3e-3)"),
}

FAMILY = scanlib.Family(
    dim="2D", family="gldm", out="gldm_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"pyradiomics": "pyradiomics", "ibsi": "ibsi"},
    notes=scanlib.dual_oracle_notes(NOTE, DUAL_ORACLE),
    table_owner=TABLE_OWNER, table_dialect="python",
    scan_helpers=True,
    other_note="guarded",
    # A drift guard is not a vetting claim, so current_test lists the oracle files only; the
    # regression and mechanics files are reported in the artifact, not required in the registry.
    current_exempt=("test_2d_gldm_regression.h", "test_2d_gldm_mechanics.h",
                    "test_2d_gldm_mechanics.py"),
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
