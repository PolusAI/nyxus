"""Regenerate ngtdm_3d_coverage.csv by scanning the 3D NGTDM tests. Stdlib only.

    python tests/vetting/audit/scan_ngtdm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_3d_ngtdm_pyradiomics.h",
    "test_3d_ngtdm_regression.h",
    "test_3d_ngtdm_mechanics.h",
    os.path.join("python", "test_nyxus.py"),
]

NOTE = {
    "3NGTDM_COARSENESS": "the matrix the five features contract is pinned per grey level as well, "
                         "in test_3d_ngtdm_matrix_pyradiomics",
}

FAMILY = scanlib.Family(
    dim="3D", family="ngtdm", out="ngtdm_3d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"pyradiomics": "pyradiomics"},
    notes=NOTE,
    enum_dim_prefix=True,
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
    checks=scanlib.CORE_CHECKS | scanlib.IDENTITY_CHECKS,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
