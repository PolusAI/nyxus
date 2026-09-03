"""Regenerate ngtdm_3d_coverage.csv by scanning the 3D NGTDM tests. Stdlib only.

    python tests/vetting/audit/scan_ngtdm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import re
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

# recipe -> the function that asserts AT that recipe. The same five features are read at two
# PyRadiomics configurations that differ only in neighbourhood radius, so feature, kind and oracle
# are identical between them and only the function name says which one a row records.
RECIPE_READER = {
    "ngtdm3d.pyradiomics_binwidth1": re.compile(r"^test_3d_ngtdm_[a-z0-9]+_pyradiomics$"),
    "ngtdm3d.pyradiomics_binwidth1_r2": re.compile(r"^test_3d_ngtdm_[a-z0-9]+_r2_pyradiomics$"),
    "ngtdm3d.regression_ut_phantom": re.compile(r"^test_3d_ngtdm_[a-z0-9]+_regression$"),
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
    recipe_reader=RECIPE_READER,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
