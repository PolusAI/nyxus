"""Regenerate glcm_3d_coverage.csv by scanning the 3D GLCM tests. Stdlib only.

    python tests/vetting/audit/scan_glcm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_3d_glcm_pyradiomics.h",
    "test_3d_glcm_regression.h",
    os.path.join("python", "test_nyxus.py"),
]

FAMILY = scanlib.Family(
    dim="3D", family="glcm", out="glcm_3d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"pyradiomics": "pyradiomics"},
    enum_dim_prefix=True,
    # the _AVE test aliases the enum (`using F = Nyxus::Feature3D;`), so cover `F::` too
    enum_alias="GLCM",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
