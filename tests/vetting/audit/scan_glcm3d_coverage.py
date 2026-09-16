"""Scan the 3D GLCM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_glcm3d_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.
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
    dim="3D", family="glcm",
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
