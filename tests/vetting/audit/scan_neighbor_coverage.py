"""Regenerate neighbor_2d_coverage.csv by scanning the 2D neighbor tests. Stdlib only.

    python tests/vetting/audit/scan_neighbor_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_2d_neighbor_cellprofiler.h",
    "test_2d_neighbor_analytic.h",
    "test_2d_neighbor_regression.h",
    "test_2d_neighbor_invariant.h",
    os.path.join("python", "test_2d_neighbor_invariant.py"),
    # a morphology file, scanned on purpose: test_2d_morphology_regression.h asserts NUM_NEIGHBORS
    # for label 1 while exercising an unrelated family, and that is a real pin this family's
    # current_test names. Only names in this family's feature set are counted, so nothing else in
    # that file leaks in.
    "test_2d_morphology_regression.h",
]

NOTE = {
    "PERCENT_TOUCHING": ("no oracle by definition, not by defect: CellProfiler measures a different "
                         "quantity and diverges on 3/5 ROIs. Drift-pinned, with its construction "
                         "bounds asserted as invariants"),
    "NUM_NEIGHBORS": ("also pinned by test_2d_morphology_regression.h, which asserts it for label 1 "
                      "while exercising an unrelated family"),
}

FAMILY = scanlib.Family(
    dim="2D", family="neighbor", out="neighbor_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"cellprofiler": "cellprofiler", "analytic": "analytic"},
    notes=NOTE,
    extra_column="Invariant",
    # every registered name, not just this family's: SOURCES includes a morphology file for the one
    # NUM_NEIGHBORS pin it carries, and a neighbour-only pattern would report that file's dozen
    # morphology cases as never running when test_all.cc registers all of them
    fn_prefix="test",
    scan_helpers=True,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
