"""Regenerate morphology_2d_coverage.csv by scanning the 2D morphology tests. Stdlib only.

    python tests/vetting/audit/scan_morphology_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_2d_morphology_matlab.h",
    "test_2d_morphology_skimage.h",
    "test_2d_morphology_imea.h",
    "test_2d_morphology_cellprofiler.h",
    "test_2d_morphology_fraclac.h",
    "test_2d_morphology_analytic.h",
    "test_2d_morphology_regression.h",
    os.path.join("python", "test_2d_morphology_fraclac.py"),
    os.path.join("python", "test_2d_morphology_hull_invariant.py"),
    os.path.join("python", "test_2d_morphology_invariant.py"),
    # the registry names this file for PERIMETER, DIAMETER_EQUAL_PERIMETER and EDGE_MEAN_INTENSITY;
    # it asserts they are 0 on a contourless speckle mask, which is edge-case coverage, not vetting
    os.path.join("python", "test_nyxus.py"),
    # added with the config-matrix cells: the whole-slide snapshot and the two out-of-core
    # assertions cover contour features, so the scanner has to see them or their rows read as
    # claiming a file that covers nothing
    os.path.join("python", "test_2d_morphology_regression.py"),
    os.path.join("python", "test_2d_ooc_regression.py"),
    os.path.join("python", "test_2d_ooc_invariant.py"),
]

NOTE = {
    "PERCENT_TOUCHING": "neighbor family convention; documented divergence from CellProfiler",
}

# The SPEC 4 token names the reference semantics and the note records the product that produced the
# pins. This family now uses licensed MATLAB rather than the former Octave surrogate.
ORACLE_NOTE = {
    "matlab": ("licensed MATLAB R2026a + Image Processing Toolbox 26.1 "
               "(gen_morphology_matlab.m); all 33 C++ pins verified; see "
               "morphology_2d_matlab_vetting_report.md"),
}


def note(feature, cov):
    parts = [NOTE[feature]] if feature in NOTE else []
    parts += [ORACLE_NOTE[o] for o in sorted(cov.oracles.get(feature, ())) if o in ORACLE_NOTE]
    return "; ".join(parts)


FAMILY = scanlib.Family(
    dim="2D", family="morphology", out="morphology_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"matlab": "matlab", "skimage": "skimage", "imea": "imea",
                   "cellprofiler": "cellprofiler", "fraclac": "fraclac", "analytic": "analytic"},
    notes=note,
    scan_helpers=True,
    # test_2d_ooc_invariant.py names the six ellipse features in a local tuple and EROSIONS_2_VANISH
    # in the comprehension that builds the compared column list, then loops that list while
    # asserting, so no name reaches an assertion line. Same concession the C++ equivalence tests get.
    py_loop_tables=True,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
