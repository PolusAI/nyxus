"""Regenerate radial_2d_coverage.csv by scanning the 2D radial-distribution tests. Stdlib only.

    python tests/vetting/audit/scan_radial_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_2d_radial_regression.h",
    "test_2d_radial_invariant.h",
    "test_2d_radial_mechanics.h",
]

NOTE = {
    "FRAC_AT_D": "8-bin vector, every bin asserted separately. Not CellProfiler-vettable: CP's "
                 "FracAtD is the fraction of the ROI's INTENSITY in a bin, Nyxus' is the fraction "
                 "of its PIXEL COUNT and never reads the image.",
    "MEAN_FRAC": "8-bin vector, every bin asserted separately. Not CellProfiler-vettable: CP's "
                 "MeanFrac is dimensionless (bin mean over ROI mean, ~1), Nyxus returns the raw "
                 "bin mean intensity.",
    "RADIAL_CV": "8-bin vector, every bin asserted separately. Not CellProfiler-vettable: CP takes "
                 "the CV of the eight wedge MEANS over the NON-EMPTY wedges, Nyxus the CV of the "
                 "eight wedge SUMS over all eight.",
}

# Scanned, reported in the artifact's Invariant_Mechanics column, and deliberately NOT expected in
# current_test. Every assertion in this file pins a value that
# audit/radial_2d_cellprofiler_vetting_report.md section 6 shows is wrong (defects 1-3), so a
# correct fix must change all of them. Crediting the file as coverage would make those defects
# acceptance criteria for the three features. The exclusion is declared here rather than applied
# silently, and is checked in both directions: naming the file in current_test is an error too, so
# the decision cannot be reversed by editing only the registry.
UNCREDITED = {
    "test_2d_radial_mechanics.h":
        "known-defect characterization (report section 6 defects 1-3); see not_covered.md A.1",
}

FAMILY = scanlib.Family(
    dim="2D", family="radial", out="radial_2d_coverage.csv",
    sources=SOURCES,
    # No oracle covers this family. Left empty rather than dropped so that adding one is a one-line
    # change and the suffix rule stays visible.
    oracle_suffix={},
    notes=NOTE,
    extra_column="Invariant_Mechanics",
    uncredited=UNCREDITED,
    checks=scanlib.DEFAULT_CHECKS | scanlib.NO_ORACLE_CLAIMED_CHECK,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
