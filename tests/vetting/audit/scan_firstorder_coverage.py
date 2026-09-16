"""Scan the 2D first-order tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_firstorder_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_2d_firstorder_ibsi.h",
    "test_2d_firstorder_matlab.h",
    "test_2d_firstorder_pyradiomics.h",
    "test_2d_firstorder_regression.h",
    os.path.join("python", "test_nyxus.py"),
    os.path.join("python", "test_2d_hu_ct_small_pydicom.py"),
    os.path.join("python", "test_2d_morphology_fraclac.py"),
    os.path.join("python", "test_2d_intensity_histogram_analytic.py"),
]

# The registry records these as vetted against MATLAB by the Octave harness (source=tracker), which
# lives outside the test tree, so no in-tree case carries the matlab token for them. What the tree
# asserts is the regression drift guard. report_features.py allowlists the same six for the same reason.
OCTAVE_PROMOTED = {"P01", "P25", "P75", "P99", "QCOD", "ROBUST_MEAN"}
OCTAVE = ("the registry's matlab vetting comes from the Octave harness outside the test tree; the "
          "in-tree assertion is the regression drift guard")

PERCENTILE = "pyradiomics agrees to rel={} (percentile interpolation convention)"

NOTE = {
    "COVERED_IMAGE_INTENSITY_RANGE": "the ROI range as a fraction of the slide's dynamic range, so it "
                                     "needs a slide-level fixture (slide_idx=0; min=0, max=65535)",
    "ENERGY": "pyradiomics agrees exactly at the binCount=64 recipe; matlab is the vetting oracle",
    "ENTROPY": "the regression drift guard runs at GREYDEPTH=20 and the pyradiomics oracle at "
               "GREYDEPTH=64: two recipes asserting one feature",
    "EXCESS_KURTOSIS": "asserted against IBSI by test_2d_firstorder_kurtosis_ibsi, whose name spells "
                       "KURTOSIS",
    "INTERQUARTILE_RANGE": "the IBSI case is test_2d_firstorder_interquartile_ibsi; "
                           + PERCENTILE.format("2.34e-2"),
    "KURTOSIS": "PyRadiomics Kurtosis is the non-excess (+3) convention and matches KURTOSIS "
                "directly; EXCESS_KURTOSIS is the -3 one",
    "MIN": "the IBSI case is test_2d_firstorder_minimum_ibsi",
    "P10": PERCENTILE.format("1.36e-2"),
    "P90": "IBSI documents P90 as 4-4.2 by implementation, so the IBSI assertion rounds before "
           "comparing; " + PERCENTILE.format("1.63e-3"),
    "ROBUST_MEAN_ABSOLUTE_DEVIATION": "asserted against pyradiomics twice on one golden, by the family "
                                      "table and by its own case",
    "UNIFORMITY": "asserted at two recipes: matlab at GREYDEPTH=20 with IBSI off, pyradiomics at "
                  "GREYDEPTH=64",
    "VARIANCE": "PyRadiomics Variance is the /N population variance and Nyxus VARIANCE the /(N-1) "
                "sample one, so they differ by the Bessel factor; VARIANCE_BIASED is the /N "
                "counterpart",
}
NOTE.update({f: OCTAVE for f in OCTAVE_PROMOTED})

SHARED_ORACLE_CHECKS = frozenset({"vetted_no_oracle", "oracle_mismatch"})


def oracle_checks(fam, cov):
    """The two shared oracle checks, for every row but the Octave-promoted ones."""
    out = []
    for r in cov.rows:
        f = r["feature"]
        if f in OCTAVE_PROMOTED:
            continue
        if r["status"] == "vetted" and not cov.asserted.get(f):
            out.append(f"{f}: status=vetted but no oracle test asserts it")
        if r["oracle"] and r["oracle"] not in cov.oracles.get(f, set()):
            out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                       f"{sorted(cov.oracles.get(f, ())) or 'none'}")
    return out


FAMILY = scanlib.Family(
    dim="2D", family="firstorder",
    sources=SOURCES,
    oracle_suffix={"ibsi": "ibsi", "matlab": "matlab", "pyradiomics": "pyradiomics"},
    notes=NOTE,
    # test_2d_firstorder_pyradiomics loops a golden table keyed by feature name
    table_owner={"firstorder_2d_pyradiomics_ref_vals": "test_2d_firstorder_pyradiomics"},
    checks=scanlib.DEFAULT_CHECKS - SHARED_ORACLE_CHECKS,
    extra_problems=oracle_checks,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
