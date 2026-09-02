"""Regenerate intensity_histogram_2d_coverage.csv by scanning the 2D IH tests. Stdlib only.

    python tests/vetting/audit/scan_intensity_histogram_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The checks and the rendering live in scanlib.py.

This family overrides scanlib's collect for two reasons, one sound and one not:

* Sound: its features are matched by shape (`IH_*`, plus `HISTOGRAM`) rather than by the registry's
  name list, and its golden tables key some entries without the `IH_` prefix, which is re-added.
* NOT sound: it also counts a bare READOUT line (`fvals[(int)...]`) as coverage. Every other
  scanner here refuses to, because a readout-counts rule credits an oracle test with vetting
  features it never checks -- that is the defect that made report_feature_tests.py unusable. It is
  preserved here only so this refactor changes no behaviour; removing it is tracked in PR/todo.md
  and needs its own pass, because it will drop coverage this family currently claims.
"""
import os
import re
import sys

import scanlib

SOURCES = [
    "test_2d_intensity_histogram_ibsi.h",
    "test_2d_intensity_histogram_mirp.h",
    "test_2d_intensity_histogram_analytic.h",
    "test_2d_intensity_histogram_regression.h",
    "test_2d_intensity_histogram_mechanics.h",
    os.path.join("python", "test_2d_intensity_histogram_analytic.py"),
]
# a golden table is looped over by one function, and its feature names never appear in that body
TABLE_OWNER = {
    "intensity_histogram_2d_mirp_ref_vals":
        "test_2d_intensity_histogram_family_mirp",
    "intensity_histogram_2d_analytic_phantom_ref_vals":
        "test_2d_intensity_histogram_phantom_analytic",
    "intensity_histogram_2d_ibsi_ref_vals":
        "test_2d_intensity_histogram_dispersion_ibsi",
    "intensity_histogram_2d_analytic_robust_ref_vals":
        "test_2d_intensity_histogram_dispersion_robust_analytic",
    "intensity_histogram_2d_regression_robust_ref_vals":
        "test_2d_intensity_histogram_dispersion_percentile_regression",
}
ORACLE_SUFFIX = {"analytic": "analytic", "mirp": "mirp", "ibsi": "ibsi",
                 "pyradiomics": "pyradiomics", "skimage": "skimage"}

FEATURE = re.compile(r"\b(IH_[A-Z0-9_]+|HISTOGRAM)\b")
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert\b)")
READOUT = re.compile(r"fvals\s*\[\s*\(int\)")

NOTE = {
    "IH_ROBUST_MEAN_IDX": ("no MIRP or IBSI counterpart; exempted by name from the MIRP test's "
                           "coverage invariant and vetted analytically instead"),
    "IH_VARIANCE_VAL": "bin-centre domain, squared scale: VAL = binWidth^2 * IDX",
    "IH_COEFFICIENT_OF_VARIATION_VAL": ("ratio of differently-scaled quantities; not an image of "
                                        "COEFFICIENT_OF_VARIATION_IDX"),
    "HISTOGRAM": ("PixelIntensityFeatures, not the scalar IH_* class; opt-in via *ALL* and not "
                  "IBSI-gated, so it sits outside recipe ih.ibsi_fbn"),
}
for _f in ["IH_P10_VAL", "IH_P90_VAL", "IH_INTERQUANTILE_RANGE_VAL",
           "IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL"]:
    NOTE[_f] = "interpolated histogram percentile; differs from its _IDX partner by definition"
for _f in ["IH_MINIMUM_VAL", "IH_MAXIMUM_VAL", "IH_RANGE_VAL"]:
    NOTE[_f] = "raw intensity domain; not an image of its _IDX partner"
for _f in ["IH_SKEWNESS_VAL", "IH_EXCESS_KURTOSIS_VAL", "IH_ENTROPY_VAL", "IH_UNIFORMITY_VAL"]:
    NOTE[_f] = "domain-invariant: _VAL equals _IDX exactly, so MIRP vets both"
for _f in ["IH_MAX_GRADIENT", "IH_MIN_GRADIENT"]:
    NOTE[_f] = "histogram-gradient magnitude; carries no domain"


def scan(path):
    """-> {test function name: {features it covers}}."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = scanlib.strip_comments(fh.read())
    hits = {}

    for table, owner in TABLE_OWNER.items():
        if table not in text:
            continue
        body = text.split(table, 1)[1].split("{", 1)[1].split("};", 1)[0]
        for m in re.finditer(r'\{"([A-Z0-9_]+)"', body):
            name = m.group(1)
            hits.setdefault(owner, set()).add(
                name if name.startswith(("IH_", "HISTO")) else "IH_" + name)
        text = text.replace(body, "")   # keep the table out of the block scan below

    marks = [(m.start(), m.group(1) or m.group(2)) for m in scanlib.FUNC.finditer(text)]
    for i, (pos, fn) in enumerate(marks):
        block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
        if not ASSERTION.search(block):
            continue
        for line in block.splitlines():
            if ASSERTION.search(line) or READOUT.search(line):
                hits.setdefault(fn, set()).update(FEATURE.findall(line))
        for m in scanlib.LOOP_LIST.finditer(block):
            hits.setdefault(fn, set()).update(FEATURE.findall(m.group(1) or m.group(2) or ""))
    return hits


def collect(fam, feat_re):
    """-> the five coverage maps. feat_re is unused: features are matched by shape, not by name."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    for rel in fam.sources:
        for fn, feats in scan(os.path.join(scanlib.TESTS, rel)).items():
            where[fn] = os.path.basename(rel)
            kind = fn.rsplit("_", 1)[-1]
            for feat in feats:
                if kind in ORACLE_SUFFIX:
                    asserted.setdefault(feat, set()).add(fn)
                    oracles.setdefault(feat, set()).add(ORACLE_SUFFIX[kind])
                elif kind == "regression":
                    regression.setdefault(feat, set()).add(fn)
                else:                   # invariant / mechanics - coverage, never vetting
                    other.setdefault(feat, set()).add(fn)
    return asserted, oracles, regression, other, where


FAMILY = scanlib.Family(
    dim="2D", family="intensity_histogram", out="intensity_histogram_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix=ORACLE_SUFFIX,
    notes=NOTE,
    collect_override=collect,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
