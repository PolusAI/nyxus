"""Regenerate intensity_histogram_2d_coverage.csv by scanning the 2D IH tests. Stdlib only.

    python tests/vetting/audit/scan_intensity_histogram_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from PR423-family-plan: every `vetted` row in oracle_coverage.csv must be asserted
by an oracle test, that test's oracle must be the one the row names, and `current_test` must name
exactly the files that cover the feature.

Coverage rule: a feature is covered by a test function when its name appears on an assertion line in
that function, on the line that reads it out of the feature buffer, or in a literal list the
function loops over while asserting. Comments are stripped first - several of them name features
they do not assert, and one quotes an `fvals[(int)...]` subscript. The kind of coverage comes from
the function-name suffix, per SPEC 2 naming, so only an oracle-suffixed function contributes an
oracle token.
"""
import argparse
import csv
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VETTING = os.path.dirname(HERE)
TESTS = os.path.dirname(VETTING)
OUT = os.path.join(HERE, "intensity_histogram_2d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

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
}
ORACLE_SUFFIX = {"analytic": "analytic", "mirp": "mirp", "ibsi": "ibsi",
                 "pyradiomics": "pyradiomics", "skimage": "skimage"}

FEATURE = re.compile(r"\b(IH_[A-Z0-9_]+|HISTOGRAM)\b")
FUNC = re.compile(r"^(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert\b)")
READOUT = re.compile(r"fvals\s*\[\s*\(int\)")
# C++    for (auto fc : { Feature2D::IH_A, Feature2D::IH_B }) { ... ASSERT ... }
# Python for col in ["IH_A", "IH_B"]: assert ...
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

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
        text = fh.read()
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
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

    marks = [(m.start(), m.group(1) or m.group(2)) for m in FUNC.finditer(text)]
    for i, (pos, fn) in enumerate(marks):
        block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
        if not ASSERTION.search(block):
            continue
        for line in block.splitlines():
            if ASSERTION.search(line) or READOUT.search(line):
                hits.setdefault(fn, set()).update(FEATURE.findall(line))
        for m in LOOP_LIST.finditer(block):
            hits.setdefault(fn, set()).update(FEATURE.findall(m.group(1) or m.group(2) or ""))
    return hits


def collect():
    """-> (oracle fns, oracle tokens, regression fns, other-kind fns, fn -> file)."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    for rel in SOURCES:
        for fn, feats in scan(os.path.join(TESTS, rel)).items():
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


def registry_rows():
    with open(REGISTRY, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)
                if r["dim"] == "2D" and r["family"] == "intensity_histogram"]


def render(rows, asserted, oracles, regression):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Notes"])
    for r in rows:
        f = r["feature"]
        w.writerow(["2D", "intensity_histogram", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    NOTE.get(f, "")])
    return buf.getvalue()


def unregistered_tests(where):
    """gtest functions that exist but no TEST() in test_all.cc calls - they never run.

    Only the C++ headers are checked; pytest collects the .py functions by name.
    """
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        registered = set(re.findall(r"(test_2d_intensity_histogram_\w+)\s*\(\s*\)", fh.read()))
    return sorted(fn for fn, src in where.items()
                  if src.endswith(".h") and fn not in registered)


def disagreements(rows, asserted, oracles, regression, other, where):
    out = []
    for r in rows:
        f = r["feature"]
        covering = asserted.get(f, set()) | regression.get(f, set()) | other.get(f, set())
        files = {where[fn] for fn in covering}
        claimed = {t for t in r["current_test"].split(";") if t}
        if r["status"] == "vetted" and not asserted.get(f):
            out.append(f"{f}: status=vetted but no oracle test asserts it")
        if r["oracle"] and r["oracle"] not in oracles.get(f, set()):
            out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                       f"{sorted(oracles.get(f, ())) or 'none'}")
        for stale in sorted(claimed - files):
            out.append(f"{f}: current_test names {stale}, which covers nothing for it")
        for gap in sorted(files - claimed):
            out.append(f"{f}: {gap} covers it but current_test omits it")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and registry disagreements instead of rewriting")
    a = ap.parse_args(argv)

    asserted, oracles, regression, other, where = collect()
    rows = registry_rows()
    text = render(rows, asserted, oracles, regression)
    problems = disagreements(rows, asserted, oracles, regression, other, where)
    problems += [f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs"
                 for fn in unregistered_tests(where)]

    if a.check:
        with open(OUT, newline="", encoding="utf-8") as fh:
            if fh.read() != text:
                problems.insert(0, f"{os.path.basename(OUT)} is stale; rerun without --check")
        for p in problems:
            print("ERROR:", p)
        print(f"checked {len(rows)} rows: {'clean' if not problems else str(len(problems)) + ' problem(s)'}")
        return 1 if problems else 0

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT} ({len(rows)} rows)")
    for p in problems:
        print("WARNING:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
