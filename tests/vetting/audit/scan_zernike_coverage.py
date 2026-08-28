"""Regenerate zernike_2d_coverage.csv by scanning the 2D Zernike tests. Stdlib only.

    python tests/vetting/audit/scan_zernike_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from the family plan: every `vetted` row in oracle_coverage.csv must be asserted by
an oracle test, that test's oracle must be the one the row names, and `current_test` must name
exactly the files that cover the feature.

Coverage rule: a feature is covered by a test function when its name appears on an ASSERTION line in
that function, or in a golden table that a function loops over while asserting. Comments are
stripped first -- several of them name features they do not assert.

Deliberately NOT counted: a line that merely READS a feature out of the buffer. A
readout-counts-as-coverage rule credits a test with checking features it never looks at. Every file
in this family streams the feature name into the failure message of the assertions that concern it,
so nothing here needs the weaker rule.

The kind of coverage comes from the function-name suffix, per SPEC 2 naming, so only an
oracle-suffixed function contributes an oracle token. That is what keeps the invariant and mechanics
files -- which assert plenty about ZERNIKE2D -- from reading as vetting claims.
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
OUT = os.path.join(HERE, "zernike_2d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

SOURCES = [
    "test_2d_zernike_analytic.h",
    "test_2d_zernike_regression.h",
    "test_2d_zernike_invariant.h",
    "test_2d_zernike_mechanics.h",
]
# Both tables are read by the function directly below them, and that function names the feature on
# its own assertion lines, so no table needs attributing here.
TABLE_OWNER = {}

# The closed form is the family's only oracle; `analytic` is the SPEC 4 token for it.
ORACLE_SUFFIX = {"analytic": "analytic"}

FUNC = re.compile(r"^(?:inline\s+)?(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# No trailing \b after `assert`: the regression file asserts through
# assert_zernike_vector_feature_regression(...), whose call line is what names the feature.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

NOTE = {
    "ZERNIKE2D": "30 magnitudes, one per (n,m) with n<=9 and n-m even; every entry asserted "
                 "separately. Vetted against the closed form (factorial series for R_nm) at "
                 "ZernikeFeature's own geometry, which tests the Singh & Walia recurrence rather "
                 "than restating it; centrosome's polynomials corroborate to 4.9e-15. NOT "
                 "comparable to CellProfiler's Zernikes -- different disk and normalisation.",
}


def feature_names():
    """The family's feature names, longest first so a name that prefixes another cannot win."""
    names = {r["feature"] for r in registry_rows()}
    return sorted(names, key=len, reverse=True)


def feature_re(names):
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")


def scan(path, feat_re):
    """-> {test function name: {features it covers}}."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    hits = {}

    for table, owner in TABLE_OWNER.items():
        if table not in text:
            continue
        body = text.split(table, 1)[1].split("};", 1)[0]
        for m in re.finditer(r'\{"([A-Z0-9_]+)"', body):
            hits.setdefault(owner, set()).add(m.group(1))
        text = text.replace(body, "")

    marks = [(m.start(), m.group(1) or m.group(2)) for m in FUNC.finditer(text)]
    for i, (pos, fn) in enumerate(marks):
        block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
        if not ASSERTION.search(block):
            continue
        for line in block.splitlines():
            if ASSERTION.search(line):
                hits.setdefault(fn, set()).update(feat_re.findall(line))
        for m in LOOP_LIST.finditer(block):
            hits.setdefault(fn, set()).update(feat_re.findall(m.group(1) or m.group(2) or ""))
    return hits


def collect(feat_re):
    """-> (oracle fns, oracle tokens, regression fns, other-kind fns, fn -> file)."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    for rel in SOURCES:
        for fn, feats in scan(os.path.join(TESTS, rel), feat_re).items():
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
                if r["dim"] == "2D" and r["family"] == "zernike"]


def render(rows, asserted, oracles, regression, other):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Invariant_Mechanics", "Notes"])
    for r in rows:
        f = r["feature"]
        w.writerow(["2D", "zernike", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    ";".join(sorted(other.get(f, ()))),
                    NOTE.get(f, "")])
    return buf.getvalue()


def unregistered_tests(where):
    """gtest functions that exist but no TEST() in test_all.cc calls - they never run."""
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        registered = set(re.findall(r"(test_\w+)\s*\(\s*\)", fh.read()))
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
        if not r["oracle"] and oracles.get(f):
            out.append(f"{f}: registry claims no oracle but {sorted(oracles[f])} test(s) assert it "
                       f"under an oracle-suffixed name")
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

    rows = registry_rows()
    feat_re = feature_re(feature_names())
    asserted, oracles, regression, other, where = collect(feat_re)
    text = render(rows, asserted, oracles, regression, other)
    problems = disagreements(rows, asserted, oracles, regression, other, where)
    problems += [f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs"
                 for fn in unregistered_tests(where)]

    if a.check:
        with open(OUT, newline="", encoding="utf-8") as fh:
            if fh.read() != text:
                problems.insert(0, f"{os.path.basename(OUT)} is stale; rerun without --check")
        for p in problems:
            print("ERROR:", p)
        print(f"checked {len(rows)} rows: "
              f"{'clean' if not problems else str(len(problems)) + ' problem(s)'}")
        return 1 if problems else 0

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT} ({len(rows)} rows)")
    for p in problems:
        print("WARNING:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
