"""Regenerate gldm_2d_coverage.csv by scanning the 2D GLDM tests. Stdlib only.

    python tests/vetting/audit/scan_gldm_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from the family plan: every `vetted` row in oracle_coverage.csv must be asserted by
an oracle test, that test's oracle must be the one the row names, and `current_test` must name
exactly the files that cover the feature.

Coverage rule: a feature is covered by a test function when its name appears on an ASSERTION line in
that function, or in a golden table that a function loops over while asserting. Comments are
stripped first -- several of them name features they do not assert.

Deliberately NOT counted: a line that merely READS a feature out of the buffer. A
readout-counts-as-coverage rule credits an oracle test with vetting features it never checks --
report_feature_tests.py does exactly that, which is how the 2D morphology gap count came out two
rows short of the real one.

The kind of coverage comes from the function-name suffix, per SPEC 2 naming, so only an
oracle-suffixed function contributes an oracle token.
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
OUT = os.path.join(HERE, "gldm_2d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

SOURCES = [
    "test_2d_gldm_ibsi.h",
    "test_2d_gldm_pyradiomics.h",
    "test_2d_gldm_regression.h",
    "test_2d_gldm_mechanics.h",
    os.path.join("python", "test_2d_gldm_mechanics.py"),
]
# A golden table whose keys are never named in the asserting function's body: the pytest mechanics
# guard loops over GLDM_BACKGROUND_EXCLUDED_REF_VALS, so the features it pins appear only there.
TABLE_OWNER = {
    "GLDM_BACKGROUND_EXCLUDED_REF_VALS": "test_2d_gldm_background_not_counted_mechanics",
}

ORACLE_SUFFIX = {"pyradiomics": "pyradiomics", "ibsi": "ibsi"}

# `inline void` is how the mechanics header declares its case, so the optional inline keyword
# is part of the pattern -- without it that function is invisible and its features look
# uncovered.
FUNC = re.compile(r"^(?:inline\s+)?(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# A module-level helper in the pytest files, e.g. `def _one(...)`. The python tests read the
# dataframe column inside such a helper and assert on the returned scalar, so the feature name never
# appears in the test body -- credit the helper's features to every test function that calls it.
HELPER = re.compile(r"^def\s+(_\w+)\s*\(", re.M)
# No trailing \b after `assert`: this family asserts through per-oracle helpers
# (assert_gldm_feature_pyradiomics, assert_gldm_feature_ibsi, ...), so the call line that names the
# feature is `assert_gldm_feature_<oracle>(Feature2D::X, "X")`, not a bare ASSERT_ macro.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

# All 14 features are deliberately asserted twice, by test_2d_gldm_ibsi.h and
# test_2d_gldm_pyradiomics.h. That is not redundancy: the IBSI consensus values are published to
# three significant figures and fix the DEFINITION (rel=1e-2), while PyRadiomics reproduces Nyxus to
# 1.6e-16 and fixes the DIGITS (rel=1e-9). Dropping either weakens the family - see
# audit/gldm_2d_pyradiomics_vetting_report.md.
DUAL_ORACLE = ("asserted against both oracles by design: IBSI fixes the definition at its published "
               "3-significant-figure precision, pyradiomics fixes the digits at 1.6e-16")

# GLDM_DE is the one feature that does not reach the family's exact tier: calc_DE() takes its
# logarithm through Nyxus::fast_log10, the shared float log2 approximation the whole texture set
# reads, so it lands a measured 1.3e-3 from PyRadiomics and asserts at twice that.
NOTE = {
    "GLDM_DE": ("asserted against both oracles by design: IBSI fixes the definition at its "
                "published 3-significant-figure precision, pyradiomics fixes the digits, at "
                "rel=2.5e-3 rather than the family's 1e-9 because calc_DE() takes its logarithm "
                "through the shared float fast_log10() approximation (measured residual 1.3e-3)"),
}


def feature_names():
    """The family's feature names, longest first so GLDM_SDLGLE wins over GLDM_SDE."""
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
        body = text.split(table, 1)[1].split("{", 1)[1].split("}", 1)[0]
        for m in re.finditer(r'"([A-Z0-9_]+)"', body):
            hits.setdefault(owner, set()).add(m.group(1))
        text = text.replace(body, "")

    helpers = helper_features(text, feat_re)

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
        for name, feats in helpers.items():
            if re.search(r"\b" + re.escape(name) + r"\s*\(", block):
                hits.setdefault(fn, set()).update(feats)
    return hits


def helper_features(text, feat_re):
    """-> {module-level python helper name: features it reads out of the result frame}."""
    out = {}
    marks = [(m.start(), m.group(1)) for m in HELPER.finditer(text)]
    for i, (pos, name) in enumerate(marks):
        block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
        feats = set(feat_re.findall(block))
        if feats:
            out[name] = feats
    return out


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
                if r["dim"] == "2D" and r["family"] == "gldm"]


def render(rows, asserted, oracles, regression, other):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Notes"])
    for r in rows:
        f = r["feature"]
        note = NOTE.get(f, DUAL_ORACLE if len(oracles.get(f, ())) > 1 else "")
        if f in other:
            note += ("; also guarded on the production config by "
                     + ";".join(sorted(other[f])))
        w.writerow(["2D", "gldm", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    note])
    return buf.getvalue()


def unregistered_tests(where):
    """gtest functions that exist but no TEST() in test_all.cc calls - they never run.

    Only the C++ headers are checked; pytest collects the .py functions by name.
    """
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        registered = set(re.findall(r"(test_2d_gldm_\w+)\s*\(\s*\)", fh.read()))
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
        # A drift guard is not a vetting claim, so current_test lists the oracle files only; the
        # regression and mechanics files are reported here, not required in the registry.
        for gap in sorted(files - claimed):
            if gap in ("test_2d_gldm_regression.h", "test_2d_gldm_mechanics.h",
                       "test_2d_gldm_mechanics.py"):
                continue
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
