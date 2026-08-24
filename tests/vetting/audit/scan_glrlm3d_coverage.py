"""Regenerate glrlm_3d_coverage.csv by scanning the 3D GLRLM tests. Stdlib only.

    python tests/vetting/audit/scan_glrlm3d_coverage.py [--check]

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
OUT = os.path.join(HERE, "glrlm_3d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

SOURCES = [
    "test_3d_glrlm_pyradiomics.h",
    "test_3d_glrlm_regression.h",
    os.path.join("python", "test_nyxus.py"),
]
# A golden table whose keys are never named in the asserting function's body. The 16 per-angle
# assertions name their feature on the assertion line, so only the pytest table needs this: it keys
# a dict by the PyRadiomics feature name and asserts through the Nyxus name, and its dict literal
# sits outside any loop.
TABLE_OWNER = {}

ORACLE_SUFFIX = {"pyradiomics": "pyradiomics"}

FUNC = re.compile(r"^(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# A module-level helper in the pytest files, e.g. `def _fd(label)`. The python tests read the
# dataframe column inside such a helper and assert on the returned scalar, so the feature name never
# appears in the test body -- credit the helper's features to every test function that calls it.
HELPER = re.compile(r"^def\s+(_\w+)\s*\(", re.M)
# No trailing \b after `assert`: this family asserts through per-oracle helpers
# (assert_morphology_feature_regression, assert_caliper_imea, ...), so the call line that names the
# feature is `assert_<something>(fvals, Feature2D::X, "X")`, not a bare ASSERT_ macro.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

NOTE = {
    "3GLRLM_RP": "exceeds its mathematical bound of 1 at positive GLRLM_GREYDEPTH values; in range "
                 "at the binCount binning both tests use",
    "3GLRLM_RP_AVE": "registry read oracle=mirp while the base row read oracle=pyradiomics for the "
                     "same quantity; no mirp run for 3D GLRLM has ever existed in the tree",
    "3GLRLM_RE": "the family's only sum over logarithms; fast_log10 puts it 3.9e-4 from "
                 "PyRadiomics, so it is asserted at rel=5e-3 where the rest are at rel=1e-9",
    "3GLRLM_RE_AVE": "same mirp mislabel as 3GLRLM_RP_AVE, and the same log tolerance as 3GLRLM_RE",
}


def feature_names():
    """The family's feature names, longest first so MAXCHORDS_MAX_ANG wins over MAXCHORDS_MAX."""
    names = {r["feature"] for r in registry_rows()}
    return sorted(names, key=len, reverse=True)


def feature_re(names):
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")


def scan(path, feat_re):
    """-> {test function name: {features it covers}}."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    # 3D tests name features by enum (Feature3D::GLRLM_SRE_AVE) while the registry carries the
    # leading dimension digit (3GLRLM_SRE_AVE). Normalise so one pattern matches both spellings.
    # The _AVE test aliases the enum (`using F = Nyxus::Feature3D;`), so cover `F::` too.
    text = text.replace("Feature3D::", "Feature3D::3")
    text = re.sub(r"\bF::(?=GLRLM_)", "F::3", text)
    hits = {}

    for table, owner in TABLE_OWNER.items():
        if table not in text:
            continue
        body = text.split(table, 1)[1].split("{", 1)[1].split("};", 1)[0]
        for m in re.finditer(r'\{"([A-Z0-9_]+)"', body):
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
        # A table the function builds and then loops over while asserting -- the equivalence tests
        # put their feature pairs in `std::vector<Pair> pairs = { ... };` and assert inside the
        # loop, so the names never appear on an assertion line. Credited only when the function
        # both asserts and range-loops, so a plain lookup table is not mistaken for coverage.
        if re.search(r"for\s*\([^)]*:\s*\w+\s*\)", block):
            for m in re.finditer(r"=\s*\{(.*?)\};", block, re.S):
                hits.setdefault(fn, set()).update(feat_re.findall(m.group(1)))
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
                if r["dim"] == "3D" and r["family"] == "glrlm"]


def render(rows, asserted, oracles, regression, other):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Notes"])
    for r in rows:
        f = r["feature"]
        notes = [NOTE[f]] if f in NOTE else []
        # A function whose name-suffix is neither an oracle nor `regression` contributes coverage but
        # no oracle token, so it would otherwise be invisible in this artifact. Naming it keeps the
        # duplication legible: test_3d_glrlm_compatibility asserts the same _AVE features against the
        # same goldens through the Python API, so the two must be re-tightened together.
        for fn in sorted(other.get(f, ())):
            notes.append(f"also asserted by {fn} (kind is neither oracle nor regression)")
        w.writerow(["3D", "glrlm", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    " | ".join(notes)])
    return buf.getvalue()


def unregistered_tests(where):
    """gtest functions that exist but no TEST() in test_all.cc calls - they never run.

    Only the C++ headers are checked; pytest collects the .py functions by name.
    """
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        registered = set(re.findall(r"(test_3d_glrlm_\w+)\s*\(\s*\)", fh.read()))
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
