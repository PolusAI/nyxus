"""Regenerate ngtdm_3d_coverage.csv by scanning the 3D NGTDM tests. Stdlib only.

    python tests/vetting/audit/scan_ngtdm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from the family plan: every `vetted` row in oracle_coverage.csv must be asserted by
an oracle test, that test's oracle must be the one the row names, and `current_test` must name the
file holding the assertion the row's `test_name` identifies.

A row describes ONE assertion -- feature x config x reference (SPEC 3) -- so `current_test` names the
file that assertion lives in, not every file that touches the feature. It used to require the whole
covering set, which put the parameterized coverage sweep and the drift guard in one field and left
the row unable to say which of them its recipe, tolerance and benchmark belonged to. The sweep is
still scanned and still reported below; it simply is not the row's assertion.

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
OUT = os.path.join(HERE, "ngtdm_3d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

# test_3d_ngtdm_coverage.h instantiates two parameterized suites over the family's featureset, so
# which features it touches is decided at runtime and cannot be read statically. It establishes no
# vetting either way (SPEC 1), so it is credited to every feature of the family rather than scanned.
SWEEP = "test_3d_ngtdm_coverage.h"

SOURCES = [
    "test_3d_ngtdm_pyradiomics.h",
    "test_3d_ngtdm_regression.h",
    "test_3d_ngtdm_mechanics.h",
    os.path.join("python", "test_nyxus.py"),
]
# A golden table whose keys are never named in the asserting function's body. The pytest case keys
# its dict by the PyRadiomics feature name and asserts through the Nyxus name, but it names the
# Nyxus name on the assertion line itself, so nothing here needs the indirection. The two matrix
# tables are keyed by grey level rather than by feature and cover no feature at all.
TABLE_OWNER = {}

ORACLE_SUFFIX = {"pyradiomics": "pyradiomics"}

FUNC = re.compile(r"^(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# A module-level helper in the pytest files, e.g. `def _fd(label)`. The python tests read the
# dataframe column inside such a helper and assert on the returned scalar, so the feature name never
# appears in the test body -- credit the helper's features to every test function that calls it.
HELPER = re.compile(r"^def\s+(_\w+)\s*\(", re.M)
# No trailing \b after `assert`: this family asserts through per-kind helpers
# (assert_3d_ngtdm_feature_pyradiomics, assert_3d_ngtdm_feature_regression), so the call line that
# names the feature is `assert_<something>(Feature3D::X, "3X")`, not a bare ASSERT_ macro.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

NOTE = {
    "3NGTDM_COARSENESS": "the matrix the five features contract is pinned per grey level as well, "
                         "in test_3d_ngtdm_matrix_pyradiomics",
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
    # 3D tests name features by enum (Feature3D::NGTDM_BUSYNESS) while the registry carries the
    # leading dimension digit (3NGTDM_BUSYNESS). Normalise so one pattern matches both spellings.
    text = text.replace("Feature3D::", "Feature3D::3")
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
                if r["dim"] == "3D" and r["family"] == "ngtdm"]


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
        # duplication legible: test_3d_ngtdm_compatibility asserts the same five features against the
        # same PyRadiomics goldens through the Python API, so the two must be re-tightened together.
        for fn in sorted(other.get(f, ())):
            notes.append(f"also asserted by {fn} (kind is neither oracle nor regression)")
        w.writerow(["3D", "ngtdm", f,
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
        registered = set(re.findall(r"(test_3d_ngtdm_\w+)\s*\(\s*\)", fh.read()))
    return sorted(fn for fn, src in where.items()
                  if src.endswith(".h") and fn not in registered)


def case_to_file(where):
    """-> {gtest case name: the source file defining the function it calls}.

    test_all.cc registers `TEST(SUITE, CASE) { ASSERT_NO_THROW(fn()); }`, and `where` already maps a
    test function to the file that defines it, so the two compose into case -> file. That is what
    lets the check below confirm a row's test_name and its current_test describe the same assertion
    rather than merely both being true of the feature.

    The body is matched as a brace-free span rather than as "up to a closing brace in column 1",
    because the file registers cases in two shapes: a three-line block, and a one-liner whose
    closing brace sits at the end of the line. A pattern anchored on the block shape runs straight
    through every one-liner to the next block's brace, and silently attributes the cases it swallowed
    to whichever registration opened the span.
    """
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    out = {}
    for suite, case, body in re.findall(
            r"TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{([^{}]*)\}", txt):
        for fn in re.findall(r"(test_3d_ngtdm_\w+)\s*\(\s*\)", body):
            if fn in where:
                out[f"{suite}.{case}"] = where[fn]
    return out


def disagreements(rows, asserted, oracles, regression, other, where):
    out = []
    cases = case_to_file(where)
    for r in rows:
        f = r["feature"]
        covering = asserted.get(f, set()) | regression.get(f, set()) | other.get(f, set())
        files = {where[fn] for fn in covering} | {SWEEP}
        claimed = {t for t in r["current_test"].split(";") if t}
        if r["status"] == "vetted" and not asserted.get(f):
            out.append(f"{f}: status=vetted but no oracle test asserts it")
        if r["oracle"] and r["oracle"] not in oracles.get(f, set()):
            out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                       f"{sorted(oracles.get(f, ())) or 'none'}")
        for stale in sorted(claimed - files):
            out.append(f"{f}: current_test names {stale}, which covers nothing for it")

        # The row describes one assertion, so its two identifiers must agree: the file named in
        # current_test is the file the case named in test_name is defined in.
        name = r.get("test_name", "")
        if not name:
            out.append(f"{f}: no test_name, so current_test names an assertion nothing identifies")
        elif name not in cases:
            out.append(f"{f}: test_name {name} resolves to no registered case in test_all.cc")
        elif cases[name] not in claimed:
            out.append(f"{f}: test_name {name} is defined in {cases[name]}, which current_test "
                       f"({r['current_test'] or 'empty'}) does not name")
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
