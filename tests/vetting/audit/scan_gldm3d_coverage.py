"""Regenerate gldm_3d_coverage.csv by scanning the 3D GLDM tests. Stdlib only.

    python tests/vetting/audit/scan_gldm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks below.

Coverage rule: a feature is covered by a test function when its name appears on an ASSERTION line in
that function, or in a golden table that a function loops over while asserting. Comments are
stripped first -- several of them name features they do not assert.

THIS FAMILY HAS TWO ROWS PER FEATURE, so a row is checked against the tests of ITS OWN KIND. A
vetted row is answered only by an oracle file, a regression row only by the snapshot file. Checking
each row against the union of everything covering the feature is what lets a registry that has
stopped meaning anything still go green: with an oracle row and a snapshot row side by side, the
union satisfies both no matter which file actually carries which.

TWO FUNCTIONS ARE EXCLUDED FROM ORACLE ATTRIBUTION BY NAME, deliberately. `test_3d_gldm_dump_*` wear
a kind suffix but assert no reference value -- they print a table for regeneration. Every scanner in
this tree reads the oracle token off the function-name suffix (revet.txt 9), so without this
exclusion a dump would credit all fourteen features with coverage from a case that compares nothing.
They are reported under Notes instead, and no registry row names them.

THE PYTEST IS MAPPED EXPLICITLY, for the opposite reason. `test_3d_gldm_compatibility` in
tests/python/test_nyxus.py asserts the same PyRadiomics goldens at the same recipe as the C++ oracle
file, but its name carries no oracle token, so the suffix rule would file a genuine oracle assertion
under "other". Its siblings in that file share the `_compatibility` shape, so it is mapped here
rather than renamed in isolation.

Beyond the coverage table, `--check` runs the THREE-WAY KEY / READER / REGISTRATION check: every key
of a golden table must be read by a test function, that function's name must agree with the feature
it passes, and a TEST() in test_all.cc must call it. A pinned key that nothing reads is where a bad
number lives, because no assertion ever evaluates it -- which is how this family shipped 3GLDM_LGLE
pinned to 3GLDM_SDE's value, off by a factor of 353, under a function that asserted 3GLDM_SDE.
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
OUT = os.path.join(HERE, "gldm_3d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")
TEST_ALL = os.path.join(TESTS, "test_all.cc")

SOURCES = [
    "test_3d_gldm_pyradiomics.h",
    "test_3d_gldm_regression.h",
    "test_3d_gldm_common.h",
    os.path.join("python", "test_nyxus.py"),
]

# table -> the function that loops it while asserting. The matrix tables are keyed by {level, dep,
# count} triples and carry no feature names, so only the two scalar tables need an owner, and each
# is read by the per-feature helper in its own file rather than by a loop.
TABLE_OWNER = {}

ORACLE_SUFFIX = {"pyradiomics": "pyradiomics"}
# see the module docstring
EXCLUDE_FROM_KIND = re.compile(r"^test_3d_gldm_dump_")
PYTEST_ORACLE = {"test_3d_gldm_compatibility": "pyradiomics"}

# which file answers which kind of registry row
ORACLE_FILES = {"test_3d_gldm_pyradiomics.h", "test_nyxus.py"}
REGRESSION_FILES = {"test_3d_gldm_regression.h"}

FUNC = re.compile(r"^(?:inline\s+)?(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
HELPER = re.compile(r"^def\s+(_\w+)\s*\(", re.M)
# Every top-level def, helper or test. A helper's body ends at the NEXT TOP-LEVEL DEF OF ANY KIND,
# which is what bounds it correctly -- ending it at the next *helper* instead lets the last helper in
# a file swallow every test function below it, so the helper picks up every feature name in the file
# and each test that calls it inherits the lot.
TOPLEVEL_DEF = re.compile(r"^def\s+\w+\s*\(", re.M)
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

# the two scalar tables, and the file each lives in
TABLES = {
    "gldm_3d_pyradiomics_ref_vals": "test_3d_gldm_pyradiomics.h",
    "gldm_3d_regression_ref_vals": "test_3d_gldm_regression.h",
}

NOTE = {
    "3GLDM_DE": ("the family's only sum over logarithms; banded at abs=4e-3 against a measured "
                 "1.7512e-3 for fast_log10(), where the other thirteen hold at abs=1e-9"),
    "3GLDM_LGLE": ("the retired snapshot pinned this to 3GLDM_SDE's value (0.26 against a measured "
                   "0.00073572), under a function that asserted 3GLDM_SDE; both halves regenerated"),
}


def registry_rows():
    with open(REGISTRY, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)
                if r["dim"] == "3D" and r["family"] == "gldm"]


def feature_names():
    return sorted({r["feature"] for r in registry_rows()}, key=len, reverse=True)


def feature_re(names):
    return re.compile(r"(?<![A-Z0-9_])(" + "|".join(re.escape(n) for n in names) + r")\b")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def scan(path, feat_re):
    """-> {test function name: {features it covers}}."""
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), read(path))
    hits = {}

    for table, owner in TABLE_OWNER.items():
        if table not in text:
            continue
        body = text.split(table, 1)[1].split("};", 1)[0]
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
        for name, feats in helpers.items():
            if re.search(r"\b" + re.escape(name) + r"\s*\(", block):
                hits.setdefault(fn, set()).update(feats)
    return {fn: f for fn, f in hits.items() if f}


def helper_features(text, feat_re):
    """-> {module-level python helper name: features it reads out of the result frame}."""
    out = {}
    bounds = [m.start() for m in TOPLEVEL_DEF.finditer(text)]
    for m in HELPER.finditer(text):
        pos = m.start()
        after = [b for b in bounds if b > pos]
        block = text[pos:after[0] if after else len(text)]
        feats = set(feat_re.findall(block))
        if feats:
            out[m.group(1)] = feats
    return out


def collect(feat_re):
    """-> (oracle fns, oracle tokens, regression fns, other-kind fns, fn -> file, dump fns)."""
    asserted, oracles, regression, other, where, dumps = {}, {}, {}, {}, {}, {}
    for rel in SOURCES:
        for fn, feats in scan(os.path.join(TESTS, rel), feat_re).items():
            where[fn] = os.path.basename(rel)
            if EXCLUDE_FROM_KIND.match(fn):
                for feat in feats:
                    dumps.setdefault(feat, set()).add(fn)
                continue
            token = PYTEST_ORACLE.get(fn) or ORACLE_SUFFIX.get(fn.rsplit("_", 1)[-1])
            kind = fn.rsplit("_", 1)[-1]
            for feat in feats:
                if token:
                    asserted.setdefault(feat, set()).add(fn)
                    oracles.setdefault(feat, set()).add(token)
                elif kind == "regression":
                    regression.setdefault(feat, set()).add(fn)
                else:                   # invariant / mechanics - coverage, never vetting
                    other.setdefault(feat, set()).add(fn)
    return asserted, oracles, regression, other, where, dumps


def render(rows, asserted, oracles, regression, other):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Invariant", "Notes"])
    for f in sorted({r["feature"] for r in rows}):
        w.writerow(["3D", "gldm", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    ";".join(sorted(other.get(f, ()))),
                    NOTE.get(f, "")])
    return buf.getvalue()


def registered_calls():
    """-> {gtest case name: the function its body calls}."""
    text = read(TEST_ALL)
    return {f"{s}.{c}": fn for s, c, fn in re.findall(
        r"TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{\s*ASSERT_NO_THROW\s*\(\s*(\w+)\s*\(", text)}


def table_body(text, table):
    m = re.search(re.escape(table) + r"\s*\{", text)
    if not m:
        return None
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[m.end():i - 1]


def key_reader_problems():
    """Three set differences: table keys vs the functions reading them vs the TEST() registrations.

    A key nothing reads is where a bad number lives. Also catches a function whose name disagrees
    with the feature it passes, and a feature asserted twice under two names.
    """
    out = []
    registered = set(registered_calls().values())
    for table, rel in TABLES.items():
        text = read(os.path.join(TESTS, rel))
        body = table_body(text, table)
        if body is None:
            out.append(f"{table}: not found in {rel}")
            continue
        keys = set(re.findall(r'\{\s*"(3GLDM_[A-Z0-9_]+)"\s*,', body))

        readers = {}
        for fn, args in re.findall(r"void\s+(test_3d_gldm_\w+)\s*\(\s*\)\s*\{([^}]*)\}", text):
            m = re.search(r'"(3GLDM_[A-Z0-9_]+)"', args)
            if m:
                readers[fn] = m.group(1)

        for unread in sorted(keys - set(readers.values())):
            out.append(f"{table}: {unread} is pinned but no test function asserts it")
        for fn, feat in sorted(readers.items()):
            want = fn[len("test_3d_gldm_"):].rsplit("_", 1)[0].upper()
            if want != feat[len("3GLDM_"):]:
                out.append(f"{fn}() passes {feat} but its name says 3GLDM_{want}")
            if fn not in registered:
                out.append(f"{fn}() reads a pin but no TEST() in test_all.cc calls it")
        seen = list(readers.values())
        for dup in sorted({v for v in seen if seen.count(v) > 1}):
            out.append(f"{table}: {dup} is asserted by more than one function")
    return out


def disagreements(rows, asserted, oracles, regression, where):
    """Each registry row against the tests of ITS OWN KIND - see the module docstring."""
    out = []
    for r in rows:
        f, st = r["feature"], r["status"].strip()
        claimed = {t for t in r["current_test"].split(";") if t}
        if st == "vetted":
            covering = asserted.get(f, set())
            if not covering:
                out.append(f"{f}: status=vetted but no oracle test asserts it")
            if r["oracle"] and r["oracle"] not in oracles.get(f, set()):
                out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                           f"{sorted(oracles.get(f, ())) or 'none'}")
            allowed = ORACLE_FILES
        elif st == "regression":
            covering = regression.get(f, set())
            if not covering:
                out.append(f"{f}: status=regression but no snapshot test asserts it")
            if r["oracle"].strip():
                out.append(f"{f}: status=regression but names oracle {r['oracle']!r}")
            allowed = REGRESSION_FILES
        else:
            continue
        files = {where[fn] for fn in covering}
        for stale in sorted(claimed - files):
            out.append(f"{f} ({st}): current_test names {stale}, which carries no assertion of "
                       f"this kind for it")
        for gap in sorted(files - claimed):
            out.append(f"{f} ({st}): {gap} asserts it but current_test omits it")
        for bad in sorted(t for t in claimed if "mechanics" in t or "coverage" in t):
            out.append(f"{f} ({st}): current_test names {bad}, which pins no reference value")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and registry disagreements instead of rewriting")
    a = ap.parse_args(argv)

    rows = registry_rows()
    feat_re = feature_re(feature_names())
    asserted, oracles, regression, other, where, dumps = collect(feat_re)
    text = render(rows, asserted, oracles, regression, other)
    problems = disagreements(rows, asserted, oracles, regression, where)
    problems += key_reader_problems()

    if a.check:
        if not os.path.exists(OUT):
            problems.insert(0, f"{os.path.basename(OUT)} is missing; run without --check")
        elif read(OUT) != text:
            problems.insert(0, f"{os.path.basename(OUT)} is stale; rerun without --check")
        for p in problems:
            print("ERROR:", p)
        print(f"checked {len(rows)} rows: "
              f"{'clean' if not problems else str(len(problems)) + ' problem(s)'}")
        return 1 if problems else 0

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT} ({len(rows)} rows, "
          f"{len({r['feature'] for r in rows})} features)")
    if dumps:
        print(f"note: {len(dumps)} feature(s) also appear in dump helpers, excluded from coverage")
    for p in problems:
        print("WARNING:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
