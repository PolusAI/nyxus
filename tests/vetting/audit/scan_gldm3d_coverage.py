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

AND IT RESOLVES EACH ROW'S `test_name`. The two checks above answer different questions -- which
files cover a feature, and which functions read a table -- and neither looks at the column that says
which assertion the row IS. check_coverage.py looks, but only asks whether the name is *a* gtest
case, so between the three a row could name an unrelated existing case and stay green; that is how
the per-feature rows here came to name TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS, a case asserting
dependence cells on a hand-written unbinned volume rather than the feature on the row's benchmark.
Each name is now resolved through test_all.cc to the function it runs, and that function has to
carry an assertion of the row's own feature, at the row's own kind, in a file current_test names,
and at the row's own config_recipe -- the last because this family has two regression recipes over
the same fourteen features in the same file, so feature, kind and file alone cannot tell them apart.
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

# recipe -> the function that asserts AT that recipe. A row's config_recipe is the configuration its
# numbers were taken at, and the function name is where that configuration lives in the tree, so the
# two have to agree. Feature, kind and file are all identical between the family's two regression
# recipes -- one file, one kind, the same fourteen features at GLDM_GREYDEPTH 64 and 0 -- so without
# this the two sets of rows could be swapped and every other check would stay green.
RECIPE_READER = {
    "gldm3d.pyradiomics_bincount20": re.compile(r"^test_3d_gldm_[a-z0-9]+_pyradiomics$"),
    "gldm3d.regression_ut_phantom": re.compile(r"^test_3d_gldm_[a-z0-9]+_regression$"),
    "gldm3d.regression_ut_phantom_nobinning": re.compile(r"^test_3d_gldm_[a-z0-9]+_nobinning_regression$"),
    "gldm3d.regression_constant_roi": re.compile(r"^test_3d_gldm_constant_roi_regression$"),
}

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

# the scalar tables: the file each lives in, and the suffix its per-feature reader wears. The suffix
# is what attributes readers PER TABLE rather than per file -- two tables now share the regression
# file, one per config, so a file-wide reader set would report every feature as asserted twice and
# would let a key of one table be answered by the other table's reader.
TABLES = {
    "gldm_3d_pyradiomics_ref_vals": ("test_3d_gldm_pyradiomics.h", "_pyradiomics"),
    "gldm_3d_regression_ref_vals": ("test_3d_gldm_regression.h", "_regression"),
    "gldm_3d_regression_nobinning_ref_vals": ("test_3d_gldm_regression.h", "_nobinning_regression"),
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
    """Every key of a golden table against the function named for it and the TEST() that runs it.

    A key nothing reads is where a bad number lives, because no assertion ever evaluates it -- which
    is how this family shipped 3GLDM_LGLE pinned to 3GLDM_SDE's value, off by a factor of 353, under
    a function that asserted 3GLDM_SDE. The key is resolved to the function its own name implies
    rather than to whatever happens to read it, so a missing reader is as loud as a mismatched one.

    The reader's feature token is [a-z0-9]+ with no underscore, which is what keeps the tables apart:
    `_regression` is a suffix of `_nobinning_regression` too, and matching on the suffix alone would
    let each table's readers answer for the other's keys.
    """
    out = []
    registered = set(registered_calls().values())
    for table, (rel, suffix) in TABLES.items():
        text = read(os.path.join(TESTS, rel))
        body = table_body(text, table)
        if body is None:
            out.append(f"{table}: not found in {rel}")
            continue
        keys = sorted(set(re.findall(r'\{\s*"(3GLDM_[A-Z0-9_]+)"\s*,', body)))

        reader_name = re.compile(r"^test_3d_gldm_[a-z0-9]+" + re.escape(suffix) + r"$")
        readers = {}
        for fn, args in re.findall(r"void\s+(test_3d_gldm_\w+)\s*\(\s*\)\s*\{([^}]*)\}", text):
            if EXCLUDE_FROM_KIND.match(fn) or not reader_name.match(fn):
                continue
            m = re.search(r'"(3GLDM_[A-Z0-9_]+)"', args)
            if m:
                readers[fn] = m.group(1)

        for key in keys:
            want = "test_3d_gldm_" + key[len("3GLDM_"):].lower() + suffix
            got = readers.pop(want, None)
            if got is None:
                out.append(f"{table}: {key} is pinned but {want}() does not exist to assert it")
            elif got != key:
                out.append(f"{want}() passes {got} but its name says {key}")
            elif want not in registered:
                out.append(f"{want}() reads a pin but no TEST() in test_all.cc calls it")
        for stray, feat in sorted(readers.items()):
            out.append(f"{table}: {stray}() asserts {feat}, which this table does not pin")
    return out


def test_name_problems(r, f, st, covering, claimed, cases, where):
    """One row's test_name resolved to the assertion it identifies: feature, kind and file.

    check_coverage.py asks only whether the name is *a* gtest case, and the coverage checks ask only
    what covers the feature; between them a row could name an unrelated existing case and stay green.
    Resolving it is what ties one row to one assertion -- the case has to run a function that carries
    an assertion of THIS feature, at THIS row's kind, in a file current_test names. That is what
    rejects a matrix case under a per-feature row: TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS asserts
    dependence cells on a hand-written unbinned volume, so it covers no feature and is not on the
    row's benchmark.
    """
    out = []
    names = [t.strip() for t in (r.get("test_name") or "").split(";") if t.strip()]
    if not names:
        out.append(f"{f} ({st}): no test_name, so the row identifies no assertion")
    for name in names:
        fn = cases.get(name)
        if fn is None:
            out.append(f"{f} ({st}): test_name {name} resolves to no TEST() in test_all.cc")
        elif fn not in covering:
            out.append(f"{f} ({st}): test_name {name} runs {fn}(), which carries no {st} "
                       f"assertion of {f}")
        elif where[fn] not in claimed:
            out.append(f"{f} ({st}): test_name {name} is defined in {where[fn]}, which "
                       f"current_test ({r['current_test'] or 'empty'}) does not name")
        else:
            recipe = r["config_recipe"].strip()
            reader = RECIPE_READER.get(recipe)
            if reader is None:
                out.append(f"{f} ({st}): config_recipe {recipe!r} has no reader in RECIPE_READER, "
                           f"so test_name cannot be checked against the configuration")
            elif not reader.match(fn):
                out.append(f"{f} ({st}): test_name {name} runs {fn}(), which does not assert at "
                           f"config_recipe {recipe}")
    return out


def disagreements(rows, asserted, oracles, regression, where):
    """Each registry row against the tests of ITS OWN KIND - see the module docstring."""
    out = []
    cases = registered_calls()
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
        out += test_name_problems(r, f, st, covering, claimed, cases, where)
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
