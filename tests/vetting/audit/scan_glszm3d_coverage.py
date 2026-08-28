"""Regenerate glszm_3d_coverage.csv by scanning the 3D GLSZM tests. Stdlib only.

    python tests/vetting/audit/scan_glszm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from the family plan: every `vetted` row in oracle_coverage.csv must be asserted by
an oracle test, that test's oracle must be the one the row names, and the row's two identifiers --
`test_name` and `current_test` -- must describe the same assertion.

A row describes ONE assertion -- feature x config x reference (SPEC 3) -- so `current_test` names the
file that assertion lives in, and nothing else. It used to require the whole covering set, which put
the parameterized coverage sweep and the drift guard in one field and left the row unable to say
which of them its recipe, tolerance and benchmark belonged to. The sweep has since been retired; the
rule outlives it, because the drift guard and the oracle tests conflate the same way.

That is checked in BOTH directions, because a one-way check is how the conflation survived: a row
whose `current_test` held the right oracle file plus three unrelated ones passed a rule that only
asked whether the right file was present. So `current_test` must be exactly the file the row's
`test_name` is defined in -- no supersets -- and three further checks say the row is about the
assertion it names rather than merely near it:

  * the KIND of the case matches the row. A `vetted` row must name a case whose function carries an
    oracle suffix, and that suffix must be the oracle the row claims; a `regression` row must name a
    `_regression` case and claim no oracle.
  * one case asserts one configuration, so two rows for the same feature naming the same `test_name`
    must carry the same `config_recipe`. This is the check that the four-file `current_test` was
    hiding: the -20 oracle and the +64 regression were one row.
  * an assertion is recorded once, so (feature, config_recipe, oracle) is unique, and a row that
    names no recipe or no tolerance is not saying which assertion it is.

Those five walk the registry, so they can only judge rows that exist -- a whole case can assert
sixteen features at a configuration of its own and stay invisible. So the walk also runs the other
way: every gtest case whose function asserts a feature value must be named by a row, because a row
is what carries the recipe, the band and the benchmark. Mechanics and invariant cases are exempt,
carrying no row anywhere in the registry, and the `_dump_` regenerators fall out on their own since
they print rather than compare.

Coverage rule: a feature is covered by a test function when its name appears on an ASSERTION line in
that function, in a golden table that a function loops over while asserting, or -- naming nothing at
all -- when the function range-loops the calculator's own featureset, which asserts the whole family.
Comments are stripped first -- several of them name features they do not assert.

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
OUT = os.path.join(HERE, "glszm_3d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

SOURCES = [
    "test_3d_glszm_pyradiomics.h",
    "test_3d_glszm_regression.h",
    "test_3d_glszm_mechanics.h",
    os.path.join("python", "test_nyxus.py"),
]
# A golden table whose keys are never named in the asserting function's body. Two cases assert their
# sixteen features by looping their table and resolving each name at runtime, so the names appear
# only in the table: the row is credited to the function that loops it. The pytest case keys its dict
# by the PyRadiomics feature name and asserts through the Nyxus name, but it names the Nyxus name on
# the assertion line itself, so nothing here needs the indirection; the three matrix tables are keyed
# by grey level rather than by feature and cover no feature at all.
TABLE_OWNER = {
    "glszm_3d_pyradiomics_gapped_ref_vals": "test_3d_glszm_ibsi_gapped_pyradiomics",
    "glszm_3d_regression_nobinning_ref_vals": "test_3d_glszm_default_greydepth_regression",
}

ORACLE_SUFFIX = {"pyradiomics": "pyradiomics"}

FUNC = re.compile(r"^(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# A module-level helper in the pytest files, e.g. `def _fd(label)`. The python tests read the
# dataframe column inside such a helper and assert on the returned scalar, so the feature name never
# appears in the test body -- credit the helper's features to every test function that calls it.
HELPER = re.compile(r"^def\s+(_\w+)\s*\(", re.M)
# No trailing \b after `assert`: this family asserts through per-kind helpers
# (assert_3d_glszm_feature_pyradiomics, assert_3d_glszm_feature_regression), so the call line that
# names the feature is `assert_<something>(Feature3D::X, "3X")`, not a bare ASSERT_ macro.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)
# `for (auto fc : D3_GLSZM_feature::featureset)` -- a function that range-loops the calculator's own
# featureset asserts every feature of the family while naming none of them, so no pattern above sees
# it. It is not a table this file could own either: the list comes from the featureset at runtime.
FEATURESET_LOOP = re.compile(r"for\s*\([^)]*:\s*D3_GLSZM_feature::featureset\s*\)")

# Every feature of this family is a contraction of one size-zone matrix, so the note belongs on all
# sixteen rows rather than on a representative one.
FAMILY_NOTE = ("the size-zone matrix all sixteen features contract is pinned cell by cell as well, "
               "read off the feature object calculate() filled, in test_3d_glszm_matrix_pyradiomics")


def feature_names():
    """The family's feature names, longest first so MAXCHORDS_MAX_ANG wins over MAXCHORDS_MAX."""
    names = {r["feature"] for r in registry_rows()}
    return sorted(names, key=len, reverse=True)


def feature_re(names):
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")


def scan(path, feat_re, all_names):
    """-> {test function name: {features it covers}}."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    # 3D tests name features by enum (Feature3D::GLSZM_SAE) while the registry carries the
    # leading dimension digit (3GLSZM_SAE). Normalise so one pattern matches both spellings.
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
        # Asserting over the featureset covers the whole family, which is stronger than naming its
        # members: a feature added to the calculator is covered the day it is added.
        if FEATURESET_LOOP.search(block):
            hits.setdefault(fn, set()).update(all_names)
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


def collect(feat_re, all_names):
    """-> (oracle fns, oracle tokens, regression fns, other-kind fns, fn -> file)."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    for rel in SOURCES:
        for fn, feats in scan(os.path.join(TESTS, rel), feat_re, all_names).items():
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
                if r["dim"] == "3D" and r["family"] == "glszm"]


def render(rows, asserted, oracles, regression, other):
    """The artifact is a feature -> test map, so it carries one line per feature.

    The registry carries one line per ASSERTION, and a feature has several: two oracle recipes and
    two drift guards here. Rendering it row for row would repeat every feature four times with the
    same test lists, which reads as four findings rather than one.
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Notes"])
    seen = set()
    for r in rows:
        f = r["feature"]
        if f in seen:
            continue
        seen.add(f)
        notes = [FAMILY_NOTE]
        # A function whose name-suffix is neither an oracle nor `regression` contributes coverage but
        # no oracle token, so it would otherwise be invisible in this artifact. Naming it keeps the
        # duplication legible: test_3d_glszm_compatibility asserts the same sixteen features against the
        # same PyRadiomics goldens through the Python API, so the two must be re-tightened together.
        for fn in sorted(other.get(f, ())):
            notes.append(f"also asserted by {fn} (kind is neither oracle nor regression)")
        w.writerow(["3D", "glszm", f,
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
        registered = set(re.findall(r"(test_3d_glszm_\w+)\s*\(\s*\)", fh.read()))
    return sorted(fn for fn, src in where.items()
                  if src.endswith(".h") and fn not in registered)


def unregistered_assertions(rows, asserted, regression, where):
    """gtest cases that assert feature values but that no registry row names.

    `disagreements()` walks the registry, so it can only validate rows that already exist: a case can
    be added, assert sixteen features at a configuration of its own, and stay invisible to `--check`
    because nothing points at it. This is the other direction -- a row is what carries an assertion's
    recipe, its band and its benchmark, so a case that pins feature values and has no row is an
    assertion the registry does not know about.

    Only the kinds that earn a row are considered. An oracle case is a `vetted` row and a regression
    case is a `regression` row; mechanics and invariant cases assert a behaviour rather than a value
    and carry no row anywhere in the registry. The gate is the function actually asserting a feature,
    which is also what keeps the `_dump_` regenerators out: they print, they do not compare.
    """
    named = {r["test_name"].strip() for r in rows if r["test_name"].strip()}
    asserting = {fn for fns in list(asserted.values()) + list(regression.values()) for fn in fns}
    return sorted(
        f"{case}: {fn} asserts feature values but no registry row names this case, so nothing "
        f"records which configuration it pins"
        for case, (fn, _src) in case_to_fn(where).items()
        if fn in asserting and case not in named)


def case_to_fn(where):
    """-> {gtest case name: (the function it calls, the source file defining that function)}.

    test_all.cc registers `TEST(SUITE, CASE) { ASSERT_NO_THROW(fn()); }`, and `where` already maps a
    test function to the file that defines it, so the two compose into case -> (fn, file). That is
    what lets the checks below confirm a row's test_name and its current_test describe the same
    assertion rather than merely both being true of the feature -- and, through the function's
    name-suffix, that the assertion is of the kind the row claims.
    """
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    out = {}
    for suite, case, body in re.findall(
            r"TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{(.*?)\n\}", txt, re.S):
        for fn in re.findall(r"(test_3d_glszm_\w+)\s*\(\s*\)", body):
            if fn in where:
                out[f"{suite}.{case}"] = (fn, where[fn])
    return out


def disagreements(rows, asserted, oracles, regression, other, where):
    out = []
    cases = case_to_fn(where)
    by_case = {}          # test_name -> the recipes claiming it, per feature
    seen_assertions = {}  # (feature, recipe, oracle) -> how many rows record it
    for r in rows:
        f = r["feature"]
        covering = asserted.get(f, set()) | regression.get(f, set()) | other.get(f, set())
        files = {where[fn] for fn in covering}
        claimed = {t for t in r["current_test"].split(";") if t}
        recipe = (r.get("config_recipe") or "").strip()
        name = (r.get("test_name") or "").strip()

        if r["status"] == "vetted" and not asserted.get(f):
            out.append(f"{f}: status=vetted but no oracle test asserts it")
        if r["oracle"] and r["oracle"] not in oracles.get(f, set()):
            out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                       f"{sorted(oracles.get(f, ())) or 'none'}")
        for stale in sorted(claimed - files):
            out.append(f"{f}: current_test names {stale}, which covers nothing for it")

        # A row is one assertion, so it has to say which one: a recipe and a band, recorded once.
        if not recipe:
            out.append(f"{f}: no config_recipe, so the row does not say which configuration it is")
        if not (r.get("tolerance") or "").strip():
            out.append(f"{f}: no tolerance, so the row does not say what agreement it claims")
        key = (f, recipe, r["oracle"])
        seen_assertions[key] = seen_assertions.get(key, 0) + 1
        if seen_assertions[key] == 2:
            out.append(f"{f}: two rows record the assertion (recipe {recipe or 'empty'}, oracle "
                       f"{r['oracle'] or 'none'}); one assertion is one row")

        # ...and its two identifiers must agree. current_test is the file the case in test_name is
        # defined in, and ONLY that file: a superset passes a one-way check while still conflating
        # this row's assertion with somebody else's.
        if not name:
            out.append(f"{f}: no test_name, so current_test names an assertion nothing identifies")
            continue
        if name not in cases:
            out.append(f"{f}: test_name {name} resolves to no registered case in test_all.cc")
            continue
        fn, src = cases[name]
        if claimed != {src}:
            out.append(f"{f}: test_name {name} is defined in {src}, but current_test is "
                       f"{r['current_test'] or 'empty'}; a row names the one file its assertion "
                       f"lives in")

        # the kind of the case is the kind of the row, read off the function's name-suffix (SPEC 2)
        kind = fn.rsplit("_", 1)[-1]
        if r["status"] == "vetted":
            if kind not in ORACLE_SUFFIX:
                out.append(f"{f}: status=vetted but {name} is a {kind!r} case, which claims no oracle")
            elif ORACLE_SUFFIX[kind] != r["oracle"]:
                out.append(f"{f}: row claims oracle={r['oracle']!r} but {name} is a "
                           f"{ORACLE_SUFFIX[kind]!r} assertion")
        elif r["status"] == "regression":
            if kind != "regression":
                out.append(f"{f}: status=regression but {name} is a {kind!r} case")
            if r["oracle"]:
                out.append(f"{f}: status=regression but the row claims oracle={r['oracle']!r}")

        # one case asserts one configuration
        prev = by_case.setdefault((f, name), recipe)
        if prev != recipe:
            out.append(f"{f}: {name} is claimed by two recipes, {prev or 'empty'} and "
                       f"{recipe or 'empty'}; one case asserts one configuration")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and registry disagreements instead of rewriting")
    a = ap.parse_args(argv)

    rows = registry_rows()
    names = feature_names()
    asserted, oracles, regression, other, where = collect(feature_re(names), names)
    text = render(rows, asserted, oracles, regression, other)
    problems = disagreements(rows, asserted, oracles, regression, other, where)
    problems += [f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs"
                 for fn in unregistered_tests(where)]
    problems += unregistered_assertions(rows, asserted, regression, where)

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
