"""Regenerate glszm_3d_coverage.csv by scanning the 3D GLSZM tests. Stdlib only.

    python tests/vetting/audit/scan_glszm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting. The scan, the
rendering and the run loop live in scanlib.py; the acceptance model below is this family's own.

It is the strictest in the tree, and deliberately so: a row here must say WHICH assertion it records
-- a recipe, a band, and a `test_name` resolving to a case whose file is the ONLY file
`current_test` names -- and the case's kind, read off its name-suffix (SPEC 2), must be the kind the
row claims. It also looks the other way, at cases that assert feature values and have no row at all.
scanlib's shared checks validate rows that exist; the point of `unregistered_assertions` is the
assertion nothing points at.
"""
import os
import re
import sys

import scanlib

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

# recipe -> the function that asserts AT that recipe. Five recipes read the same sixteen features,
# three of them out of one file at one kind, so feature, kind and file together still do not say
# which configuration a row records -- only the function name does. The feature token is
# `[a-z0-9]+` with no underscore, which is what keeps the per-feature sweeps apart from the
# whole-family cases: `_regression` is a suffix of `_constant_roi_regression` too.
RECIPE_READER = {
    "glszm3d.pyradiomics_bincount20": re.compile(r"^test_3d_glszm_[a-z0-9]+_pyradiomics$"),
    "glszm3d.pyradiomics_ibsi_gapped": re.compile(r"^test_3d_glszm_ibsi_gapped_pyradiomics$"),
    "glszm3d.regression_ut_phantom": re.compile(r"^test_3d_glszm_[a-z0-9]+_regression$"),
    "glszm3d.regression_ut_phantom_nobinning":
        re.compile(r"^test_3d_glszm_default_greydepth_regression$"),
    "glszm3d.regression_constant_roi": re.compile(r"^test_3d_glszm_constant_roi_regression$"),
}

# `for (auto fc : D3_GLSZM_feature::featureset)` -- a function that range-loops the calculator's own
# featureset asserts every feature of the family while naming none of them, so no other pattern sees
# it. It is not a table this file could own either: the list comes from the featureset at runtime.
FEATURESET_LOOP = re.compile(r"for\s*\([^)]*:\s*D3_GLSZM_feature::featureset\s*\)")

# Every feature of this family is a contraction of one size-zone matrix, so the note belongs on all
# sixteen rows rather than on a representative one.
FAMILY_NOTE = ("the size-zone matrix all sixteen features contract is pinned cell by cell as well, "
               "read off the feature object calculate() filled, in test_3d_glszm_matrix_pyradiomics")


def note(feature, cov):
    return FAMILY_NOTE


def case_to_fn(cov):
    """-> {gtest case name: (the function it calls, the source file defining that function)}.

    test_all.cc registers `TEST(SUITE, CASE) { ASSERT_NO_THROW(fn()); }`, and `where` already maps a
    test function to the file that defines it, so the two compose into case -> (fn, file). That is
    what lets the checks below confirm a row's test_name and its current_test describe the same
    assertion rather than merely both being true of the feature -- and, through the function's
    name-suffix, that the assertion is of the kind the row claims.
    """
    with open(scanlib.TEST_ALL, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    out = {}
    for suite, case, body in re.findall(
            r"TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{(.*?)\n\}", txt, re.S):
        for fn in re.findall(r"(test_3d_glszm_\w+)\s*\(\s*\)", body):
            if fn in cov.where:
                out[f"{suite}.{case}"] = (fn, cov.where[fn])
    return out


def unregistered_assertions(cov):
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
    named = {r["test_name"].strip() for r in cov.rows if r["test_name"].strip()}
    asserting = {fn for fns in list(cov.asserted.values()) + list(cov.regression.values())
                 for fn in fns}
    return sorted(
        f"{case}: {fn} asserts feature values but no registry row names this case, so nothing "
        f"records which configuration it pins"
        for case, (fn, _src) in case_to_fn(cov).items()
        if fn in asserting and case not in named)


def disagreements(fam, cov):
    out = []
    cases = case_to_fn(cov)
    by_case = {}          # test_name -> the recipes claiming it, per feature
    seen_assertions = {}  # (feature, recipe, oracle) -> how many rows record it
    for r in cov.rows:
        f = r["feature"]
        covering = (cov.asserted.get(f, set()) | cov.regression.get(f, set())
                    | cov.other.get(f, set()))
        files = {cov.where[fn] for fn in covering}
        claimed = {t for t in r["current_test"].split(";") if t}
        recipe = (r.get("config_recipe") or "").strip()
        name = (r.get("test_name") or "").strip()

        if r["status"] == "vetted" and not cov.asserted.get(f):
            out.append(f"{f}: status=vetted but no oracle test asserts it")
        if r["oracle"] and r["oracle"] not in cov.oracles.get(f, set()):
            out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                       f"{sorted(cov.oracles.get(f, ())) or 'none'}")
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

    return out + unregistered_assertions(cov)


FAMILY = scanlib.Family(
    dim="3D", family="glszm", out="glszm_3d_coverage.csv",
    sources=SOURCES,
    oracle_suffix=ORACLE_SUFFIX,
    notes=note,
    table_owner=TABLE_OWNER,
    enum_dim_prefix=True,
    featureset_loop=FEATURESET_LOOP,
    loop_tables=True,
    scan_helpers=True,
    other_note="asserted",
    # the shared per-feature checks are replaced wholesale by the model above; the never-runs sweep
    # is the one built-in this family keeps
    checks={"unregistered"},
    extra_problems=disagreements,
    recipe_reader=RECIPE_READER,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
