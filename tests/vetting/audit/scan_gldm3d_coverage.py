"""Regenerate gldm_3d_coverage.csv by scanning the 3D GLDM tests. Stdlib only.

    python tests/vetting/audit/scan_gldm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting. The scan, the
rendering and the run loop live in scanlib.py; the reading rules and the acceptance model below are
this family's own.

WHAT THIS FAMILY DOES DIFFERENTLY, and why each one is here rather than in scanlib:

  dump helpers        `test_3d_gldm_dump_*` print a table for regeneration; they never compare. They
                      are excluded from the kind buckets so a regenerator cannot read as coverage.
  a pytest oracle     `test_3d_gldm_compatibility` asserts against PyRadiomics goldens through the
                      Python API, so its oracle token cannot come from a name-suffix.
  per-kind rows       a vetted row answers to the oracle files, a regression row to the snapshot
                      one -- so `current_test` must name the files that cover the feature AT THAT
                      ROW'S KIND, not merely somewhere.
  recipe readers      feature, kind and file are all identical between this family's two regression
                      recipes -- one file, one kind, the same fourteen features at GLDM_GREYDEPTH
                      64 and 0 -- so without RECIPE_READER the two sets of rows could be swapped and
                      every other check would stay green.
  keys vs readers     every key of a golden table is resolved to the function its OWN NAME implies,
                      so a missing reader is as loud as a mismatched one. That is how this family
                      shipped 3GLDM_LGLE pinned to 3GLDM_SDE's value, off by a factor of 353, under
                      a function that asserted 3GLDM_SDE.
"""
import os
import re
import sys

import scanlib

SOURCES = [
    "test_3d_gldm_pyradiomics.h",
    "test_3d_gldm_regression.h",
    "test_3d_gldm_common.h",
    os.path.join("python", "test_nyxus.py"),
]

ORACLE_SUFFIX = {"pyradiomics": "pyradiomics"}
# see the module docstring
EXCLUDE_FROM_KIND = re.compile(r"^test_3d_gldm_dump_")
PYTEST_ORACLE = {"test_3d_gldm_compatibility": "pyradiomics"}

# which file answers which kind of registry row
ORACLE_FILES = {"test_3d_gldm_pyradiomics.h", "test_nyxus.py"}
REGRESSION_FILES = {"test_3d_gldm_regression.h"}

# recipe -> the function that asserts AT that recipe. A row's config_recipe is the configuration its
# numbers were taken at, and the function name is where that configuration lives in the tree, so the
# two have to agree.
RECIPE_READER = {
    "gldm3d.pyradiomics_bincount20": re.compile(r"^test_3d_gldm_[a-z0-9]+_pyradiomics$"),
    "gldm3d.regression_ut_phantom": re.compile(r"^test_3d_gldm_[a-z0-9]+_regression$"),
    "gldm3d.regression_ut_phantom_nobinning": re.compile(r"^test_3d_gldm_[a-z0-9]+_nobinning_regression$"),
    "gldm3d.regression_constant_roi": re.compile(r"^test_3d_gldm_constant_roi_regression$"),
}

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

# Filled by collect() and reported by the rewrite summary: features a dump helper names, which are
# deliberately not coverage.
_DUMPS = {}


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def collect(fam, feat_re):
    """-> the five coverage maps, with dump helpers excluded and the pytest oracle credited."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    _DUMPS.clear()
    for rel in fam.sources:
        hits = scanlib.scan(fam, os.path.join(scanlib.TESTS, rel), feat_re)
        for fn, feats in hits.items():
            if not feats:
                continue
            where[fn] = os.path.basename(rel)
            if EXCLUDE_FROM_KIND.match(fn):
                for feat in feats:
                    _DUMPS.setdefault(feat, set()).add(fn)
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
    return asserted, oracles, regression, other, where


def dump_summary(cov):
    """The write line counts rows; this family's artifact is read per feature, so say both."""
    out = [f"{len(cov.features(FAMILY.order))} features"]
    if _DUMPS:
        out.append(f"note: {len(_DUMPS)} feature(s) also appear in dump helpers, "
                   f"excluded from coverage")
    return "\n".join(out)


def registered_calls():
    """-> {gtest case name: the function its body calls}."""
    return {f"{s}.{c}": fn for s, c, fn in re.findall(
        r"TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{\s*ASSERT_NO_THROW\s*\(\s*(\w+)\s*\(",
        read(scanlib.TEST_ALL))}


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

    A key nothing reads is where a bad number lives, because no assertion ever evaluates it.

    The reader's feature token is [a-z0-9]+ with no underscore, which is what keeps the tables apart:
    `_regression` is a suffix of `_nobinning_regression` too, and matching on the suffix alone would
    let each table's readers answer for the other's keys.
    """
    out = []
    registered = set(registered_calls().values())
    for table, (rel, suffix) in TABLES.items():
        text = read(os.path.join(scanlib.TESTS, rel))
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


def disagreements(fam, cov):
    """Each registry row against the tests of ITS OWN KIND - see the module docstring."""
    out = []
    cases = registered_calls()
    for r in cov.rows:
        f, st = r["feature"], r["status"].strip()
        claimed = {t for t in r["current_test"].split(";") if t}
        if st == "vetted":
            covering = cov.asserted.get(f, set())
            if not covering:
                out.append(f"{f}: status=vetted but no oracle test asserts it")
            if r["oracle"] and r["oracle"] not in cov.oracles.get(f, set()):
                out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                           f"{sorted(cov.oracles.get(f, ())) or 'none'}")
        elif st == "regression":
            covering = cov.regression.get(f, set())
            if not covering:
                out.append(f"{f}: status=regression but no snapshot test asserts it")
            if r["oracle"].strip():
                out.append(f"{f}: status=regression but names oracle {r['oracle']!r}")
        else:
            continue
        files = {cov.where[fn] for fn in covering}
        for stale in sorted(claimed - files):
            out.append(f"{f} ({st}): current_test names {stale}, which carries no assertion of "
                       f"this kind for it")
        for gap in sorted(files - claimed):
            out.append(f"{f} ({st}): {gap} asserts it but current_test omits it")
        for bad in sorted(t for t in claimed if "mechanics" in t or "coverage" in t):
            out.append(f"{f} ({st}): current_test names {bad}, which pins no reference value")
        out += test_name_problems(r, f, st, covering, claimed, cases, cov.where)
    return out + key_reader_problems()


FAMILY = scanlib.Family(
    dim="3D", family="gldm", out="gldm_3d_coverage.csv",
    sources=SOURCES,
    oracle_suffix=ORACLE_SUFFIX,
    notes=NOTE,
    scan_helpers=True,
    # a name must not match at the tail of a longer one; see scanlib.feature_re
    boundary="strict",
    extra_column="Invariant",
    order="sorted",
    collect_override=collect,
    extra_summary=dump_summary,
    # every built-in check is replaced by the per-kind model above, registration included
    checks=frozenset(),
    extra_problems=disagreements,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
