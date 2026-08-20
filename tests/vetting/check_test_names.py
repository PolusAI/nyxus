#!/usr/bin/env python3
"""Enforce the SPEC.md 6.1/6.2 test-naming conventions over the test tree.

Two rules, both mechanical:

  6.1 files      test_<dim>_<family>_<kind>.{h,cc,py}
  6.2 functions  test_<dim>_<family>[_<subject>]_<kind>    (gtest case = UPPER(function))

<dim> is 2d or 3d and is mandatory: the same family is computed by different code in each
dimension against different oracles, so a name without it does not say which implementation
an assertion covers. A file whose subject has no image dimensionality carries no token and
must be listed in DIM_AGNOSTIC with a reason, so "no token" is a checked claim rather than an
omission; imq names its own dimension (registry dim=IMQ) and is listed there too. A function's
dim must equal its file's - a 3d function in a 2d file is as wrong as a _regression function
in an _ibsi file.

<kind> is an oracle token (the assertion is vetted against that tool) or one of
regression / invariant / mechanics. A function's kind states what ITS assertion is, so a
function whose kind differs from its file's kind is a file-purity violation (SPEC 2) and
must be listed in KIND_EXCEPTIONS with a reason - the honest name wins over file purity.

Non-assertion files are recognized by suffix and carry no kind:
  *_common.{h,py}   shared fixture/helper header, no assertions of its own
  *_coverage.h      parameterized completeness sweep (per-feature oracle varies by row)
Pure fixtures/harness are listed in FIXTURES.

  6.3.1 includes   a test_<dim>_<family>_* header is that family's, and only that family's
                   files may include it

A fixture two families need is not a reason for one of them to include the other's header: that
puts a family's file in the include graph of assertions it knows nothing about, and it is how a
shared table reached callers it could not see. The neutral home is test_main_nyxus.h; a header
that genuinely belongs to no family is listed in FAMILY_NEUTRAL with the reason.

Helpers are NOT tests and must not use the test_ prefix (the tree convention is assert_*),
so every test_* function found here is treated as a test.

Usage:
    python tests/vetting/check_test_names.py --check     # exit 1 on any violation
    python tests/vetting/check_test_names.py             # list violations, exit 0
"""
import argparse
import pathlib
import re
import sys

# SPEC 4 oracle tokens
ORACLES = {
    "pyradiomics", "radiomicsj", "skimage", "mirp", "matlab", "cellprofiler", "mitk",
    "feature2djava", "wndcharm", "imea", "imagej", "fraclac", "pydicom", "opencv",
    "ibsi", "analytic",
}
# SPEC 2 non-oracle kinds
KINDS = ORACLES | {"regression", "invariant", "mechanics"}

# SPEC 6.3: pure fixtures / harness / framework entry points - out of the taxonomy
FIXTURES = {
    "test_all.cc",              # the gtest translation unit
    "test_data.h", "test_main_nyxus.h", "test_dsb2018_data.h", "test_gabor_truth.h",
    "test_data.py", "test_tissuenet_data.py",
    "test_ref_vals.h",          # the SPEC 6.3.1 table aliases; declares types, asserts nothing
}
# files that predate the convention and are tracked as follow-ups (MIGRATION.md 5.19)
GRANDFATHERED = {
    "test_nyxus.py",            # 88 API assertions across families; needs a by-family split
}

# SPEC 6.1 dim tokens
DIMS = {"2d", "3d"}

# Files that legitimately carry NO dim token, each with the reason it has none. Membership is
# the positive claim "this test's subject has no image dimensionality"; anything not listed
# must name its dimension, so a new 2D file cannot slip in unmarked.
DIM_AGNOSTIC = {
    "test_arrow_mechanics.h": "Arrow/Parquet writer plumbing - no image is read",
    "test_arrow_file_name_mechanics.h": "output-file naming rules",
    "test_initialization_mechanics.h": "environment init",
    "test_feature_manager_mechanics.h": "FeatureManager registration/dependency compile, both dims at once",
    "test_roi_blacklist_mechanics.h": "ROI blacklist parsing",
    "test_hu_analytic.h": "closed form of the scalar SlideProps::uint_friendly_inten map",
    "test_feature_calculation_common.h": "the assert_feature template, used from both dims",
    "test_vetting_mechanics.py": "self-test of check_coverage.py / check_test_names.py",
    "test_environment_lifecycle_mechanics.py":
        "instance -> Environment binding in the bindings; no image dimensionality involved",
    # imq carries its own dimension: dim=IMQ in oracle_coverage.csv, not 2D/3D
    "test_imq_opencv.h": "dim=IMQ in the registry",
    "test_imq_cellprofiler.h": "dim=IMQ in the registry",
    "test_imq_regression.h": "dim=IMQ in the registry",
}

# Headers that belong to no family, so any file may include them. Everything else named
# test_<dim>_<family>_* is that family's and is included only from that family (SPEC 6.3.1); a
# fixture more than one family needs goes to test_main_nyxus.h instead. Membership here is the
# positive claim "this header names a kind or a harness, not a family".
FAMILY_NEUTRAL = {
    "test_3d_coverage_common.h": "named for the _coverage kind: one parameterized sweep per family",
    "test_feature_calculation_common.h": "the assert_feature template, used from both dims",
}

# the gtest translation unit: it includes every assertion header by definition
TRANSLATION_UNIT = "test_all.cc"

# functions whose assertion kind differs from their file's kind, kept in place until the
# file is split (each needs the shared fixture extracted first)
KIND_EXCEPTIONS = {
    # A function's suffix states the kind of ITS OWN assertion, so it normally equals its file's kind.
    # These ten cannot, and the honest function name wins over file purity (SPEC 6.2). All are riders
    # in a Python module whose kind is set by its majority: splitting a .py by kind would fragment a
    # shared fixture across files for one or two assertions.
    "test_2d_glcm_contrast_nonzero_by_default_mechanics":
        "guards the GLCM offset=0 default bug; rides along in the pyradiomics module, which shares its ROI fixture",
    "test_2d_intensity_histogram_requires_ibsi_mechanics":
        "IH gating (features absent unless ibsi=true); rides along in the analytic module",
    "test_2d_intensity_histogram_enabled_with_ibsi_mechanics":
        "IH gating, the positive case; rides along in the analytic module",
    "test_2d_intensity_histogram_index_features_within_bins_invariant":
        "bin-range bound rather than a computed value; rides along in the analytic module",
    "test_2d_intensity_histogram_mergerois_per_label_default_mechanics":
        "--mergerois API behaviour; rides along in the analytic module",
    "test_2d_intensity_histogram_mergerois_collapses_to_one_roi_mechanics":
        "--mergerois API behaviour; rides along in the analytic module",
    "test_2d_intensity_histogram_mergerois_excludes_background_mechanics":
        "--mergerois API behaviour; rides along in the analytic module",
    "test_2d_morphology_boxcount_known_dimension_analytic":
        "closed-form dimensions (square 2.0, line 1.0, Sierpinski log2(3)); rides along in the fraclac module beside the ImageJ/FracLac comparison",
    "test_2d_morphology_perimeter_disk_analytic":
        "closed-form dimension of a disk boundary; rides along in the fraclac module",
    "test_2d_morphology_perimeter_koch_snowflake_analytic":
        "closed-form Koch dimension log4/log3; rides along in the fraclac module",
}

# SPEC 6.3.1 golden reference tables: <family>_<dim>_<oracle>[_<subject>]_ref_vals
TABLE_SUFFIXES = ("_ref_vals_by_label", "_ref_vals_by_angle", "_ref_tols", "_ref_vals")

# A file-scope table holding reference data announces itself with one of these in its name. Any
# such table must satisfy 6.3.1 - which also means a conforming table cannot quietly drift back
# to an old-style name, because _ref_vals is itself a trigger.
TABLE_MARKERS = ("golden", "_gt", "oracle", "reference", "ref_vals", "ref_tols")

# Tables that do not conform yet, with the reason. Empty: every table names one family, one
# dimension and one oracle. An entry here is tracked work, not a waiver - the four that were
# listed were all split by the oracle their registry rows record, rather than renamed to whichever
# oracle happened to cover most of their keys.
TABLE_EXCEPTIONS = {}

# The SPEC 6.3.1 aliases from test_ref_vals.h. A declaration spelled with one of these IS a
# reference table whatever it is named, which is what makes the name rule below apply to the
# complete set rather than only to tables that already sound like one.
TABLE_ALIASES = ("ref_vals_map_by_label", "ref_vals_map_by_angle", "ref_vals_map", "ref_vals_list")

# file-scope declaration (column 0; function locals are always indented in this tree). The brace
# often sits on the next line, so the initialiser is checked by the caller. The tail accepts the
# three ways a table opens -- bare, "{", "=", and "= {" -- as separately optional pieces rather than
# a choice of one: written as (?:=|\{)? it matched "name =" and "name {" but not "name = {", so every
# table declared on one line in that style was skipped and never name-checked at all.
TABLE_DECL = re.compile(
    r"^(?:static[ \t]+)?(?:const[ \t]+)?(?:inline[ \t]+)?"
    r"(?:(?:" + "|".join(TABLE_ALIASES) + r")[ \t]*<.*>"
    r"|std::(?:unordered_)?(?:map|multimap|vector|array|set)[ \t]*<.*>|auto)"
    r"[ \t]+([A-Za-z_]\w*)[ \t]*(?:=[ \t]*)?\{?[ \t]*$")

# The other way a table reaches file scope: an accessor returning a function-local static, where the
# accessor's name is the table's public name. TABLE_DECL cannot see it -- it matches a *variable*
# declaration, and here the line ends in "& name()". Three tables hid in this form, one of them a set
# of MATLAB goldens with a vetting claim in its comment, so the shape is checked rather than trusted.
TABLE_ACCESSOR = re.compile(
    r"^(?:static[ \t]+)?(?:const[ \t]+)?(?:inline[ \t]+)?"
    r"(?:(?:" + "|".join(TABLE_ALIASES) + r")[ \t]*<.*>"
    r"|std::(?:unordered_)?(?:map|multimap|vector|array|set)[ \t]*<.*>)"
    r"[ \t]*&[ \t]*([A-Za-z_]\w*)[ \t]*\([ \t]*\)[ \t]*$")

INCLUDE = re.compile(r'^[ 	]*#include[ 	]+"(test_[^"]+)"', re.M)

CPP_DEF = re.compile(
    r"^[ \t]*(?:static[ \t]+)?(?:inline[ \t]+)?void[ \t]+(test_[A-Za-z0-9_]*)[ \t]*\(([^)]*)\)", re.M)
PY_DEF = re.compile(r"^[ \t]*def[ \t]+(test_[A-Za-z0-9_]*)[ \t]*\(", re.M)
GTEST_CASE = re.compile(r"^TEST\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)", re.M)


def read(path):
    with open(path, "r", newline="", encoding="utf-8", errors="surrogateescape") as fh:
        return fh.read()


def file_dim(name):
    """-> (dim, why). dim is None for files that legitimately carry no dim token."""
    if name in FIXTURES or name in GRANDFATHERED:
        return None, "exempt"          # named by SPEC 6.3, or tracked as a follow-up
    if name in DIM_AGNOSTIC:
        return None, "declared dimension-independent"
    parts = name.rsplit(".", 1)[0].split("_")
    if len(parts) < 2 or parts[0] != "test" or parts[1] not in DIMS:
        return None, "BAD"
    return parts[1], "ok"


def file_kind(name):
    """-> (kind, why). kind is None for recognized non-assertion files."""
    if name in FIXTURES or name in GRANDFATHERED:
        return None, "exempt"
    stem = name.rsplit(".", 1)[0]
    if stem.endswith("_common"):
        return None, "shared fixture"
    if stem.endswith("_coverage"):
        return None, "parameterized sweep"
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "test" or parts[-1] not in KINDS:
        return None, "BAD"
    return parts[-1], "ok"


def file_family(name):
    """-> the family a test file belongs to, or None if it belongs to none.

    The family is what sits between the dim token and the kind, so it may itself contain
    underscores (intensity_histogram). imq carries no dim token and is still a family.
    """
    if name in FIXTURES or name in GRANDFATHERED or name in FAMILY_NEUTRAL:
        return None
    parts = name.rsplit(".", 1)[0].split("_")
    if len(parts) < 3 or parts[0] != "test":
        return None
    parts = parts[1:]
    if parts[0] in DIMS:
        parts = parts[1:]
    if len(parts) < 2:
        return None
    return "_".join(parts[:-1])


def fn_dim(fn):
    """-> the dim token a test function declares, or None if it declares none."""
    parts = fn.split("_")
    return parts[1] if len(parts) > 1 and parts[1] in DIMS else None


def table_violation(name):
    """-> why `name` breaks SPEC 6.3.1, or None if it conforms.

    <family>_<dim>_<oracle>[_<subject>]_ref_vals. family may itself contain underscores
    (intensity_histogram), so the dim token is what splits the name rather than a field count.
    """
    suffix = next((s for s in TABLE_SUFFIXES if name.endswith(s)), None)
    if not suffix:
        return ("does not end in %s (SPEC 6.3.1)" % " / ".join(TABLE_SUFFIXES))
    stem = name[: -len(suffix)]
    dim = next((d for d in DIMS if ("_%s_" % d) in stem), None)
    if not dim:
        return "carries no _2d_/_3d_ dim token (SPEC 6.3.1)"
    family, _, rest = stem.partition("_%s_" % dim)
    if not family:
        return "names no family before the dim token (SPEC 6.3.1)"
    oracle = rest.split("_")[0]
    if oracle not in KINDS:
        return ("'%s' is not an oracle or kind token; the segment after the dim states where the "
                "numbers came from (SPEC 6.3.1)" % oracle)
    return None


def check(root):
    root = pathlib.Path(root)
    tests = root / "tests"
    errors = []

    files = sorted(
        [p for p in tests.glob("*.h")] + [p for p in tests.glob("*.cc")] +
        [p for p in (tests / "python").glob("*.py")])

    # ---- 6.1 file names ----
    kinds, dims = {}, {}
    for p in files:
        kind, why = file_kind(p.name)
        kinds[p.name] = kind
        if why == "BAD":
            errors.append(f"{p.name}: file name does not match "
                          f"test_<dim>_<family>_<kind>{p.suffix} (SPEC 6.1)")
        dim, dwhy = file_dim(p.name)
        dims[p.name] = dim
        if dwhy == "BAD":
            errors.append(f"{p.name}: file name carries no 2d/3d dim token - add one, or list "
                          f"it in DIM_AGNOSTIC with the reason it needs none (SPEC 6.1)")

    # ---- 6.3.1 a family's header is included only from that family ----
    # A fixture two families need belongs in test_main_nyxus.h. Reaching it through the other
    # family's _common.h drags that family's whole include graph -- and, before the tables were
    # split out, its reference data -- into assertions it has nothing to do with.
    for p in files:
        if p.suffix == ".py" or p.name == TRANSLATION_UNIT:
            continue
        mine = file_family(p.name)
        for m in INCLUDE.finditer(read(p)):
            theirs = file_family(m.group(1))
            if theirs and theirs != mine:
                whose = f"the {mine} family" if mine else "no family"
                errors.append(f"{p.name}: belongs to {whose} but includes {m.group(1)}, which is "
                              f"the {theirs} family's - move the shared fixture to "
                              f"test_main_nyxus.h, or list the header in FAMILY_NEUTRAL with the "
                              f"reason it belongs to no family (SPEC 6.3.1)")

    # ---- 6.2 function names ----
    for p in files:
        if p.name in FIXTURES or p.name in GRANDFATHERED:
            continue
        txt = read(p)
        if p.suffix == ".py":
            fns = [(m.group(1), "") for m in PY_DEF.finditer(txt)]
        else:
            fns = [(m.group(1), m.group(2).strip()) for m in CPP_DEF.finditer(txt)]
        for fn, params in fns:
            if params and params not in ("", "void"):
                errors.append(f"{p.name}: helper {fn}() takes arguments - helpers must not use "
                              f"the test_ prefix, rename to assert_* (SPEC 6.2)")
                continue
            fdim = dims.get(p.name)
            if fdim and fn_dim(fn) != fdim:
                errors.append(f"{p.name}: {fn} does not carry its file's _{fdim}_ dim token "
                              f"(SPEC 6.2)")
            elif not fdim and fn_dim(fn) and p.name not in FIXTURES:
                errors.append(f"{p.name}: {fn} claims dim _{fn_dim(fn)}_ but its file is listed "
                              f"as dimension-independent (SPEC 6.1)")
            suffix = fn.rsplit("_", 1)[-1]
            if suffix not in KINDS:
                errors.append(f"{p.name}: {fn} does not end in a kind/oracle token (SPEC 6.2)")
                continue
            fkind = kinds.get(p.name)
            if fkind and suffix != fkind and fn not in KIND_EXCEPTIONS:
                errors.append(f"{p.name}: {fn} is a _{suffix} assertion in a _{fkind} file - "
                              f"move it or add it to KIND_EXCEPTIONS with a reason (SPEC 2)")

    # ---- 6.2 gtest case names mirror the function they call ----
    allcc = tests / "test_all.cc"
    txt = read(allcc)
    defined = set()
    for p in files:
        if p.suffix == ".py":
            continue
        for m in CPP_DEF.finditer(read(p)):
            if not m.group(2).strip() or m.group(2).strip() == "void":
                defined.add(m.group(1))
    for m in GTEST_CASE.finditer(txt):
        suite, case = m.group(1), m.group(2)
        if suite != "TEST_NYXUS":
            errors.append(f"test_all.cc: TEST({suite}, {case}) - gtest suite must be "
                          f"TEST_NYXUS (SPEC 6.2)")
        body_start = txt.find("{", m.end())
        depth, j = 0, body_start
        while j < len(txt):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        called = {c for c in re.findall(r"\b(test_[A-Za-z0-9_]*)\s*\(", txt[body_start:j])
                  if c in defined}
        # case = UPPER(function) is a 1:1 rule, so it can only be checked against a single
        # callee. A body calling two or more test_ functions is rejected rather than waved
        # through: the mirror check would silently degrade to the weak suffix test below,
        # which is the gap that let a mismatched case name survive once already.
        if len(called) == 1:
            fn = called.pop()
            if case != fn.upper():
                errors.append(f"test_all.cc: TEST case {case} must be UPPER({fn}) (SPEC 6.2)")
        elif len(called) > 1:
            errors.append(f"test_all.cc: TEST case {case} calls {len(called)} test_ functions "
                          f"({', '.join(sorted(called))}) - case = UPPER(function) needs one "
                          f"callee, so split the case or make the extras assert_* (SPEC 6.2)")
        # len(called) == 0: the body only calls assert_* helpers, so there is no function name
        # to mirror; the case must still declare its kind.
        elif case.rsplit("_", 1)[-1].lower() not in KINDS:
            errors.append(f"test_all.cc: TEST case {case} does not end in a kind/oracle "
                          f"token (SPEC 6.2)")

    # ---- 6.3.1 golden reference tables ----
    for p in files:
        if p.suffix == ".py":
            continue
        lines = read(p).splitlines()
        for i, line in enumerate(lines):
            if not line or line[0] in " \t/#":
                continue                      # indented -> function local; / or # -> comment
            m = TABLE_DECL.match(line.rstrip())
            if m:
                nxt = next((l.strip() for l in lines[i + 1:i + 3] if l.strip()), "")
                if not (line.rstrip().endswith(("=", "{")) or nxt.startswith(("{", "="))):
                    continue                  # a declaration, not a table with an initialiser
            else:
                # an accessor wrapping a function-local static: the body holds the initialiser, so
                # what is checked is that a static of the same type is declared inside it.
                m = TABLE_ACCESSOR.match(line.rstrip())
                if not m:
                    continue
                body = "\n".join(lines[i + 1:i + 4])
                if "static" not in body:
                    continue                  # returns a reference to something built elsewhere
            name = m.group(1)
            # A table declares itself either by TYPE (one of the 6.3.1 aliases - authoritative,
            # a badly named table cannot hide from it) or, for anything not yet converted, by
            # carrying a reference-data word in its name.
            by_type = any(a in line for a in TABLE_ALIASES)
            if not by_type and not any(k in name.lower() for k in TABLE_MARKERS):
                continue                      # a fixture/helper, not reference data
            if name in TABLE_EXCEPTIONS:
                continue
            # 6.3.1: a table belongs with the assertions that read it. A shared header is read by
            # callers it cannot see, which is how _matlab / _skimage / _cellprofiler functions ended
            # up judging themselves against snapshot tables. _common.h carries fixtures, not values.
            if p.stem.endswith("_common"):
                errors.append(f"{p.name}: golden table {name} is declared in a _common.h; move it "
                              f"to the file whose assertions read it (SPEC 6.3.1)")
                continue
            # 6.3.1: one spelling for one thing. A table written as a bare std::unordered_map is
            # reference data that only the name markers can see, so it is one rename away from being
            # invisible to this check entirely - which is the circularity the aliases removed.
            if not by_type:
                errors.append(f"{p.name}: golden table {name} is declared as a raw container; "
                              f"declare it through a test_ref_vals.h alias "
                              f"({' / '.join(TABLE_ALIASES)}) so it is identifiable by type "
                              f"(SPEC 6.3.1)")
            why = table_violation(name)
            if why:
                errors.append(f"{p.name}: golden table {name} {why}")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if anything violates the spec")
    ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[2]),
                    help="repository root (default: inferred from this script's location)")
    args = ap.parse_args(argv)

    errors = check(args.root)
    for e in errors:
        print("naming:", e)
    print(f"\n{len(errors)} naming violation(s)")
    return 1 if (errors and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
