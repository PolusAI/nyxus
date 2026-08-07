#!/usr/bin/env python3
"""Enforce the SPEC.md 6.1/6.2 test-naming conventions over the test tree.

Two rules, both mechanical:

  6.1 files      test_[3d_]<family>_<kind>.{h,cc,py}
  6.2 functions  test_<family>[_<subject>]_<kind>          (gtest case = UPPER(function))

<kind> is an oracle token (the assertion is vetted against that tool) or one of
regression / invariant / mechanics. A function's kind states what ITS assertion is, so a
function whose kind differs from its file's kind is a file-purity violation (SPEC 2) and
must be listed in KIND_EXCEPTIONS with a reason - the honest name wins over file purity.

Non-assertion files are recognized by suffix and carry no kind:
  *_common.{h,py}   shared fixture/helper header, no assertions of its own
  *_coverage.h      parameterized completeness sweep (per-feature oracle varies by row)
Pure fixtures/harness are listed in FIXTURES.

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
}
# files that predate the convention and are tracked as follow-ups (MIGRATION.md 5.19)
GRANDFATHERED = {
    "test_nyxus.py",            # 88 API assertions across families; needs a by-family split
}

# functions whose assertion kind differs from their file's kind, kept in place until the
# file is split (each needs the shared fixture extracted first)
KIND_EXCEPTIONS = {
    # A function's suffix states the kind of ITS OWN assertion, so it normally equals its file's kind.
    # These ten cannot, and the honest function name wins over file purity (SPEC 6.2). All are riders
    # in a Python module whose kind is set by its majority: splitting a .py by kind would fragment a
    # shared fixture across files for one or two assertions.
    "test_glcm_contrast_nonzero_by_default_mechanics":
        "guards the GLCM offset=0 default bug; rides along in the pyradiomics module, which shares its ROI fixture",
    "test_intensity_histogram_requires_ibsi_mechanics":
        "IH gating (features absent unless ibsi=true); rides along in the analytic module",
    "test_intensity_histogram_enabled_with_ibsi_mechanics":
        "IH gating, the positive case; rides along in the analytic module",
    "test_intensity_histogram_index_features_within_bins_invariant":
        "bin-range bound rather than a computed value; rides along in the analytic module",
    "test_intensity_histogram_mergerois_per_label_default_mechanics":
        "--mergerois API behaviour; rides along in the analytic module",
    "test_intensity_histogram_mergerois_collapses_to_one_roi_mechanics":
        "--mergerois API behaviour; rides along in the analytic module",
    "test_intensity_histogram_mergerois_excludes_background_mechanics":
        "--mergerois API behaviour; rides along in the analytic module",
    "test_morphology_boxcount_known_dimension_analytic":
        "closed-form dimensions (square 2.0, line 1.0, Sierpinski log2(3)); rides along in the fraclac module beside the ImageJ/FracLac comparison",
    "test_morphology_perimeter_disk_analytic":
        "closed-form dimension of a disk boundary; rides along in the fraclac module",
    "test_morphology_perimeter_koch_snowflake_analytic":
        "closed-form Koch dimension log4/log3; rides along in the fraclac module",
}

CPP_DEF = re.compile(
    r"^[ \t]*(?:static[ \t]+)?(?:inline[ \t]+)?void[ \t]+(test_[A-Za-z0-9_]*)[ \t]*\(([^)]*)\)", re.M)
PY_DEF = re.compile(r"^[ \t]*def[ \t]+(test_[A-Za-z0-9_]*)[ \t]*\(", re.M)
GTEST_CASE = re.compile(r"^TEST\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)", re.M)


def read(path):
    with open(path, "r", newline="", encoding="utf-8", errors="surrogateescape") as fh:
        return fh.read()


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


def check(root):
    root = pathlib.Path(root)
    tests = root / "tests"
    errors = []

    files = sorted(
        [p for p in tests.glob("*.h")] + [p for p in tests.glob("*.cc")] +
        [p for p in (tests / "python").glob("*.py")])

    # ---- 6.1 file names ----
    kinds = {}
    for p in files:
        kind, why = file_kind(p.name)
        kinds[p.name] = kind
        if why == "BAD":
            errors.append(f"{p.name}: file name does not match "
                          f"test_[3d_]<family>_<kind>{p.suffix} (SPEC 6.1)")

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
            if params and params not in ("", "void") and fn != "test_gabor_skimage":
                errors.append(f"{p.name}: helper {fn}() takes arguments - helpers must not use "
                              f"the test_ prefix, rename to assert_* (SPEC 6.2)")
                continue
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
        if len(called) == 1:
            fn = called.pop()
            if case != fn.upper():
                errors.append(f"test_all.cc: TEST case {case} must be UPPER({fn}) (SPEC 6.2)")
        elif case.rsplit("_", 1)[-1].lower() not in KINDS:
            errors.append(f"test_all.cc: TEST case {case} does not end in a kind/oracle "
                          f"token (SPEC 6.2)")
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
