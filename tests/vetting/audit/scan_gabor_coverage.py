"""Scan the 2D Gabor tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_gabor_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration, plus the one reading rule the shared scan does not have.

The family is one feature, GABOR, asserted at two configurations. Both oracle cases hand a golden
table to `assert_2d_gabor_scores_skimage()`, a file-local helper that holds every assertion, so no
case names the feature itself and the shared line rule reads nothing out of `test_2d_gabor_skimage.cc`.
`collect()` below credits a case with what the helper it calls asserts.
"""
import os
import re
import sys

import scanlib

SKIMAGE = "test_2d_gabor_skimage.cc"
MECHANICS = "test_2d_gabor_mechanics.h"
SOURCES = [SKIMAGE, MECHANICS]

# a file-local C++ helper that asserts, e.g. `static void assert_2d_gabor_scores_skimage (...)`
CPP_HELPER = re.compile(r"^(?:static\s+)?void\s+(assert_\w+)\s*\(", re.M)

NOTE = {
    "GABOR": "two config points, both oracle-asserted on bench_dsb2018_2d. skimage supplies the "
             "KERNEL; the f0=0 member and the WND-CHARM count-ratio SCORE are analytic (skimage has "
             "no native equivalent of the score, and gabor_kernel cannot express frequency 0). No "
             "regression file exists: every pinned value carries an oracle claim. The GPU path "
             "carries no oracle claim and is guarded size-only by the mechanics case",
}


def helper_credit(path, feat_re):
    """-> {test function: features} for the cases in `path` that call an asserting helper."""
    with open(os.path.join(scanlib.TESTS, path), encoding="utf-8", errors="replace") as fh:
        text = scanlib.strip_comments(fh.read())
    marks = sorted([(m.start(), m.group(1)) for m in CPP_HELPER.finditer(text)]
                   + [(m.start(), m.group(1) or m.group(2)) for m in scanlib.FUNC.finditer(text)])
    blocks = {name: text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
              for i, (pos, name) in enumerate(marks)}

    asserted = {}
    for name, block in blocks.items():
        if name.startswith("assert_"):
            asserted[name] = {f for line in block.splitlines() if scanlib.ASSERTION.search(line)
                              for f in feat_re.findall(line)}
    out = {}
    for name, block in blocks.items():
        if not name.startswith("test_"):
            continue
        for helper, feats in asserted.items():
            if feats and re.search(r"\b" + re.escape(helper) + r"\s*\(", block):
                out.setdefault(name, set()).update(feats)
    return out


def collect(fam, feat_re):
    names = {r["feature"] for r in scanlib.registry_rows(fam)}
    asserted, oracles, regression, other, where = scanlib.collect(fam, feat_re, names)
    for fn, feats in helper_credit(SKIMAGE, feat_re).items():
        where[fn] = SKIMAGE
        kind = fn.rsplit("_", 1)[-1]
        for feat in feats:
            if kind in fam.oracle_suffix:
                asserted.setdefault(feat, set()).add(fn)
                oracles.setdefault(feat, set()).add(fam.oracle_suffix[kind])
    return asserted, oracles, regression, other, where


FAMILY = scanlib.Family(
    dim="2D", family="gabor",
    sources=SOURCES,
    oracle_suffix={"skimage": "skimage"},
    notes=NOTE,
    collect_override=collect,
    # the GPU plumbing guard is not a vetting claim, so current_test names the oracle file only
    current_exempt=(MECHANICS,),
    # each row names the case it runs in, and the case's name carries its configuration
    checks=scanlib.DEFAULT_CHECKS | scanlib.IDENTITY_CHECKS,
    recipe_reader={
        "gabor.cpp_static_defaults": re.compile(r"test_2d_gabor_cpp_static_defaults_"),
        "gabor.python_raw_defaults": re.compile(r"test_2d_gabor_python_raw_defaults_"),
    },
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
