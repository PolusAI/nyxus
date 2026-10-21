"""Scan the 3D GLCM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_glcm3d_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration, plus the one reading rule that is its own (chained_helper_hits below).
"""
import os
import re
import sys

import scanlib

VOLMIN_TEST = os.path.join("python", "test_3d_glcm_pyradiomics.py")
ANALYTIC_TEST = os.path.join("python", "test_3d_glcm_analytic.py")

SOURCES = [
    "test_3d_glcm_pyradiomics.h",
    "test_3d_glcm_regression.h",
    os.path.join("python", "test_nyxus.py"),
    VOLMIN_TEST,
    ANALYTIC_TEST,
]

ORACLE_SUFFIX = {"pyradiomics": "pyradiomics", "analytic": "analytic"}


def chained_helper_hits(feat_re):
    """-> {test function: features} in ANALYTIC_TEST, read through helpers that call helpers.

    Its two cases assert through `_assert_matches_definition()`, which compares against the dict
    `_reference()` returns, and only `_reference()` names the features. scanlib credits a case with
    the features of the helpers it calls directly; this follows the calls from helper to helper,
    so the case is credited with what the chain it runs compares.
    """
    with open(os.path.join(scanlib.TESTS, ANALYTIC_TEST), encoding="utf-8", errors="replace") as fh:
        text = scanlib.strip_comments(fh.read())
    starts = [m.start() for m in scanlib.TOPLEVEL_DEF.finditer(text)] + [len(text)]
    blocks = {re.match(r"def\s+(\w+)", text[a:]).group(1): text[a:b]
              for a, b in zip(starts, starts[1:])}

    def reach(name, seen):
        feats = set(feat_re.findall(blocks[name])) if name.startswith("_") else set()
        for callee in blocks:
            if (callee.startswith("_") and callee not in seen
                    and re.search(r"\b" + re.escape(callee) + r"\s*\(", blocks[name])):
                feats |= reach(callee, seen | {callee})
        return feats

    return {name: reach(name, {name}) for name, block in blocks.items()
            if name.startswith("test_") and scanlib.ASSERTION.search(block)}


def collect(fam, feat_re):
    """-> the five coverage maps: scanlib's, with the chained-helper credit of ANALYTIC_TEST added."""
    asserted, oracles, regression, other, where = scanlib.collect(fam, feat_re)
    for fn, feats in chained_helper_hits(feat_re).items():
        where[fn] = os.path.basename(ANALYTIC_TEST)
        token = ORACLE_SUFFIX.get(fn.rsplit("_", 1)[-1])
        for feat in feats:
            if token:
                asserted.setdefault(feat, set()).add(fn)
                oracles.setdefault(feat, set()).add(token)
            else:
                other.setdefault(feat, set()).add(fn)
    return asserted, oracles, regression, other, where


FAMILY = scanlib.Family(
    dim="3D", family="glcm",
    sources=SOURCES,
    oracle_suffix=ORACLE_SUFFIX,
    # the volmin case asserts the keys of its dict through a shared helper, so the table is credited
    # to the case that loops it; the family's C++ tables need no owner
    table_owner={"GLCM_3D_VOLMIN_PYRADIOMICS": "test_3d_glcm_volmin_pyradiomics"},
    table_dialect="python",
    enum_dim_prefix=True,
    # the _AVE test aliases the enum (`using F = Nyxus::Feature3D;`), so cover `F::` too
    enum_alias="GLCM",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
    collect_override=collect,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
