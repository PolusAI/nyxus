"""Negative controls for the coverage rule itself. Stdlib only, no build, no test tree needed.

    python tests/vetting/audit/scanlib_selftest.py

The per-family `--check` compares an artifact to the tree; nothing there can say whether the RULE
that produced the artifact is right, because both sides of that comparison come from the same rule.
These are the other direction: a fixture where the answer is known, and a fault injected into it.

Every case here is a shape that once read as evidence and is not:

  a literal the function holds but never loops     the loop says WHICH list is compared
  a loop that reads its list without comparing     the loop has to do the asserting, or feed it
  a returned value the caller discards             coverage is an assertion, not a read
  a case registered on one line                    the body must be brace-matched, not line-matched
  an oracle case under a regression row            a row answers to evidence of its own kind
  the wrong tool's case under a vetted row         and to the oracle it actually claims
  the feature from one case, the recipe from       one function has to answer the whole row
    another

Each is written twice, once where the feature IS covered and once where the same text stops just
short of covering it, because a check that only ever sees the passing half is not a check.
"""
import os
import re
import sys
import tempfile

import scanlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, got, want):
    got, want = sorted(got), sorted(want)
    if got != want:
        FAILURES.append(f"{name}: got {got}, want {want}")
    print(f"  {'ok  ' if got == want else 'FAIL'}  {name}")


def scanned(text, names, **kw):
    """-> {function: features} for `text` written to a temp .py and read by the shared scan."""
    fam = scanlib.Family(dim="2D", family="selftest", out="none.csv", sources=[],
                         oracle_suffix={"analytic": "analytic"}, **kw)
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return scanlib.scan(fam, path, scanlib.feature_re(names))
    finally:
        os.remove(path)


# ---------------------------------------------------------------- py_loop_tables
# The concession: a case that names its features in a list and loops the list never puts a name on
# an assertion line. The control: a list it does not loop is not that.
LITERALS = '''
def test_2d_selftest_reach_analytic():
    unused = ("PERIMETER",)
    ellipse = ("ECCENTRICITY",)
    cols = [c for c in frame if c.startswith("EROSIONS_2_VANISH") or c in ellipse]
    for c in cols:
        assert frame[c] == other[c]
'''


def literals():
    print("py_loop_tables")
    names = {"PERIMETER", "ECCENTRICITY", "EROSIONS_2_VANISH", "AREA"}
    hits = scanned(LITERALS, names, py_loop_tables=True)["test_2d_selftest_reach_analytic"]
    # `cols` is looped; `ellipse` is reached through it; `unused` is neither
    check("iterated literal is credited", hits & {"EROSIONS_2_VANISH"}, {"EROSIONS_2_VANISH"})
    check("literal a reached literal names is credited", hits & {"ECCENTRICITY"}, {"ECCENTRICITY"})
    check("literal beside a real loop is NOT credited", hits & {"PERIMETER"}, set())

    no_loop = LITERALS.replace("    for c in cols:\n        assert frame[c] == other[c]\n",
                               "    assert cols\n")
    check("no loop at all credits no literal",
          scanned(no_loop, names, py_loop_tables=True)["test_2d_selftest_reach_analytic"], set())

    check("the concession is opt-in",
          scanned(LITERALS, names)["test_2d_selftest_reach_analytic"], set())

    # the shape the tree actually uses: the loop collects and the assertion is below it, outside
    accumulating = LITERALS.replace(
        "    for c in cols:\n        assert frame[c] == other[c]\n",
        "    bad = []\n"
        "    for c in cols:\n"
        "        if frame[c] != other[c]:\n"
        "            bad.append(c)\n"
        "    assert not bad, bad\n")
    check("a loop that collects for an assertion below it is credited",
          scanned(accumulating, names, py_loop_tables=True)["test_2d_selftest_reach_analytic"],
          {"ECCENTRICITY", "EROSIONS_2_VANISH"})

    # the control: the same loop, doing nothing with what it iterates, beside a real assertion
    logging = LITERALS.replace(
        "    for c in cols:\n        assert frame[c] == other[c]\n",
        "    for c in cols:\n        print(c)\n    assert frame is not other\n")
    check("a loop that only reads its list is NOT credited",
          scanned(logging, names, py_loop_tables=True)["test_2d_selftest_reach_analytic"], set())


# ---------------------------------------------------------------- helper attribution
# The concession: a pytest case asserts on a scalar a helper returns, so the feature name is in the
# helper. The control: the half of the return the caller throws away is not asserted by it.
HELPERS = '''
def _fd(label):
    """Return (box-count D, perimeter D)."""
    row = _run(label)
    bc = [c for c in row.index if c.endswith("FRACT_DIM_BOXCOUNT")][0]
    pf = [c for c in row.index if c.endswith("FRACT_DIM_PERIMETER")][0]
    return float(row[bc]), float(row[pf])


def _area(label):
    return float(_run(label)["AREA"])


def test_2d_selftest_boxcount_analytic():
    bc, _ = _fd(mask)
    assert abs(bc - 2.0) < 0.1


def test_2d_selftest_perimeter_analytic():
    _, pf = _fd(disk)
    assert abs(pf - 1.0) < 0.05


def test_2d_selftest_bound_analytic():
    both = _fd(disk)
    assert both[0] >= both[1]


def test_2d_selftest_area_analytic():
    assert _area(mask) > 0
'''


def helpers():
    print("helper attribution")
    names = {"FRACT_DIM_BOXCOUNT", "FRACT_DIM_PERIMETER", "AREA"}
    hits = scanned(HELPERS, names, scan_helpers=True)
    check("caller asserting the first of two gets only the first",
          hits["test_2d_selftest_boxcount_analytic"], {"FRACT_DIM_BOXCOUNT"})
    check("caller asserting the second of two gets only the second",
          hits["test_2d_selftest_perimeter_analytic"], {"FRACT_DIM_PERIMETER"})
    check("caller that does not unpack keeps the whole helper",
          hits["test_2d_selftest_bound_analytic"],
          {"FRACT_DIM_BOXCOUNT", "FRACT_DIM_PERIMETER"})
    check("a single-value helper is unchanged", hits["test_2d_selftest_area_analytic"], {"AREA"})

    # the fault: the caller binds the value it never asserts. Coverage has to follow the assertion,
    # not the binding, or `_` is the only thing keeping the rule honest.
    bound = HELPERS.replace("    bc, _ = _fd(mask)\n    assert abs(bc - 2.0) < 0.1",
                            "    bc, pf = _fd(mask)\n    assert abs(bc - 2.0) < 0.1")
    check("a bound name that reaches no assertion is not coverage",
          scanned(bound, names, scan_helpers=True)["test_2d_selftest_boxcount_analytic"],
          {"FRACT_DIM_BOXCOUNT"})


# ---------------------------------------------------------------- case -> function
# test_all.cc writes a fifth of its registrations on one line. A body read to the first `}` at the
# start of a line runs past every one of them into the next multi-line case.
REGISTRATIONS = '''
TEST(TEST_NYXUS, TEST_2D_SELFTEST_ONE_LINE_ANALYTIC) { ASSERT_NO_THROW (test_2d_selftest_one_line_analytic()); }

TEST(TEST_NYXUS, TEST_2D_SELFTEST_MULTI_LINE_ANALYTIC)
{
\tASSERT_NO_THROW(test_2d_selftest_multi_line_analytic());
}
'''


def registrations():
    print("case -> function")
    fam = scanlib.Family(dim="2D", family="selftest", out="none.csv", sources=[],
                         oracle_suffix={"analytic": "analytic"})
    cov = scanlib.Coverage([], {}, {}, {}, {},
                           {"test_2d_selftest_one_line_analytic": "test_2d_selftest_analytic.h",
                            "test_2d_selftest_multi_line_analytic": "test_2d_selftest_analytic.h"})
    fd, path = tempfile.mkstemp(suffix=".cc")
    os.close(fd)
    real = scanlib.TEST_ALL
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(REGISTRATIONS)
        scanlib.TEST_ALL = path
        cases = scanlib.case_to_fns(fam, cov)
    finally:
        scanlib.TEST_ALL = real
        os.remove(path)
    check("a one-line registration resolves to its own function",
          cases.get("TEST_NYXUS.TEST_2D_SELFTEST_ONE_LINE_ANALYTIC", []),
          ["test_2d_selftest_one_line_analytic"])
    check("and does not swallow the case below it",
          cases.get("TEST_NYXUS.TEST_2D_SELFTEST_MULTI_LINE_ANALYTIC", []),
          ["test_2d_selftest_multi_line_analytic"])


# ---------------------------------------------------------------- the row-scoped verdict
# Read against the real registry and the real tree, because the fault being controlled for is a row
# agreeing with a tree it was never compared to.
def verdicts():
    print("row-scoped verdict")
    import report_features as rf

    cov, rows = rf.scan_all(), rf.registry()
    vetted = {(r["dim"], r["feature"]) for r in rows if r["status"].strip() == "vetted"}
    row = next(r for r in rows
               if r["dim"] == "3D" and r["feature"] == "3GLSZM_GLN"
               and r["config_recipe"].strip() == "glszm3d.pyradiomics_bincount20")
    check("the row as it stands", [rf.verdict_of(row, cov, vetted)], [("agree", "row+config")])

    swapped = dict(row, config_recipe="glszm3d.regression_constant_roi")
    check("its recipe swapped for another of the family's",
          [rf.verdict_of(swapped, cov, vetted)], [("recipe-mismatch", "row+config")])

    unknown = dict(row, config_recipe="glszm3d.not_a_recipe")
    check("a recipe the family declares no reader for",
          [rf.verdict_of(unknown, cov, vetted)], [("recipe-unreadable", "row+config")])

    # the case exists and asserts the family, but not this feature
    other = dict(row, test_name="TEST_NYXUS.TEST_3D_GLSZM_SAE_PYRADIOMICS")
    check("test_name pointing at another feature's case",
          [rf.verdict_of(other, cov, vetted)], [("row-test-lacks-feature", "row")])

    bogus = dict(row, test_name="TEST_NYXUS.TEST_3D_GLSZM_NO_SUCH_CASE")
    check("test_name naming no registered case",
          [rf.verdict_of(bogus, cov, vetted)], [("test-name-unresolved", "row")])

    # the feature and the recipe have to be answered by the SAME case. The first of these asserts
    # GLN at the wrong recipe and the second asserts SAE at the right one, so each question has an
    # answer and the row still has no evidence.
    split = dict(row, test_name="TEST_NYXUS.TEST_3D_GLSZM_IBSI_GAPPED_PYRADIOMICS;"
                                "TEST_NYXUS.TEST_3D_GLSZM_SAE_PYRADIOMICS")
    check("feature proven by one case and recipe by another",
          [rf.verdict_of(split, cov, vetted)], [("recipe-mismatch", "row+config")])

    # a regression row answers to the snapshot guard, not to an oracle case that happens to assert
    # the same feature
    guard = next(r for r in rows
                 if r["dim"] == "2D" and r["feature"] == "NGTDM_COARSENESS"
                 and r["status"].strip() == "regression")
    check("the regression row as it stands", [rf.verdict_of(guard, cov, vetted)], [("agree", "row")])
    check("a regression row pointed at an oracle case",
          [rf.verdict_of(dict(guard, test_name="TEST_NYXUS.TEST_2D_NGTDM_COARSENESS_IBSI"),
                         cov, vetted)],
          [("row-test-wrong-kind", "row")])

    # and a vetted row answers to the oracle it claims, not to whichever tool the tree asserts
    claim = next(r for r in rows
                 if r["dim"] == "2D" and r["feature"] == "NGTDM_COARSENESS"
                 and r["status"].strip() == "vetted")
    check("the vetted row as it stands", [rf.verdict_of(claim, cov, vetted)], [("agree", "row")])
    check("oracle=mirp with only the IBSI case named",
          [rf.verdict_of(dict(claim, test_name="TEST_NYXUS.TEST_2D_NGTDM_COARSENESS_IBSI"),
                         cov, vetted)],
          [("row-test-wrong-oracle", "row")])

    blank = dict(row, test_name="")
    check("a row naming no assertion falls back to the feature",
          [rf.verdict_of(blank, cov, vetted)], [("agree", "feature")])


def main():
    literals()
    helpers()
    registrations()
    verdicts()
    for f in FAILURES:
        print("ERROR:", f)
    print(f"{'clean' if not FAILURES else str(len(FAILURES)) + ' failure(s)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
