"""Regenerate ngtdm_2d_coverage.csv by scanning the 2D NGTDM tests. Stdlib only.

    python tests/vetting/audit/scan_ngtdm_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from the family plan: every `vetted` row in oracle_coverage.csv must be asserted by
an oracle test, that test's oracle must be the one the row names, and `current_test` must name
exactly the files that cover the feature AT THAT ROW'S CONFIG.

The registry carries one row per (feature x config), so this family has two rows per feature -- the
IBSI-mode row the oracles vet and the default-mode row only the snapshots pin -- and each is checked
against the tests of its own kind. The mechanics file answers to neither: it pins no reference value,
so a row citing it would be claiming coverage it does not have.

Coverage rule: a feature is covered by a test function when its name appears on an ASSERTION line in
that function, or in a golden table that a function loops over while asserting. Comments are
stripped first -- several of them name features they do not assert.

Deliberately NOT counted: a line that merely READS a feature out of the buffer. A
readout-counts-as-coverage rule credits an oracle test with vetting features it never checks.

The kind of coverage comes from the function-name suffix, per SPEC 2 naming, so only an
oracle-suffixed function contributes an oracle token. A mechanics-suffixed function therefore contributes
coverage but never an oracle token, which is what keeps test_2d_ngtdm_mechanics.h from reading as a
vetting claim.
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
OUT = os.path.join(HERE, "ngtdm_2d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

SOURCES = [
    "test_2d_ngtdm_ibsi.h",
    "test_2d_ngtdm_mirp.h",
    "test_2d_ngtdm_regression.h",
    "test_2d_ngtdm_mechanics.h",
]
# Each oracle file loops its five features calling a per-oracle helper that names the feature on the
# call line, so no table needs attributing to a function here.
TABLE_OWNER = {}

ORACLE_SUFFIX = {"mirp": "mirp", "ibsi": "ibsi"}

# `inline void` is a legal way to declare a case, so the optional keyword is part of the
# pattern -- without it such a function is invisible and its features look uncovered.
FUNC = re.compile(r"^(?:inline\s+)?(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# A module-level helper in the pytest files, e.g. `def _run(...)`. The python tests build the frame
# inside such a helper, so credit the helper's features to every test function that calls it.
HELPER = re.compile(r"^def\s+(_\w+)\s*\(", re.M)
# Every top-level def, helper or test. A helper's body ends at the NEXT TOP-LEVEL DEF OF ANY KIND,
# which is what bounds it correctly -- ending it at the next *helper* instead lets the last helper in
# a file swallow every test function below it, so the helper picks up every feature name in the file
# and each test that calls it inherits the lot. That over-credited NUM_NEIGHBORS and
# CLOSEST_NEIGHBOR1_DIST to a PERCENT_TOUCHING-only test here.
TOPLEVEL_DEF = re.compile(r"^def\s+\w+\s*\(", re.M)
# No trailing \b after `assert`: this family asserts through per-oracle helpers
# (assert_ngtdm_feature_mirp, assert_ngtdm_feature_ibsi, ...), so the call line that names the
# feature is `assert_ngtdm_feature_<oracle>(Feature2D::X, "X")`, not a bare ASSERT_ macro.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

# All five features are deliberately asserted by both oracle files. That is not redundancy: the IBSI
# consensus values are published to three significant figures and fix the DEFINITION (rel=1e-2),
# while mirp reproduces Nyxus to 3.2e-16 and fixes the DIGITS (SPEC 7's exact tier, abs=1e-9). See
# audit/ngtdm_2d_mirp_vetting_report.md.
DUAL_ORACLE = ("asserted against both oracles by design: IBSI fixes the definition at its published "
               "3-significant-figure precision, mirp fixes the digits at 3.2e-16, with a PyRadiomics "
               "run corroborating mirp to 1.6e-16")
NOTE = {}


def feature_names():
    """The family's feature names, longest first so a name that prefixes another cannot win."""
    names = {r["feature"] for r in registry_rows()}
    return sorted(names, key=len, reverse=True)


def feature_re(names):
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")


def scan(path, feat_re):
    """-> {test function name: {features it covers}}."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
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
    return hits


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
    """-> (oracle fns, oracle tokens, regression fns, other-kind fns, fn -> file)."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    for rel in SOURCES:
        for fn, feats in scan(os.path.join(TESTS, rel), feat_re).items():
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
                if r["dim"] == "2D" and r["family"] == "ngtdm"]


def render(rows, asserted, oracles, regression, other):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Mechanics", "Notes"])
    # one artifact line per feature, not per registry row: the registry splits a feature across a
    # vetted row and a regression row, and this table is about the tests, which do not split
    for f in dict.fromkeys(r["feature"] for r in rows):
        w.writerow(["2D", "ngtdm", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    ";".join(sorted(other.get(f, ()))),
                    NOTE.get(f, DUAL_ORACLE if len(oracles.get(f, ())) > 1 else "")])
    return buf.getvalue()


def unregistered_tests(where):
    """gtest functions that exist but no TEST() in test_all.cc calls - they never run.

    Only the C++ headers are checked; pytest collects the .py functions by name.
    """
    # every registered name, not just this family's, so a scanner that is later pointed at another
    # family's file does not report that file's cases as never running
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        registered = set(re.findall(r"(test_\w+)\s*\(\s*\)", fh.read()))
    return sorted(fn for fn, src in where.items()
                  if src.endswith(".h") and fn not in registered)


def disagreements(rows, asserted, oracles, regression, other, where):
    """Each row is answerable for the tests of its own kind, and no row for the mechanics ones."""
    out = []
    mechanics_files = {where[fn] for fns in other.values() for fn in fns}
    for r in rows:
        f = r["feature"]
        claimed = {t for t in r["current_test"].split(";") if t}
        if r["status"] == "vetted":
            files = {where[fn] for fn in asserted.get(f, ())}
            if not asserted.get(f):
                out.append(f"{f}: status=vetted but no oracle test asserts it")
            if r["oracle"] and r["oracle"] not in oracles.get(f, set()):
                out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                           f"{sorted(oracles.get(f, ())) or 'none'}")
            if not r["oracle"]:
                out.append(f"{f}: status=vetted but the row names no oracle")
        else:
            files = {where[fn] for fn in regression.get(f, ())}
            if r["oracle"]:
                out.append(f"{f}: status={r['status']} but the row names oracle {r['oracle']!r}")
        for stale in sorted(claimed - files):
            if stale in mechanics_files:
                out.append(f"{f}: current_test names {stale}, a mechanics file - it pins no "
                           f"reference value, so no registry row is covered by it")
            else:
                out.append(f"{f}: current_test names {stale}, which covers nothing for it at "
                           f"recipe {r['config_recipe'] or '(none)'}")
        for gap in sorted(files - claimed):
            out.append(f"{f}: {gap} covers it but current_test omits it")

    # the reverse gap: a kind of test with no row to answer for it
    for f in sorted(asserted):
        if not any(r["feature"] == f and r["status"] == "vetted" for r in rows):
            out.append(f"{f}: {sorted(asserted[f])} assert it but no registry row is vetted")
    for f in sorted(regression):
        if not any(r["feature"] == f and r["status"] == "regression" for r in rows):
            out.append(f"{f}: {sorted(regression[f])} pin it but no registry row is a regression one")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and registry disagreements instead of rewriting")
    a = ap.parse_args(argv)

    rows = registry_rows()
    feat_re = feature_re(feature_names())
    asserted, oracles, regression, other, where = collect(feat_re)
    text = render(rows, asserted, oracles, regression, other)
    problems = disagreements(rows, asserted, oracles, regression, other, where)
    problems += [f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs"
                 for fn in unregistered_tests(where)]

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
