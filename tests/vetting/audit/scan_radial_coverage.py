"""Regenerate radial_2d_coverage.csv by scanning the 2D radial tests. Stdlib only.

    python tests/vetting/audit/scan_radial_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from the family plan: every `vetted` row in oracle_coverage.csv must be asserted by
an oracle test, that test's oracle must be the one the row names, and `current_test` must name
exactly the files that cover the feature.

Coverage rule: a feature is covered by a test function when its name appears on an ASSERTION line in
that function, or in a golden table that a function loops over while asserting. Comments are
stripped first -- several of them name features they do not assert.

Deliberately NOT counted: a line that merely READS a feature out of the buffer. A
readout-counts-as-coverage rule credits a test with checking features it never looks at. This family
asserts through a helper on a per-feature call line in the regression file, and streams the feature
name into the failure message of every invariant and mechanics assertion, so nothing here needs the
weaker rule.

One file is scanned but deliberately excluded from that last rule: `test_2d_radial_mechanics.h` is
known-defect characterization, so it is reported as coverage in the artifact and is expected to be
absent from `current_test`. See UNCREDITED below.

The kind of coverage comes from the function-name suffix, per SPEC 2 naming. This family has no
oracle file at all -- ORACLE_SUFFIX is empty on purpose -- so no function can contribute an oracle
token, which is what keeps the whole family from reading as a vetting claim. CellProfiler was run
against it and does not vet it: see audit/radial_2d_cellprofiler_vetting_report.md.
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
OUT = os.path.join(HERE, "radial_2d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

SOURCES = [
    "test_2d_radial_regression.h",
    "test_2d_radial_invariant.h",
    "test_2d_radial_mechanics.h",
]

# Scanned, reported in the artifact's Invariant_Mechanics column, and deliberately NOT expected in
# current_test. Every assertion in this file pins a value that
# audit/radial_2d_cellprofiler_vetting_report.md section 6 shows is wrong (defects 1-3), so a
# correct fix must change all of them. Crediting the file as coverage would make those defects
# acceptance criteria for the three features. The exclusion is declared here rather than applied
# silently, and is checked in both directions below: naming the file in current_test is an error
# too, so the decision cannot be reversed by editing only the registry.
UNCREDITED = {
    "test_2d_radial_mechanics.h":
        "known-defect characterization (report section 6 defects 1-3); see not_covered.md A.1",
}
# The regression table is read by a helper the test function calls once per feature, and that call
# line names the feature, so the table needs no attributing to a function here.
TABLE_OWNER = {}

# No oracle covers this family. Left empty rather than deleted so that adding one is a one-line
# change and the suffix rule stays visible.
ORACLE_SUFFIX = {}

FUNC = re.compile(r"^(?:inline\s+)?(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# No trailing \b after `assert`: the regression file asserts through
# assert_radial_vector_feature_regression(...), whose call line is what names the feature.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

NOTE = {
    "FRAC_AT_D": "8-bin vector, every bin asserted separately. Not CellProfiler-vettable: CP's "
                 "FracAtD is the fraction of the ROI's INTENSITY in a bin, Nyxus' is the fraction "
                 "of its PIXEL COUNT and never reads the image.",
    "MEAN_FRAC": "8-bin vector, every bin asserted separately. Not CellProfiler-vettable: CP's "
                 "MeanFrac is dimensionless (bin mean over ROI mean, ~1), Nyxus returns the raw "
                 "bin mean intensity.",
    "RADIAL_CV": "8-bin vector, every bin asserted separately. Not CellProfiler-vettable: CP takes "
                 "the CV of the eight wedge MEANS over the NON-EMPTY wedges, Nyxus the CV of the "
                 "eight wedge SUMS over all eight.",
}


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
    return hits


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
                if r["dim"] == "2D" and r["family"] == "radial"]


def render(rows, asserted, oracles, regression, other):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Invariant_Mechanics", "Notes"])
    for r in rows:
        f = r["feature"]
        w.writerow(["2D", "radial", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    ";".join(sorted(other.get(f, ()))),
                    NOTE.get(f, "")])
    return buf.getvalue()


def unregistered_tests(where):
    """gtest functions that exist but no TEST() in test_all.cc calls - they never run."""
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        registered = set(re.findall(r"(test_\w+)\s*\(\s*\)", fh.read()))
    return sorted(fn for fn, src in where.items()
                  if src.endswith(".h") and fn not in registered)


def disagreements(rows, asserted, oracles, regression, other, where):
    out = []
    for r in rows:
        f = r["feature"]
        covering = asserted.get(f, set()) | regression.get(f, set()) | other.get(f, set())
        files = {where[fn] for fn in covering}
        claimed = {t for t in r["current_test"].split(";") if t}
        if r["status"] == "vetted" and not asserted.get(f):
            out.append(f"{f}: status=vetted but no oracle test asserts it")
        if r["oracle"] and r["oracle"] not in oracles.get(f, set()):
            out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                       f"{sorted(oracles.get(f, ())) or 'none'}")
        if not r["oracle"] and oracles.get(f):
            out.append(f"{f}: registry claims no oracle but {sorted(oracles[f])} test(s) assert it "
                       f"under an oracle-suffixed name")
        for stale in sorted(claimed - files):
            out.append(f"{f}: current_test names {stale}, which covers nothing for it")
        for named in sorted(claimed & set(UNCREDITED)):
            out.append(f"{f}: current_test names {named}, which is uncredited on purpose - "
                       f"{UNCREDITED[named]}")
        for gap in sorted(files - claimed - set(UNCREDITED)):
            out.append(f"{f}: {gap} covers it but current_test omits it")
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
        print(f"checked {len(rows)} rows: "
              f"{'clean' if not problems else str(len(problems)) + ' problem(s)'}")
        return 1 if problems else 0

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT} ({len(rows)} rows)")
    for p in problems:
        print("WARNING:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
