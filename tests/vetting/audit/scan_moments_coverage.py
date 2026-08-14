"""Regenerate moments_2d_coverage.csv by scanning the 2D moments tests. Stdlib only.

    python tests/vetting/audit/scan_moments_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from PR423-family-plan: every `vetted` row in oracle_coverage.csv must be asserted
by an oracle test, that test's oracle must be the one the row names, `current_test` must name
exactly the files that cover the feature, and no gtest function may be left unregistered.

Coverage rule: this family pins goldens as `ref_vals_list<GeomomentGoldenValue>` entries shaped
`{Nyxus::Feature2D::X, "X", value}`, and each test function asserts one whole table via
assert_2d_geomoment_features(). So a feature is covered by the function that loops the table its
entry lives in. Comments are stripped first. The kind of coverage comes from the function-name
suffix, per SPEC 2 naming, so only an oracle-suffixed function contributes an oracle token.
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
OUT = os.path.join(HERE, "moments_2d_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")

SOURCES = ["test_2d_moments_skimage.h", "test_2d_moments_regression.h", "test_2d_moments_common.h"]
ORACLE_SUFFIX = {"skimage": "skimage", "pyradiomics": "pyradiomics", "mirp": "mirp",
                 "ibsi": "ibsi", "analytic": "analytic"}

COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
TABLE = re.compile(r"ref_vals_list<GeomomentGoldenValue>\s+(\w+)\s*\{(.*?)\n\};", re.S)
ENTRY = re.compile(r'\{\s*Nyxus::Feature2D::(\w+)\s*,\s*"(\w+)"')
FUNC = re.compile(r"^void\s+(test_2d_moments_\w+)\s*\(", re.M)
USES = re.compile(r"assert_2d_geomoment_features\s*\([^,]+,\s*(\w+)")


def scan():
    """-> (feature -> {function}, function -> file, table -> {features})."""
    covers, where, tables = {}, {}, {}
    for rel in SOURCES:
        with open(os.path.join(TESTS, rel), encoding="utf-8", errors="replace") as fh:
            text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), fh.read())
        for name, body in TABLE.findall(text):
            feats = set()
            for m in ENTRY.finditer(body):
                if m.group(1) != m.group(2):
                    print(f"ERROR: {rel}: enum/name mismatch {m.group(1)} vs {m.group(2)}")
                feats.add(m.group(1))
            tables[name] = feats
        marks = [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
        for i, (pos, fn) in enumerate(marks):
            block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
            where[fn] = os.path.basename(rel)
            for tbl in USES.findall(block):
                for feat in tables.get(tbl, ()):
                    covers.setdefault(feat, set()).add(fn)
    return covers, where, tables


def registry_rows():
    with open(REGISTRY, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)
                if r["dim"] == "2D" and r["family"] == "moments"]


def split(covers, where):
    oracles, asserted, regression, other = {}, {}, {}, {}
    for feat, fns in covers.items():
        for fn in fns:
            kind = fn.rsplit("_", 1)[-1]
            if kind in ORACLE_SUFFIX:
                asserted.setdefault(feat, set()).add(fn)
                oracles.setdefault(feat, set()).add(ORACLE_SUFFIX[kind])
            elif kind == "regression":
                regression.setdefault(feat, set()).add(fn)
            else:
                other.setdefault(feat, set()).add(fn)
    return oracles, asserted, regression, other


def render(rows, oracles, asserted, regression):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Notes"])
    for r in rows:
        f = r["feature"]
        note = ""
        if re.match(r"^(NORM_SPAT_MOMENT|IMOM_NRM)_\d\d$", f):
            note = ("normalized raw moment; skimage has no native function - moments_normalized() "
                    "is the central-moment quantity")
        elif re.match(r"^(WEIGHTED_|WT_NORM_CTR_MOM_|IMOM_W)", f):
            note = ("distance-to-contour weighted; dist comes from the approximate min_sqdist, "
                    "measured 1.372x over the exact distance, so no tool reproduces it")
        w.writerow(["2D", "moments", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    note])
    return buf.getvalue()


def unregistered(where):
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        reg = set(re.findall(r"(test_2d_moments_\w+)\s*\(\s*\)", fh.read()))
    return sorted(fn for fn, src in where.items() if src.endswith(".h") and fn not in reg)


def disagreements(rows, oracles, asserted, regression, other, where):
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
        for stale in sorted(claimed - files):
            out.append(f"{f}: current_test names {stale}, which covers nothing for it")
        for gap in sorted(files - claimed):
            out.append(f"{f}: {gap} covers it but current_test omits it")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and registry disagreements instead of rewriting")
    a = ap.parse_args(argv)

    covers, where, _ = scan()
    oracles, asserted, regression, other = split(covers, where)
    rows = registry_rows()
    text = render(rows, oracles, asserted, regression)
    problems = disagreements(rows, oracles, asserted, regression, other, where)
    problems += [f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs"
                 for fn in unregistered(where)]

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
