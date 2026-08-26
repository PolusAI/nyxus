"""Regenerate imq_coverage.csv by scanning the IMQ tests. Stdlib only.

    python tests/vetting/audit/scan_imq_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check from the family plan: every `vetted` row in oracle_coverage.csv must be asserted by
an oracle test, that test's oracle must be the one the row names, and `current_test` must name
exactly the files that cover the feature AT THAT ROW'S CONFIG. Each row is checked against the tests
of its OWN kind -- a vetted row against the oracle files, a regression row against the snapshot one.

Coverage rule: a feature is covered by a test function when its name appears on an ASSERTION line in
that function. Comments are stripped first -- several of them name features they do not assert.
The kind of coverage comes from the function-name suffix, per SPEC 2 naming, so only an
oracle-suffixed function contributes an oracle token.

THREE CHECKS BEYOND THE ROW MAPPING, all set differences and all cheap:

  keys vs readers        every key in a golden table must be read by an assertion in the same file,
                         and every feature an assertion names must have a key in that file's table.
                         A pinned key nothing reads is where a bad number lives, because no
                         assertion ever evaluates it -- that is how a 3D GLDM golden sat at another
                         feature's value, 353x off, under a function that asserted the other one.
  readers vs registration a test function no TEST() calls never runs.
  registry vs the enum   the IMQ rows and Nyxus::UserFacingFeatureNames must name the same six
                         features, checked in BOTH directions. Comparing counts instead of sets
                         cannot see a swap, and a braced list's last entry has no trailing comma,
                         so counting by comma is off by one to begin with.
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
REPO = os.path.dirname(TESTS)
OUT = os.path.join(HERE, "imq_coverage.csv")
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")
FEATURESET = os.path.join(REPO, "src", "nyx", "featureset.cpp")

SOURCES = [
    "test_imq_opencv.h",
    "test_imq_cellprofiler.h",
    "test_imq_regression.h",
]
# The golden table each file carries, so a key can be checked against the assertions that read it.
TABLE_OF = {
    "test_imq_opencv.h": "imq_opencv_ref_vals",
    "test_imq_cellprofiler.h": "imq_cellprofiler_ref_vals",
    "test_imq_regression.h": "imq_regression_ref_vals",
}

ORACLE_SUFFIX = {"opencv": "opencv", "cellprofiler": "cellprofiler"}

FUNC = re.compile(r"^(?:inline\s+)?(?:static\s+)?void\s+(test_\w+)", re.M)
# This family asserts through per-oracle helpers (assert_imq_opencv, assert_imq_cellprofiler,
# assert_imq_regression), so the line that names the feature is the helper call, not a bare
# ASSERT_ macro. No trailing \b after `assert` for the same reason.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

NOTE = {
    "LOCAL_FOCUS_SCORE": "opencv covers the per-tile statistic only: the tile extraction and "
                         "the scale^2 divisor are Nyxus' own definition, reproduced in the "
                         "generator. See not_covered.md section E",
    "POWER_SPECTRUM_SLOPE": "pinned at the guard, not the algorithm: rps() returns early below a "
                            "24 px short side, so the 0 is the early return. See matrix/imq.md",
    "SHARPNESS": "candidate oracle measured and refuted: the reference DOM implementation returns "
                 "0.5459 against Nyxus' 2.1905 on this fixture. See "
                 "audit/imq_pydom_sharpness_vetting_report.md",
}


def registry_rows():
    with open(REGISTRY, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["dim"] == "IMQ" and r["family"] == "imq"]


def feature_names():
    """The family's feature names, longest first so a name that prefixes another cannot win."""
    return sorted({r["feature"] for r in registry_rows()}, key=len, reverse=True)


def feature_re(names):
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")


def strip_comments(text):
    return COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def table_keys(text, table):
    """The keys a golden table pins, brace-counted rather than matched with a non-greedy regex."""
    m = re.search(re.escape(table) + r"\s*\{", text)
    if not m:
        return None
    depth, j = 1, m.end()
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return set(re.findall(r'\{\s*"(\w+)"\s*,', text[m.end():j - 1]))


def scan(path, feat_re):
    """-> ({test function name: {features it covers}}, {keys the file's table pins})."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = strip_comments(fh.read())
    keys = table_keys(text, TABLE_OF[os.path.basename(path)])

    hits = {}
    marks = [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
    for i, (pos, fn) in enumerate(marks):
        block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
        if not ASSERTION.search(block):
            continue
        for line in block.splitlines():
            if ASSERTION.search(line):
                hits.setdefault(fn, set()).update(feat_re.findall(line))
    return hits, keys


def collect(feat_re):
    """-> (oracle fns, oracle tokens, regression fns, other-kind fns, fn -> file, file -> keys)."""
    asserted, oracles, regression, other, where, keys = {}, {}, {}, {}, {}, {}
    for rel in SOURCES:
        hits, table = scan(os.path.join(TESTS, rel), feat_re)
        keys[rel] = table
        for fn, feats in hits.items():
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
    return asserted, oracles, regression, other, where, keys


def render(rows, asserted, oracles, regression, other):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                "Regression", "Reg_Test_Name", "Mechanics", "Notes"])
    for f in dict.fromkeys(r["feature"] for r in rows):
        w.writerow(["IMQ", "imq", f,
                    ";".join(sorted(oracles.get(f, ()))),
                    ";".join(sorted(asserted.get(f, ()))),
                    "Y" if f in regression else "N",
                    ";".join(sorted(regression.get(f, ()))),
                    ";".join(sorted(other.get(f, ()))),
                    NOTE.get(f, "")])
    return buf.getvalue()


def registered_cases():
    """-> {function name: gtest case} for every TEST() body in test_all.cc.

    Every registered name, not just this family's, so a scanner later pointed at another family's
    file does not report that file's cases as never running.
    """
    with open(os.path.join(TESTS, "test_all.cc"), encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out = {}
    for case, body in re.findall(r"TEST\s*\(\s*\w+\s*,\s*(\w+)\s*\)\s*\{(.*?)\n\}", text, re.S):
        for fn in re.findall(r"(test_\w+)\s*\(\s*\)", body):
            out[fn] = case
    return out


def enum_features():
    """The IMQ names Nyxus::UserFacingFeatureNames maps, read out of featureset.cpp."""
    with open(FEATURESET, encoding="utf-8", errors="replace") as fh:
        return set(re.findall(r'\{\s*"(\w+)"\s*,\s*FeatureIMQ::\w+\s*\}', fh.read()))


def key_reader_problems(keys, asserted, regression, other, where):
    """A pinned key nothing reads, and an assertion with nothing pinned for it.

    Three set differences per file. The first direction is the one that matters: an assertion never
    evaluates a key it does not name, so a key sitting at another feature's value is invisible to
    every other check in this file.
    """
    out = []
    read_by_file = {}
    for bucket in (asserted, regression, other):
        for feat, fns in bucket.items():
            for fn in fns:
                read_by_file.setdefault(where[fn], set()).add(feat)
    for rel, pinned in keys.items():
        if pinned is None:
            out.append(f"{rel}: golden table {TABLE_OF[rel]} not found - the scanner cannot check "
                       f"its keys against the assertions that read them")
            continue
        read = read_by_file.get(rel, set())
        for k in sorted(pinned - read):
            out.append(f"{rel}: {TABLE_OF[rel]} pins {k} but no assertion in the file reads it, "
                       f"so nothing ever evaluates that number")
        for k in sorted(read - pinned):
            out.append(f"{rel}: an assertion names {k} but {TABLE_OF[rel]} pins nothing for it")
    return out


def disagreements(rows, asserted, oracles, regression, other, where, keys):
    """Each row is answerable for the tests of its own kind."""
    out = []
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

    out += key_reader_problems(keys, asserted, regression, other, where)

    # registration, both directions
    registered = registered_cases()
    for fn, src in sorted(where.items()):
        if src.endswith(".h") and fn not in registered:
            out.append(f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs")
    for r in rows:
        for name in (t.strip() for t in r["test_name"].split(";") if t.strip()):
            case = name.split(".")[-1]
            if case not in set(registered.values()):
                out.append(f"{r['feature']}: test_name names case {case}, which test_all.cc does "
                           f"not declare")

    # the registry and the enum must name the same features, in BOTH directions
    enum = enum_features()
    rowset = {r["feature"] for r in rows}
    for f in sorted(rowset - enum):
        out.append(f"{f}: a registry row names it but featureset.cpp maps no FeatureIMQ of that name")
    for f in sorted(enum - rowset):
        out.append(f"{f}: FeatureIMQ publishes it but no registry row covers it")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and registry disagreements instead of rewriting")
    a = ap.parse_args(argv)

    rows = registry_rows()
    feat_re = feature_re(feature_names())
    asserted, oracles, regression, other, where, keys = collect(feat_re)
    text = render(rows, asserted, oracles, regression, other)
    problems = disagreements(rows, asserted, oracles, regression, other, where, keys)

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
