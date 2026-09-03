"""Regenerate imq_coverage.csv by scanning the IMQ tests. Stdlib only.

    python tests/vetting/audit/scan_imq_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting. The rendering and
the run loop live in scanlib.py; the reading and the acceptance model below are this family's own.

Each row is checked against the tests of its OWN kind -- a vetted row against the oracle files, a
regression row against the snapshot one -- so `current_test` must name exactly the files that cover
the feature AT THAT ROW'S CONFIG. That is stricter than scanlib's per-feature reading, which is why
this family replaces the shared checks rather than configuring them.

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
import os
import re
import sys

import scanlib

FEATURESET = os.path.join(os.path.dirname(scanlib.TESTS), "src", "nyx", "featureset.cpp")

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
# The golden-table key a test reads, as the quoted SCREAMING_SNAKE literal it passes to its helper.
PIN_KEY = re.compile(r'"([A-Z][A-Z0-9_]*)"')
# SCOPED_TRACE labels are the other SCREAMING_SNAKE literal in these functions and pin nothing, so
# they are dropped before the keys are read - otherwise every trace label reads as an unpinned key.
TRACE = re.compile(r"SCOPED_TRACE\s*\(.*?\)\s*;", re.S)
# Block comments only: these headers carry no `#` comments, and stripping `#` lines would take the
# preprocessor directives with them.
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

NOTE = {
    "LOCAL_FOCUS_SCORE": "opencv covers the per-tile statistic only: the tile extraction and "
                         "the scale^2 divisor are Nyxus' own definition, reproduced in the "
                         "generator. See not_covered.md section E",
    "POWER_SPECTRUM_SLOPE": "two cells, both regression: the 8 px fixture pins the GUARD's return "
                            "value (rps() returns early below a 24 px short side, so the 0 is the "
                            "early return), and a 24x24 ROI pins the algorithm past it at "
                            "1.7837481542489078. Neither is vetted. See matrix/imq.md",
    "SHARPNESS": "candidate oracle measured and refuted: the reference DOM implementation returns "
                 "0.5459 against Nyxus' 2.1905 on this fixture. See "
                 "audit/imq_pydom_sharpness_vetting_report.md",
}

# Filled by collect() and read by disagreements(). The two run one after the other inside
# scanlib.run, and the key/reader sets are a by-product of the scan that the five coverage maps
# have nowhere to carry.
_KEYS, _READ_KEYS = {}, {}


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
    """-> ({test fn: {features it covers}}, {test fn: {pin keys it names}}, {keys the table pins})."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), fh.read())
    keys = table_keys(text, TABLE_OF[os.path.basename(path)])

    hits, read = {}, {}
    marks = [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
    for i, (pos, fn) in enumerate(marks):
        block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
        if not ASSERTION.search(block):
            continue
        for line in block.splitlines():
            if ASSERTION.search(line):
                hits.setdefault(fn, set()).update(feat_re.findall(line))
        # The pin key a function reads, taken from the literal it passes rather than inferred from
        # the feature it names. A cell outside the family's one recipe pins a QUALIFIED key -- the
        # feature plus the cell, e.g. MIN_SATURATION_CONSTANT_ROI -- so a feature name no longer
        # identifies the pin, and matching on the feature would report every such key as unread.
        # Read over the whole block, not per line: these calls wrap.
        read[fn] = set(PIN_KEY.findall(TRACE.sub("", block)))
    return hits, read, keys


def collect(fam, feat_re):
    """-> the five coverage maps, recording each file's pinned and read keys on the way through."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    _KEYS.clear()
    _READ_KEYS.clear()
    for rel in fam.sources:
        hits, read, table = scan(os.path.join(scanlib.TESTS, rel), feat_re)
        _KEYS[rel] = table
        _READ_KEYS[rel] = set().union(*read.values()) if read else set()
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
    return asserted, oracles, regression, other, where


def registered_cases():
    """-> {function name: gtest case} for every TEST() body in test_all.cc.

    Every registered name, not just this family's, so a scanner later pointed at another family's
    file does not report that file's cases as never running.
    """
    with open(scanlib.TEST_ALL, encoding="utf-8", errors="replace") as fh:
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


def key_reader_problems():
    """A pinned key nothing reads, and an assertion with nothing pinned for it.

    Two set differences per file. The first direction is the one that matters: an assertion never
    evaluates a key it does not name, so a key sitting at another feature's value is invisible to
    every other check in this file.

    Compared on the KEYS themselves, taken from the literals the tests pass. Comparing on feature
    names instead would collapse every cell of a feature onto one name, and the family pins more
    than one cell per feature -- MIN_SATURATION alone is pinned at three.
    """
    out = []
    for rel, pinned in _KEYS.items():
        if pinned is None:
            out.append(f"{rel}: golden table {TABLE_OF[rel]} not found - the scanner cannot check "
                       f"its keys against the assertions that read them")
            continue
        read = _READ_KEYS.get(rel, set())
        for k in sorted(pinned - read):
            out.append(f"{rel}: {TABLE_OF[rel]} pins {k} but no assertion in the file reads it, "
                       f"so nothing ever evaluates that number")
        for k in sorted(read - pinned):
            out.append(f"{rel}: an assertion names {k} but {TABLE_OF[rel]} pins nothing for it")
    return out


def disagreements(fam, cov):
    """Each row is answerable for the tests of its own kind."""
    out = []
    for r in cov.rows:
        f = r["feature"]
        claimed = {t for t in r["current_test"].split(";") if t}
        if r["status"] == "vetted":
            files = {cov.where[fn] for fn in cov.asserted.get(f, ())}
            if not cov.asserted.get(f):
                out.append(f"{f}: status=vetted but no oracle test asserts it")
            if r["oracle"] and r["oracle"] not in cov.oracles.get(f, set()):
                out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                           f"{sorted(cov.oracles.get(f, ())) or 'none'}")
            if not r["oracle"]:
                out.append(f"{f}: status=vetted but the row names no oracle")
        else:
            files = {cov.where[fn] for fn in cov.regression.get(f, ())}
            if r["oracle"]:
                out.append(f"{f}: status={r['status']} but the row names oracle {r['oracle']!r}")
        for stale in sorted(claimed - files):
            out.append(f"{f}: current_test names {stale}, which covers nothing for it at "
                       f"recipe {r['config_recipe'] or '(none)'}")
        for gap in sorted(files - claimed):
            out.append(f"{f}: {gap} covers it but current_test omits it")

    # the reverse gap: a kind of test with no row to answer for it
    for f in sorted(cov.asserted):
        if not any(r["feature"] == f and r["status"] == "vetted" for r in cov.rows):
            out.append(f"{f}: {sorted(cov.asserted[f])} assert it but no registry row is vetted")
    for f in sorted(cov.regression):
        if not any(r["feature"] == f and r["status"] == "regression" for r in cov.rows):
            out.append(f"{f}: {sorted(cov.regression[f])} pin it but no registry row is a "
                       f"regression one")

    out += key_reader_problems()

    # registration, both directions
    registered = registered_cases()
    for fn, src in sorted(cov.where.items()):
        if src.endswith(".h") and fn not in registered:
            out.append(f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs")
    for r in cov.rows:
        for name in (t.strip() for t in r["test_name"].split(";") if t.strip()):
            case = name.split(".")[-1]
            if case not in set(registered.values()):
                out.append(f"{r['feature']}: test_name names case {case}, which test_all.cc does "
                           f"not declare")

    # the registry and the enum must name the same features, in BOTH directions
    enum = enum_features()
    rowset = {r["feature"] for r in cov.rows}
    for f in sorted(rowset - enum):
        out.append(f"{f}: a registry row names it but featureset.cpp maps no FeatureIMQ of that name")
    for f in sorted(enum - rowset):
        out.append(f"{f}: FeatureIMQ publishes it but no registry row covers it")
    return out


FAMILY = scanlib.Family(
    dim="IMQ", family="imq", out="imq_coverage.csv",
    sources=SOURCES,
    oracle_suffix=ORACLE_SUFFIX,
    notes=NOTE,
    collect_override=collect,
    # the mechanics guards are neither oracle nor regression, and this family names them in a
    # column of their own rather than folding them into the notes
    extra_column="Mechanics",
    # every built-in check is replaced by the per-kind model above, registration included
    checks=frozenset(),
    extra_problems=disagreements,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
