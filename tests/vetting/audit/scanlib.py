"""Shared machinery for the per-family `scan_*_coverage.py` artifacts. Stdlib only.

Every scanner reads the same thing out of the tree in the same way and differs only in which files
it reads, how the family spells its feature names in C++, and which of the acceptance checks it
runs. This module is that common part, held once, so a change to the coverage rule takes effect
everywhere it applies. A `scan_*_coverage.py` file is the family's declaration on top of it, plus
whatever is genuinely its own.

## The coverage rule

A feature is covered by a test function when its name appears on an ASSERTION line in that function,
or in a golden table that the function loops over while asserting. Comments are stripped first --
several of them name features they do not assert.

Deliberately NOT counted: a line that merely READS a feature out of the buffer. A
readout-counts-as-coverage rule credits an oracle test with vetting features it never checks --
report_feature_tests.py does exactly that, which is how the 2D morphology gap count came out two
rows short of the real one.

The kind of coverage comes from the function-name suffix, per SPEC 2 naming, so only an
oracle-suffixed function contributes an oracle token.

## What a row means

A registry row describes ONE assertion -- feature x config x reference (SPEC 3) -- so its
`current_test` names the file that assertion lives in, not every file that touches the feature. The
artifact this module renders is the other shape: a feature -> test rollup, one line per feature.

## Checks

`Family.checks` selects which acceptance checks run. The four in `CORE_CHECKS` are what every
scanner has always run. The three in `IDENTITY_CHECKS` -- SPEC 3's "a row must say which assertion
covers it" tier -- are opt-in, because only four families carry the `test_name` column densely
enough to pass them today. Adding a family to that tier is one word in its declaration; doing it for
a family whose registry rows are not ready would turn a real gap into twenty failing lines rather
than a plan, which is what `PR/todo.md` tracks instead.
"""
import argparse
import csv
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VETTING = os.path.dirname(HERE)
TESTS = os.path.dirname(VETTING)
REGISTRY = os.path.join(VETTING, "oracle_coverage.csv")
TEST_ALL = os.path.join(TESTS, "test_all.cc")

# `inline void` is a legal way to declare a case, so the optional keyword is part of the pattern --
# without it such a function is invisible and its features look uncovered.
FUNC = re.compile(r"^(?:inline\s+)?(?:void|def)\s+(test_\w+)|^\s+def\s+(test_\w+)", re.M)
# A module-level helper in the pytest files, e.g. `def _fd(label)`. The python tests read the
# dataframe column inside such a helper and assert on the returned scalar, so the feature name never
# appears in the test body -- credit the helper's features to every test function that calls it.
HELPER = re.compile(r"^def\s+(_\w+)\s*\(", re.M)
# Every top-level def, helper or test. A helper's body ends at the NEXT TOP-LEVEL DEF OF ANY KIND,
# which is what bounds it correctly -- ending it at the next *helper* instead lets the last helper in
# a file swallow every test function below it, so the helper picks up every feature name in the file
# and each test that calls it inherits the lot.
TOPLEVEL_DEF = re.compile(r"^def\s+\w+\s*\(", re.M)
# No trailing \b after `assert`: these families assert through per-oracle helpers
# (assert_gldzm_feature_mirp, assert_caliper_imea, ...), so the call line that names the feature is
# `assert_<something>(fvals, Feature2D::X, "X")`, not a bare ASSERT_ macro.
ASSERTION = re.compile(r"\b(ASSERT_|EXPECT_|assert)")
LOOP_LIST = re.compile(r"for\s*\([^)]*:\s*\{([^}]*)\}\s*\)"
                       r"|for\s+\w+\s+in\s*[\[(]([^\])]*)[\])]\s*:", re.S)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^\s*#[^\n]*", re.S | re.M)

CORE_CHECKS = frozenset({"vetted_no_oracle", "oracle_mismatch", "stale_current", "unregistered"})
# The whole-covering-set reading of `current_test`: every file that covers the feature must be
# named. Fifteen families read it that way. The four reworked under SPEC 3's one-row-one-assertion
# reading (3D gldm, glszm, ngldm, ngtdm, and 3D gldzm with them) deliberately do not -- there
# `current_test` names the file the row's own assertion lives in, and demanding the whole set is
# what put the parameterized sweep and the drift guard in one field. Those families take
# IDENTITY_CHECKS instead, which pin the assertion by name rather than by set membership.
SET_CHECKS = frozenset({"missing_current"})
# A row that names an oracle must name the file its own oracle assertion lives in. Only meaningful
# where a feature carries several oracle rows -- otherwise `missing_current` already says it.
ORACLE_FILE_CHECK = frozenset({"oracle_file"})
# The reverse of oracle_mismatch: an oracle-suffixed test asserts the feature while the row claims
# no oracle at all. Only 2D radial runs it today, where it is the guard that would catch an oracle
# quietly appearing for a family the registry still calls untested.
NO_ORACLE_CLAIMED_CHECK = frozenset({"no_oracle_claimed"})
IDENTITY_CHECKS = frozenset({"missing_test_name", "unresolved_test_name", "test_name_file"})
ALL_CHECKS = (CORE_CHECKS | SET_CHECKS | ORACLE_FILE_CHECK | NO_ORACLE_CLAIMED_CHECK
              | IDENTITY_CHECKS)
DEFAULT_CHECKS = CORE_CHECKS | SET_CHECKS


class Family:
    """One family's declaration. Everything else is in this module.

    dim / family      the registry rows this scanner owns.
    out               the artifact basename, written next to this module.
    sources           test files to scan, repo-tests-relative.
    oracle_suffix     {function-name suffix: oracle token}. Empty means the family has no oracle
                      test, and `--check` then enforces only that no row claims to be vetted.
    notes             {feature: note} or a callable(feature, ctx) -> str.
    table_owner       {golden-table name: the function to credit its keys to}, for a table keyed by
                      a foreign feature name that the asserting function never spells out.
    table_dialect     how to read such a table: "cpp" for a brace-initialiser list of {"KEY", value}
                      pairs ending in `};`, "python" for a dict literal ending in `}`.
    enum_dim_prefix   True for 3D families: C++ names the feature `Feature3D::GLCM_ASM` while the
                      registry carries the leading dimension digit, `3GLCM_ASM`.
    enum_alias        a family stem, when a test aliases the enum (`using F = Nyxus::Feature3D;`)
                      and so spells features `F::GLCM_ASM`.
    fn_prefix         the test-function name stem used to find registrations in test_all.cc.
    extra_column      (header, source) to append before Notes, where source is "other" -- the
                      functions whose kind is neither oracle nor regression. imq calls that column
                      Mechanics and 3D gldm calls it Invariant; both mean the same set.
    other_note        how the functions whose kind is neither oracle nor regression appear in the
                      Notes cell instead of a column: "asserted" names each one, "guarded" appends
                      the family's production-config phrasing, None says nothing.
    current_exempt    files `missing_current` does not fault a row for omitting. A drift guard is
                      not a vetting claim, so families whose `current_test` deliberately lists the
                      oracle files only name their regression and mechanics files here.
    uncredited        {file: why}, for a file that is scanned and reported but must never be
                      credited as coverage -- 2D radial's mechanics guards pin values a vetting
                      report shows are wrong, so crediting them would make those defects acceptance
                      criteria. Checked in both directions: omitting it is fine, naming it in
                      `current_test` is an error, so the decision cannot be reversed from the
                      registry alone.
    order             "registry" keeps the registry's order, "sorted" sorts by feature name.
    count_noun        what the --check summary counts, for families that count features not rows.
    checks            which acceptance checks to run; CORE_CHECKS | SET_CHECKS by default.
    scan_helpers      credit module-level python helpers to their callers (pytest sources only).
    featureset_loop   a compiled pattern for `for (auto fc : D3_X_feature::featureset)`. A function
                      that range-loops the calculator's own featureset asserts every feature of the
                      family while naming none of them, which is stronger than naming its members:
                      a feature added to the calculator is covered the day it is added.
    py_loop_tables    the Python twin of loop_tables: a pytest case that builds a local sequence
                      literal of column names and then range-loops it while asserting names no
                      feature on any assertion line, so the shared rule cannot see it. Opt-in for
                      the same reason as loop_tables -- a plain lookup list is not coverage.
    loop_tables       credit a table the function builds and then range-loops over while asserting
                      -- the equivalence tests put their feature pairs in a local vector, so the
                      names never reach an assertion line. Opt-in: a family without such a test does
                      not want a local lookup table read as coverage.
    extra_problems    callable(fam, cov) -> [str], for a check that is genuinely this family's own.
    boundary          how a feature name is delimited when matched; see feature_re().
    extra_summary     callable(cov) -> str or None, printed after the write line in rewrite mode.
    collect_override  callable(fam, feat_re) -> the five collect() maps, for a family whose tests do
                      not name features on assertion lines at all. 2D moments is the case: its
                      assertions pass a golden TABLE to a looping helper, so coverage is resolved
                      table by table rather than line by line.
    current_scope     what `missing_current` measures a covering file against. "row" is each row's
                      own `current_test`; "feature" is the union over the feature's rows, reported
                      once at the end. A family whose registry carries several oracle rows per
                      feature needs "feature", or every row is faulted for the files the others name.
    """

    def __init__(self, dim, family, out, sources, oracle_suffix, notes=None, table_owner=None,
                 table_dialect="cpp",
                 enum_dim_prefix=False, enum_alias=None, fn_prefix=None, extra_column=None,
                 other_note=None, order="registry", count_noun="rows", checks=None,
                 scan_helpers=False, loop_tables=False, py_loop_tables=False, featureset_loop=None,
                 extra_problems=None, collect_override=None, boundary="word", extra_summary=None,
                 current_scope="row",
                 current_exempt=(), uncredited=None):
        self.dim, self.family, self.out = dim, family, out
        self.sources, self.oracle_suffix = sources, oracle_suffix
        self.notes, self.table_owner = notes or {}, table_owner or {}
        self.table_dialect = table_dialect
        self.enum_dim_prefix, self.enum_alias = enum_dim_prefix, enum_alias
        self.fn_prefix = fn_prefix or f"test_{'3d_' if dim == '3D' else '2d_'}{family}"
        self.extra_column, self.other_note = extra_column, other_note
        self.current_exempt = frozenset(current_exempt)
        self.uncredited = uncredited or {}
        self.order, self.count_noun = order, count_noun
        self.checks = frozenset(DEFAULT_CHECKS if checks is None else checks)
        self.scan_helpers, self.loop_tables = scan_helpers, loop_tables
        self.py_loop_tables = py_loop_tables
        self.featureset_loop = featureset_loop
        self.extra_problems, self.current_scope = extra_problems, current_scope
        self.collect_override = collect_override
        self.boundary, self.extra_summary = boundary, extra_summary
        bad = self.checks - ALL_CHECKS
        assert not bad, f"unknown check(s): {sorted(bad)}"

    @property
    def out_path(self):
        return os.path.join(HERE, self.out)


class Coverage:
    """What the scan found, and the registry rows it is measured against."""

    def __init__(self, rows, asserted, oracles, regression, other, where):
        self.rows, self.where = rows, where
        self.asserted, self.oracles = asserted, oracles
        self.regression, self.other = regression, other

    def features(self, order):
        names = list(dict.fromkeys(r["feature"] for r in self.rows))
        return sorted(names) if order == "sorted" else names


def registry_rows(fam):
    with open(REGISTRY, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)
                if r["dim"] == fam.dim and r["family"] == fam.family]


def feature_re(names, boundary="word"):
    """Longest first, so GLDZM_SDLGLE wins over GLDZM_SDE and MAXCHORDS_MAX_ANG over MAXCHORDS_MAX.

    boundary="strict" opens with a negative lookbehind instead of \\b, so a name cannot match at the
    tail of a longer SCREAMING_SNAKE identifier. \\b does not stop that on its own -- the character
    before is a word character on both sides -- and longest-first settles it only when both names
    are in the set being matched.
    """
    ordered = sorted(names, key=len, reverse=True)
    open_ = r"(?<![A-Z0-9_])(" if boundary == "strict" else r"\b("
    return re.compile(open_ + "|".join(re.escape(n) for n in ordered) + r")\b")


def strip_comments(text):
    """Blank comments while preserving line count.

    The newline-preserving replacement is not cosmetic: the block splitter below indexes into this
    text, so collapsing lines would shift every function boundary after the first comment.
    """
    return COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


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



# `cols = [...]` / `ellipse = (...)` at the top of a pytest case: the literal is the whole
# right-hand side, so a call that merely takes a list argument (`row = _one([...], label)`) is not
# one of these.
PY_ASSIGN_LITERAL = re.compile(r"^[ 	]*\w+\s*=\s*[\[(]", re.M)
# `for c in cols:` -- a loop over a local NAME, which is what says the literal above is iterated
# rather than merely held.
PY_LOOP_NAME = re.compile(r"for\s+\w+\s+in\s+\w+\s*:")


def _bracketed(text, start):
    """The text between the bracket at `start` and its match, or "" if it never closes."""
    pairs = {"[": "]", "(": ")"}
    close = pairs[text[start]]
    depth, i = 1, start + 1
    while i < len(text) and depth:
        if text[i] == text[start]:
            depth += 1
        elif text[i] == close:
            depth -= 1
        i += 1
    return text[start + 1:i - 1] if not depth else ""


def py_literal_features(block, feat_re):
    """Feature names in the local sequence literals of a case that loops one of them while asserting.

    A pytest case can name every feature it compares in a list it builds and then iterates -- the
    ellipse tuple in test_2d_ooc_invariant.py is the case -- so no name ever reaches an assertion
    line and the shared rule sees nothing. This is the same concession loop_tables makes for the C++
    equivalence tests, and it is gated the same way: the function has to both assert and range-loop,
    so a plain lookup list is not mistaken for coverage.
    """
    if not PY_LOOP_NAME.search(block):
        return set()
    out = set()
    for m in PY_ASSIGN_LITERAL.finditer(block):
        out.update(feat_re.findall(_bracketed(block, m.end() - 1)))
    return out


def scan(fam, path, feat_re, all_names=()):
    """-> {test function name: {features it covers}}."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = strip_comments(fh.read())
    if fam.enum_dim_prefix:
        # 3D tests name features by enum (Feature3D::GLCM_ASM) while the registry carries the
        # leading dimension digit (3GLCM_ASM). Normalise so one pattern matches both spellings.
        text = text.replace("Feature3D::", "Feature3D::3")
    if fam.enum_alias:
        text = re.sub(r"\bF::(?=" + re.escape(fam.enum_alias) + r"_)", "F::3", text)

    hits = {}
    close, key = ("};", r'\{"([A-Z0-9_]+)"') if fam.table_dialect == "cpp" else ("}", r'"([A-Z0-9_]+)"')
    for table, owner in fam.table_owner.items():
        if table not in text:
            continue
        body = text.split(table, 1)[1].split("{", 1)[1].split(close, 1)[0]
        for m in re.finditer(key, body):
            hits.setdefault(owner, set()).add(m.group(1))
        text = text.replace(body, "")   # keep the table out of the block scan below

    helpers = helper_features(text, feat_re) if fam.scan_helpers else {}

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
        if fam.py_loop_tables and path.endswith(".py"):
            hits.setdefault(fn, set()).update(py_literal_features(block, feat_re))
        if fam.featureset_loop and fam.featureset_loop.search(block):
            hits.setdefault(fn, set()).update(all_names)
        # A table the function builds and then loops over while asserting -- the equivalence tests
        # put their feature pairs in `std::vector<Pair> pairs = { ... };` and assert inside the
        # loop, so the names never appear on an assertion line. Credited only when the function
        # both asserts and range-loops, so a plain lookup table is not mistaken for coverage.
        if fam.loop_tables and re.search(r"for\s*\([^)]*:\s*\w+\s*\)", block):
            for m in re.finditer(r"=\s*\{(.*?)\};", block, re.S):
                hits.setdefault(fn, set()).update(feat_re.findall(m.group(1)))
        for name, feats in helpers.items():
            if re.search(r"\b" + re.escape(name) + r"\s*\(", block):
                hits.setdefault(fn, set()).update(feats)
    return hits


def collect(fam, feat_re, all_names=()):
    """-> Coverage, with each covering function bucketed by the kind its name-suffix declares."""
    asserted, oracles, regression, other, where = {}, {}, {}, {}, {}
    for rel in fam.sources:
        for fn, feats in scan(fam, os.path.join(TESTS, rel), feat_re, all_names).items():
            where[fn] = os.path.basename(rel)
            kind = fn.rsplit("_", 1)[-1]
            for feat in feats:
                if kind in fam.oracle_suffix:
                    asserted.setdefault(feat, set()).add(fn)
                    oracles.setdefault(feat, set()).add(fam.oracle_suffix[kind])
                elif kind == "regression":
                    regression.setdefault(feat, set()).add(fn)
                else:                   # invariant / mechanics - coverage, never vetting
                    other.setdefault(feat, set()).add(fn)
    return asserted, oracles, regression, other, where


def note_for(fam, feature, cov):
    """The Notes cell: the family's own note, plus whatever it says about the other-kind functions.

    A function whose name-suffix is neither an oracle nor `regression` contributes coverage but no
    oracle token, so without one of these it would be invisible in the artifact -- which is how a
    second assertion on the same golden stops being visible as something to re-tighten together.
    """
    note = fam.notes(feature, cov) if callable(fam.notes) else fam.notes.get(feature, "")
    others = sorted(cov.other.get(feature, ()))
    if not others or not fam.other_note:
        return note
    if fam.other_note == "guarded":
        return note + "; also guarded on the production config by " + ";".join(others)
    parts = [note] if note else []
    parts += [f"also asserted by {fn} (kind is neither oracle nor regression)" for fn in others]
    return " | ".join(parts)


def dual_oracle_notes(notes, dual):
    """A `notes` callable for the families that mark every dual-oracle feature the same way.

    Four of them assert the same feature against both IBSI and mirp on purpose -- IBSI fixes the
    DEFINITION at its published three-significant-figure precision, mirp fixes the DIGITS at the
    exact tier -- so the artifact says so on every such row rather than only where NOTE has an
    entry. An explicit NOTE still wins, because it says something this default cannot.
    """
    def note(feature, cov):
        if feature in notes:
            return notes[feature]
        return dual if len(cov.oracles.get(feature, ())) > 1 else ""
    return note


def render(fam, cov):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    head = ["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
            "Regression", "Reg_Test_Name"]
    if fam.extra_column:
        head.append(fam.extra_column)
    w.writerow(head + ["Notes"])
    for f in cov.features(fam.order):
        row = [fam.dim, fam.family, f,
               ";".join(sorted(cov.oracles.get(f, ()))),
               ";".join(sorted(cov.asserted.get(f, ()))),
               "Y" if f in cov.regression else "N",
               ";".join(sorted(cov.regression.get(f, ())))]
        if fam.extra_column:
            row.append(";".join(sorted(cov.other.get(f, ()))))
        w.writerow(row + [note_for(fam, f, cov)])
    return buf.getvalue()


def _test_all():
    with open(TEST_ALL, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def unregistered_tests(fam, cov):
    """gtest functions that exist but no TEST() in test_all.cc calls - they never run.

    Only the C++ headers are checked; pytest collects the .py functions by name. 3D GLDZM is why
    this check matters: 3GLDZM_ZDM carried a golden with no function and no registration, so a
    value 14.5x off could not fail.
    """
    registered = set(re.findall(r"(" + fam.fn_prefix + r"_\w+)\s*\(\s*\)", _test_all()))
    return sorted(fn for fn, src in cov.where.items()
                  if src.endswith(".h") and fn not in registered)


def case_to_file(fam, cov):
    """-> {gtest case name: the source file defining the function it calls}.

    test_all.cc registers `TEST(SUITE, CASE) { ASSERT_NO_THROW(fn()); }`, and `where` already maps a
    test function to the file that defines it, so the two compose into case -> file. That is what
    lets the identity checks confirm a row's test_name and its current_test describe the same
    assertion rather than merely both being true of the feature.
    """
    out = {}
    for suite, case, body in re.findall(
            r"TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{(.*?)\n\}", _test_all(), re.S):
        for fn in re.findall(r"(" + fam.fn_prefix + r"_\w+)\s*\(\s*\)", body):
            if fn in cov.where:
                out[f"{suite}.{case}"] = cov.where[fn]
    return out


def disagreements(fam, cov):
    out = []
    cases = case_to_file(fam, cov) if fam.checks & IDENTITY_CHECKS else {}
    claimed_by_feature = {}
    for r in cov.rows:
        f = r["feature"]
        covering = (cov.asserted.get(f, set()) | cov.regression.get(f, set())
                    | cov.other.get(f, set()))
        files = {cov.where[fn] for fn in covering}
        claimed = {t for t in r["current_test"].split(";") if t}
        claimed_by_feature.setdefault(f, set()).update(claimed)

        if "vetted_no_oracle" in fam.checks and r["status"] == "vetted" and not cov.asserted.get(f):
            out.append(f"{f}: status=vetted but no oracle test asserts it")
        if "oracle_mismatch" in fam.checks and r["oracle"]                 and r["oracle"] not in cov.oracles.get(f, set()):
            out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                       f"{sorted(cov.oracles.get(f, ())) or 'none'}")
        if "no_oracle_claimed" in fam.checks and not r["oracle"] and cov.oracles.get(f):
            out.append(f"{f}: registry claims no oracle but {sorted(cov.oracles[f])} test(s) assert "
                       f"it under an oracle-suffixed name")
        if "oracle_file" in fam.checks and r["oracle"]:
            oracle_files = {cov.where[fn] for fn in cov.asserted.get(f, set())
                            if fam.oracle_suffix.get(fn.rsplit("_", 1)[-1]) == r["oracle"]}
            if oracle_files and not (claimed & oracle_files):
                out.append(f"{f}: oracle={r['oracle']!r} row omits its assertion file "
                           f"{sorted(oracle_files)}")
        if "stale_current" in fam.checks:
            for stale in sorted(claimed - files):
                out.append(f"{f}: current_test names {stale}, which covers nothing for it")
        for named in sorted(claimed & set(fam.uncredited)):
            out.append(f"{f}: current_test names {named}, which is uncredited on purpose - "
                       f"{fam.uncredited[named]}")
        if "missing_current" in fam.checks and fam.current_scope == "row":
            for gap in sorted(files - claimed - fam.current_exempt - set(fam.uncredited)):
                out.append(f"{f}: {gap} covers it but current_test omits it")

        # The row describes one assertion, so its two identifiers must agree: the file named in
        # current_test is the file the case named in test_name is defined in.
        name = r.get("test_name", "")
        if not name:
            if "missing_test_name" in fam.checks:
                out.append(f"{f}: no test_name, so current_test names an assertion nothing identifies")
        elif "unresolved_test_name" in fam.checks and name not in cases:
            out.append(f"{f}: test_name {name} resolves to no registered case in test_all.cc")
        elif "test_name_file" in fam.checks and name in cases and cases[name] not in claimed:
            out.append(f"{f}: test_name {name} is defined in {cases[name]}, which current_test "
                       f"({r['current_test'] or 'empty'}) does not name")

    if "missing_current" in fam.checks and fam.current_scope == "feature":
        for f in {r["feature"] for r in cov.rows}:
            covering = (cov.asserted.get(f, set()) | cov.regression.get(f, set())
                        | cov.other.get(f, set()))
            files = {cov.where[fn] for fn in covering}
            for gap in sorted(files - claimed_by_feature.get(f, set())):
                out.append(f"{f}: {gap} covers it but current_test omits it")
    return out


def problems(fam, cov):
    """The built-in checks, then the family's own, then the never-runs sweep.

    extra_problems comes before the sweep because a family that replaces the built-in checks
    entirely -- 2D NGTDM reads `current_test` per test KIND rather than per feature -- still wants
    the sweep reported last, where every other family has it.
    """
    out = disagreements(fam, cov)
    if fam.extra_problems:
        out += fam.extra_problems(fam, cov)
    if "unregistered" in fam.checks:
        out += [f"{fn}: defined but no TEST() in test_all.cc calls it, so it never runs"
                for fn in unregistered_tests(fam, cov)]
    return out


def run(fam, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and registry disagreements instead of rewriting")
    a = ap.parse_args(argv)

    rows = registry_rows(fam)
    names = {r["feature"] for r in rows}
    feat_re = feature_re(names, fam.boundary)
    if fam.collect_override:
        cov = Coverage(rows, *fam.collect_override(fam, feat_re))
    else:
        cov = Coverage(rows, *collect(fam, feat_re, names))
    text = render(fam, cov)
    probs = problems(fam, cov)
    n = len(rows) if fam.count_noun == "rows" else len(cov.features(fam.order))

    if a.check:
        if not os.path.exists(fam.out_path):
            probs.insert(0, f"{fam.out} is missing; run without --check")
        else:
            with open(fam.out_path, newline="", encoding="utf-8") as fh:
                if fh.read() != text:
                    probs.insert(0, f"{fam.out} is stale; rerun without --check")
        for p in probs:
            print("ERROR:", p)
        print(f"checked {n} {fam.count_noun}: "
              f"{'clean' if not probs else str(len(probs)) + ' problem(s)'}")
        return 1 if probs else 0

    with open(fam.out_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {fam.out_path} ({n} {fam.count_noun})")
    if fam.extra_summary:
        extra = fam.extra_summary(cov)
        if extra:
            print(extra)
    for p in probs:
        print("WARNING:", p)
    return 0
