"""Generate features.csv and features.md -- the vetting registry joined to what the tree asserts.

    python tests/vetting/report_features.py --write     # regenerate both artifacts
    python tests/vetting/report_features.py --check     # fail if either is stale, or a verdict is
                                                        # not `agree` and not in the allowlist

`oracle_coverage.csv` records what a row CLAIMS: a status, an oracle, a tolerance, a recipe. The
per-family scanners record what the TREE ASSERTS: which gtest case covers which feature, under which
oracle, with which drift guard behind it. Neither file can check the other.

This joins them and adds the one column neither side holds: `verdict`, which says whether they agree.
That is what makes the report a gate rather than a summary -- `--check` fails on any row whose
verdict is not `agree` and is not in ALLOWLIST with a written reason.

## The key

One row per (dim, feature, oracle, config_recipe). The config cell is part of the identity of an
assertion, not a note on it: a feature vetted at `ibsi=true` can diverge by two orders of magnitude
at `ibsi=false` (measured on glrlm, glszm and gldm), so "vetted" without the cell it was vetted in is
not a fact. That key is already unique across the registry; (dim, feature, oracle) is not.

## How far the verdict reaches, which is not everywhere

Being the key is not the same as being checked, and `verdict_scope` says which of the three a row
got, per row, rather than leaving the strongest to be assumed of all of them:

  `feature`     the tree was read for the FEATURE. The row claims an oracle and some oracle test
                asserts that feature -- not necessarily the assertion this row describes. Every row
                whose `test_name` is empty gets this, and 438 of them are.
  `row`         `test_name` named a gtest case, the case resolved through test_all.cc to the
                function it runs, and THAT function asserts the feature at the row's own kind.
  `row+config`  and the function is one the family's `recipe_reader` says asserts at the row's
                `config_recipe`. This is the only scope in which the config cell is checked at all:
                a recipe is a claim about how the calculator was configured, and nothing in the
                tree records a configuration except which function the numbers were read in. In a
                family that declares no reader, swapping a row's recipe for another of that
                family's changes no verdict -- which is why the scope is a column and not a
                footnote.

## Provenance is visible in the row

Columns are grouped `claim_*` / `scan_*` / derived, so a reader can see which side of the join a
value came from without consulting the source. A `scan_*` column is a fact about the test tree at
the moment the report was generated; a `claim_*` column is a human assertion reviewed in a PR.

## What this does NOT do

It does not replace `oracle_coverage.csv`. That file stays the one thing a human edits, and stays
small; this file is generated and safe to regenerate on every merge. Nothing here should ever be
edited by hand.
"""
import argparse
import collections
import csv
import glob
import importlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit")
TESTS = os.path.dirname(HERE)
REGISTRY = os.path.join(HERE, "oracle_coverage.csv")
OUT_CSV = os.path.join(HERE, "features.csv")
OUT_MD = os.path.join(HERE, "features.md")

sys.path.insert(0, AUDIT)
import scanlib                                                      # noqa: E402
import check_coverage                                               # noqa: E402

COLUMNS = [
    # identity
    "dim", "family", "feature",
    # the claim, from oracle_coverage.csv
    "claim_status", "claim_oracle", "claim_recipe", "claim_tolerance", "claim_agreement",
    "claim_benchmark", "claim_test_name", "claim_flag", "claim_candidate_oracle", "claim_source",
    # the tree, from the family's scanner
    "scan_oracles", "scan_oracle_tests", "scan_regression_tests", "scan_other_tests",
    "scan_n_oracle", "scan_n_regression",
    # the tree, scoped to THIS row: the functions its own test_name runs
    "scan_row_tests",
    # the join
    "verdict", "verdict_scope",
    # where to read more
    "vetting_report", "golden_regen", "matrix",
    "claim_notes",
]

# Family x dim combinations with a coverage artifact but no scanner behind it. Their
# audit/<family>_<dim>_coverage.csv was written by a retired pipeline or by hand, so nothing
# re-derives it from the tree and `--check` cannot tell whether it is still true. Listed rather than
# silently skipped: this is the largest remaining hole in the report, and naming it is what keeps it
# from reading as coverage.
KNOWN_UNSCANNED = {
    ("2D", "firstorder"): "no scan_firstorder_coverage.py; artifact predates the scanner series",
    ("3D", "firstorder"): "no scan_firstorder3d_coverage.py; same",
    ("2D", "gabor"): "no scan_gabor_coverage.py; one feature, artifact written by hand",
    ("2D", "glcm"): "no scan_glcm_coverage.py, though the 3D twin has one",
    ("2D", "glrlm"): "no scan_glrlm_coverage.py, though the 3D twin has one",
}

# A verdict other than `agree` fails --check unless the (dim, feature, verdict) is listed here with
# a reason. The allowlist is the honest form of a known gap: it says the row is wrong, why, and that
# somebody decided to leave it. It shrinking to zero is the end state.
ALLOWLIST = {
    ("2D", f, "oracle-mismatch"): (
        "promoted by the Octave/MATLAB harness in nyxus/octave, which is deliberately outside the "
        "test tree, so no in-tree assertion carries the matlab token (source=tracker)")
    for f in ("P01", "P10", "P25", "P75", "P90", "P99",
              "INTERQUARTILE_RANGE", "QCOD", "ROBUST_MEAN")
}


def families():
    """-> [Family], one per audit/scan_*_coverage.py, by importing each scanner's declaration.

    Importing rather than re-implementing is the point of the shared library: the report reads the
    tree through exactly the code the per-family --check runs, so the two cannot disagree about what
    a test covers. Each scanner guards its entry point with __main__, so importing runs nothing.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(AUDIT, "scan_*_coverage.py"))):
        mod = importlib.import_module(os.path.basename(path)[:-3])
        fam = getattr(mod, "FAMILY", None)
        if fam is None:
            raise SystemExit(f"{os.path.basename(path)} declares no FAMILY; it cannot be reported")
        out.append(fam)
    return out


class Scanned:
    """A family's coverage, with the two things a ROW-scoped read of it needs.

    `cov` answers "what covers this feature"; `cases` answers "what does THIS row's test_name run",
    and `fam` carries the declaration the config check reads. Coverage alone can only ever be
    feature-wide, which is why it is not the whole of what the report joins against.
    """

    def __init__(self, fam, cov):
        self.fam, self.cov = fam, cov
        self.cases = scanlib.case_to_fns(fam, cov)


def scan_all():
    """-> {(dim, family): Scanned} for every family that has a scanner."""
    cov = {}
    for fam in families():
        rows = scanlib.registry_rows(fam)
        names = {r["feature"] for r in rows}
        feat_re = scanlib.feature_re(names, fam.boundary)
        gather = fam.collect_override or (lambda f, r: scanlib.collect(f, r, names))
        cov[(fam.dim, fam.family)] = Scanned(fam, scanlib.Coverage(rows, *gather(fam, feat_re)))
    return cov


def registry():
    with open(REGISTRY, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def dim_token(dim):
    """The token the audit filenames use. IMQ files carry no dimension at all."""
    return {"2D": "2d", "3D": "3d"}.get(dim, "")


def pointers(family, dim, oracles):
    """-> (vetting_report, golden_regen, matrix), each a repo-relative path or "".

    `oracles` is the row's claimed oracle first, then whatever the tree asserts against. The claim
    is a preference rather than the key: a family whose oracle DISAGREES with Nyxus deliberately
    claims none -- 3D gldzm and 3D ngldm both do -- and the report that recorded the disagreement is
    still the document to read for those rows.
    """
    d = dim_token(dim)
    stem = f"{family}_{d}" if d else family

    report = ""
    for cand in (f"{stem}_{o}_vetting_report.md" for o in oracles if o):
        if os.path.exists(os.path.join(AUDIT, cand)):
            report = f"audit/{cand}"
            break
    if not report:
        # neither side names an oracle. One report for the family x dim is unambiguously the one to
        # read; several are not, and an arbitrary pick would read as a considered pointer.
        found = sorted(glob.glob(os.path.join(AUDIT, f"{stem}_*_vetting_report.md")))
        if len(found) == 1:
            report = f"audit/{os.path.basename(found[0])}"

    regen = f"{stem}_golden_regen.md"
    regen = f"audit/{regen}" if os.path.exists(os.path.join(AUDIT, regen)) else ""

    # matrix files are named by family, with 3D families suffixed rather than separated. A 3D row
    # never falls back to the 2D file: that file documents the 2D calculator, so an empty cell is
    # the honest reading of "no 3D matrix exists yet".
    mat = f"{family}3d.md" if dim == "3D" else f"{family}.md"
    mat = f"matrix/{mat}" if os.path.exists(os.path.join(HERE, "matrix", mat)) else ""
    return report, regen, mat


def row_assertions(row, s):
    """-> (scope, the functions the ROW's own test_name runs, unresolved names).

    `test_name` is the only registry column that identifies an assertion rather than a feature, so
    it is the only thing that can narrow the read from the feature to the row. Where it is empty
    the read stays feature-wide, and `verdict_scope` says so on the row rather than letting a
    feature-wide agreement read as if the row itself had been checked.
    """
    names = [t.strip() for t in row["test_name"].split(";") if t.strip()]
    if not names:
        return "feature", [], []
    fns, unresolved = [], []
    for name in names:
        got = s.cases.get(name)
        fns.extend(got) if got else unresolved.append(name)
    return "row", fns, unresolved


def verdict_of(row, cov, vetted_features):
    """The one fact neither side holds: does the claim match the tree?

    Judged against the ROW's own claim, not the feature's. A feature normally carries both a vetted
    row and a drift-guard row, and the guard is not at fault for the oracle test the other row
    records -- they are two assertions about the same feature. Only the FEATURE-level question
    "an oracle asserts this and no row anywhere says vetted" is a real gap.

    The feature-wide questions come first because they are the ones every row can be asked. The
    row-scoped ones follow, and only a row naming its assertion can be asked them at all -- what
    `verdict_scope` records, so the two are never confused for each other.
    """
    key = (row["dim"], row["family"])
    if key not in cov:
        return "unscanned", "none"
    s = cov[key]
    c = s.cov
    f, status, oracle = row["feature"], row["status"].strip(), row["oracle"].strip()
    scope, fns, unresolved = row_assertions(row, s)

    if status == "vetted":
        if not c.asserted.get(f):
            return "claim-without-assertion", scope
        if oracle and oracle not in c.oracles.get(f, set()):
            return "oracle-mismatch", scope
    else:
        # a regression or invariant row claims a drift guard, so that is what it is checked against
        if c.oracles.get(f) and (row["dim"], f) not in vetted_features:
            return "assertion-without-claim", scope
        if not (c.regression.get(f) or c.other.get(f) or c.asserted.get(f)):
            return "no-assertion-at-all", scope

    if scope == "feature":
        return "agree", scope

    if unresolved:
        return "test-name-unresolved", scope
    # the row's own assertion, at the row's own kind: a vetted row answers to an oracle-suffixed
    # function, and a feature-wide `asserted` set does not say that THIS case is one of them
    covering = c.asserted.get(f, set()) if status == "vetted" else (
        c.regression.get(f, set()) | c.other.get(f, set()) | c.asserted.get(f, set()))
    if not set(fns) & covering:
        return "row-test-lacks-feature", scope

    # the configuration, where the family declares how to read one off a function name
    readers = s.fam.recipe_reader
    if not readers:
        return "agree", scope
    recipe = row["config_recipe"].strip()
    if recipe not in readers:
        return "recipe-unreadable", "row+config"
    if not any(readers[recipe].match(fn) for fn in fns):
        return "recipe-mismatch", "row+config"
    return "agree", "row+config"


def build(cov, rows):
    vetted_features = {(r["dim"], r["feature"]) for r in rows if r["status"].strip() == "vetted"}
    out = []
    for r in rows:
        key = (r["dim"], r["family"])
        s = cov.get(key)
        c = s.cov if s else None
        f = r["feature"]
        oracle = r["oracle"].strip()
        # the claimed oracle first, then the tree's, so a row claiming none still reaches its report
        report, regen, mat = pointers(r["family"], r["dim"],
                                      [oracle] + (sorted(c.oracles.get(f, ())) if c else []))
        asserted = sorted(c.asserted.get(f, ())) if c else []
        regress = sorted(c.regression.get(f, ())) if c else []
        other = sorted(c.other.get(f, ())) if c else []
        row_tests = sorted(set(row_assertions(r, s)[1])) if s else []
        verdict, scope = verdict_of(r, cov, vetted_features)
        out.append({
            "dim": r["dim"], "family": r["family"], "feature": f,
            "claim_status": r["status"].strip(), "claim_oracle": oracle,
            "claim_recipe": r["config_recipe"].strip(),
            "claim_tolerance": r["tolerance"].strip(),
            "claim_agreement": r["agreement"].strip(),
            "claim_benchmark": r["benchmark"].strip(),
            "claim_test_name": r["test_name"].strip(),
            "claim_flag": r["flag"].strip(),
            "claim_candidate_oracle": r["candidate_oracle"].strip(),
            "claim_source": r["source"].strip(),
            "scan_oracles": ";".join(sorted(c.oracles.get(f, ()))) if c else "",
            "scan_oracle_tests": ";".join(asserted),
            "scan_regression_tests": ";".join(regress),
            "scan_other_tests": ";".join(other),
            "scan_n_oracle": str(len(asserted)),
            "scan_n_regression": str(len(regress)),
            "scan_row_tests": ";".join(row_tests),
            "verdict": verdict, "verdict_scope": scope,
            "vetting_report": report, "golden_regen": regen, "matrix": mat,
            "claim_notes": r["notes"],
        })
    return out


def render_csv(recs):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(recs)
    return buf.getvalue()


def oracle_matrix(recs):
    """(dim, family) x oracle, counting the union of claimed and scanned oracles per feature.

    The union is the honest count: a feature is often checked against more than one tool, and the
    registry records only the one that PROMOTED it. Counting the promoting oracle alone understates
    every family that corroborates.

    Keyed by dim as well as family: nine family names exist in both dims against different oracles,
    so merging them puts the 2D vetting and the 3D one in one cell.
    """
    per = collections.defaultdict(lambda: collections.defaultdict(set))
    tools = set()
    for r in recs:
        fam, feat = (r["dim"], r["family"]), r["feature"]
        got = {t for t in r["scan_oracles"].split(";") if t}
        if r["claim_oracle"]:
            got.add(r["claim_oracle"])
        if got:
            tools |= got
            for t in got:
                per[fam][t].add(feat)
        else:
            per[fam]["-"].add(feat)
    return per, sorted(tools)


def render_md(recs):
    stats = check_coverage.coverage_stats(registry())
    lines = [
        "# Nyxus feature report", "",
        "_Generated by `report_features.py`. Do not edit by hand._", "",
        "One row per (dim, feature, oracle, config_recipe) in [`features.csv`](features.csv), joining",
        "what [`oracle_coverage.csv`](oracle_coverage.csv) CLAIMS to what the test tree ASSERTS. The",
        "`verdict` column is the join: `agree` means the two say the same thing.", "",
        "## Coverage", "",
        f"Features vetted by >=1 oracle: {stats['vetted']}/{stats['total']} "
        f"({100.0 * stats['vetted'] / stats['total']:.0f}%)",
        f"regression: {stats['regression']}  invariant: {stats['invariant']}  "
        f"untested: {stats['untested']}", "",
        "| family | total | vetted | regression | invariant | untested |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(stats["by_family"]):
        v = stats["by_family"][name]
        lines.append(f"| {name} | {v['total']} | {v['vetted']} | {v['regression']} | "
                     f"{v['invariant']} | {v['untested']} |")


    # verdicts
    counts = collections.Counter(r["verdict"] for r in recs)
    meaning = {
        "agree": "the claim and the tree say the same thing",
        "claim-without-assertion": "status=vetted, but no oracle test in the tree asserts it",
        "oracle-mismatch": "both sides name an oracle, but not the same one",
        "assertion-without-claim": "an oracle test asserts it while the row claims no oracle",
        "no-assertion-at-all": "nothing in the tree covers the feature, of any kind",
        "test-name-unresolved": "test_name names a case that no TEST() in test_all.cc registers",
        "row-test-lacks-feature": "the case test_name names carries no assertion of this feature "
                                  "at this row's kind",
        "recipe-mismatch": "the case test_name names does not assert at this row's config_recipe",
        "recipe-unreadable": "the family declares a recipe reader, but not for this recipe, so the "
                             "configuration cannot be checked",
        "unscanned": "no scanner covers this family x dim yet",
    }
    lines += ["", "## Verdicts", "",
              f"{counts.get('agree', 0)} of {len(recs)} assertion rows agree with the tree.", "",
              "| verdict | rows | meaning |", "|---|---:|---|"]
    for v, n in counts.most_common():
        lines.append(f"| `{v}` | {n} | {meaning.get(v, '')} |")

    # how far each verdict reaches. An `agree` is only as strong as the evidence behind it, and the
    # three scopes are three different claims -- listing them keeps the weakest from being read as
    # the strongest, which is the whole reason the column exists.
    scopes = collections.Counter(r["verdict_scope"] for r in recs)
    scope_meaning = {
        "row+config": "the case named in `test_name` asserts this feature at this `config_recipe`",
        "row": "the case named in `test_name` asserts this feature, at this row's kind; its "
               "configuration is unchecked, the family declaring no recipe reader",
        "feature": "`test_name` is empty, so the tree was read for the feature and not for this "
                   "row's own assertion",
        "none": "no scanner for this family x dim, so nothing was read at all",
    }
    lines += ["", "### How far each verdict reaches", "",
              "`verdict_scope` is the evidence behind the verdict, not a second verdict. Only",
              "`row+config` checks the configuration cell that the report's own key is built on.",
              "", "| scope | rows | what was compared |", "|---|---:|---|"]
    for v, n in scopes.most_common():
        lines.append(f"| `{v}` | {n} | {scope_meaning.get(v, '')} |")

    offenders = [r for r in recs if r["verdict"] not in ("agree", "unscanned")]
    if offenders:
        lines += ["", "### Rows that do not agree", "",
                  "`unscanned` rows are not listed here; they are a property of the family, not of",
                  "the row, and are counted in the section below.", "",
                  "| dim | feature | verdict | claim | tree | allowed because |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(offenders, key=lambda r: (r["dim"], r["family"], r["feature"])):
            why = ALLOWLIST.get((r["dim"], r["feature"], r["verdict"]), "")
            lines.append(
                f"| {r['dim']} | `{r['feature']}` | `{r['verdict']}` | "
                f"{r['claim_status']}/{r['claim_oracle'] or '-'} | "
                f"{r['scan_oracles'] or '-'} | {why or '**not allowlisted**'} |")

    # the families nothing re-derives
    un = collections.Counter((r["dim"], r["family"]) for r in recs if r["verdict"] == "unscanned")
    if un:
        feats = collections.defaultdict(set)
        for r in recs:
            if r["verdict"] == "unscanned":
                feats[(r["dim"], r["family"])].add(r["feature"])
        lines += ["", "## Families with no scanner", "",
                  f"**{sum(un.values())} of {len(recs)} rows ({100.0 * sum(un.values()) / len(recs):.0f}%) "
                  f"cannot be checked against the tree at all.** These families have an",
                  "`audit/<family>_<dim>_coverage.csv`, but nothing regenerates it from the test",
                  "sources, so the artifact records what someone believed rather than what the tree",
                  f"asserts. Writing the {len(KNOWN_UNSCANNED)} missing scanners is what closes "
                  "this.", "",
                  "| dim | family | rows | features | why |", "|---|---|---:|---:|---|"]
        for (d, fam), n in sorted(un.items()):
            why = KNOWN_UNSCANNED.get((d, fam), "**not recorded**")
            lines.append(f"| {d} | {fam} | {n} | {len(feats[(d, fam)])} | {why} |")

    # family x oracle
    per, tools = oracle_matrix(recs)
    lines += ["", "## Every oracle a feature was matched against", "",
              "The union of the promoting oracle and the oracles the tree asserts against, counted",
              "per feature. The registry names only the oracle that PROMOTED a feature, so counting",
              "that alone understates every family that corroborates.", "",
              "| dim | family | " + " | ".join(tools) + " | none |",
              "|---" * (len(tools) + 3) + "|"]
    totals = collections.Counter()
    for d, fam in sorted(per):
        cells = []
        for t in tools:
            n = len(per[(d, fam)].get(t, ()))
            totals[t] += n
            cells.append(str(n) if n else ".")
        none = len(per[(d, fam)].get("-", ()))
        totals["-"] += none
        lines.append(f"| {d} | {fam} | " + " | ".join(cells) + f" | {none or '.'} |")
    lines.append("| | **all** | " + " | ".join(f"**{totals[t]}**" for t in tools)
                 + f" | **{totals['-']}** |")

    # by dimensionality. "no oracle assertion" is counted only over SCANNED features -- in an
    # unscanned family an empty scan_oracles means nobody looked, not that nothing asserts, and
    # folding the two together would report a hole in the tests where the hole is in the tooling.
    lines += ["", "## By dimensionality", "",
              "| dim | features | vetted | scanned | no oracle assertion | unscanned |",
              "|---|---:|---:|---:|---:|---:|"]
    for d in sorted({r["dim"] for r in recs}):
        rows_d = [r for r in recs if r["dim"] == d]
        feats = {r["feature"] for r in rows_d}
        vetted = {r["feature"] for r in rows_d if r["claim_status"] == "vetted"}
        unscanned = {r["feature"] for r in rows_d if r["verdict"] == "unscanned"}
        scanned = feats - unscanned
        with_oracle = {r["feature"] for r in rows_d if r["scan_oracles"]}
        lines.append(f"| {d} | {len(feats)} | {len(vetted)} | {len(scanned)} | "
                     f"{len(scanned - with_oracle)} | {len(unscanned)} |")

    lines += ["", "## Reading a row", "",
              "`claim_*` comes from `oracle_coverage.csv` and is human-authored, reviewed in a PR.",
              "`scan_*` is what the family's scanner reads out of the test tree, and is a fact about",
              "the tree at generation time. `scan_row_tests` is the narrower of the two: the",
              "functions this row's OWN `test_name` runs, where it names one. `verdict` is the join",
              "and `verdict_scope` says which of the two it was made against. The three pointer",
              "columns name the narrative report, the regeneration recipe and the config matrix for",
              "the row, where each exists.", ""]
    return "\n".join(lines) + "\n"


def stale_or_missing(path, text):
    if not os.path.exists(path):
        return f"{os.path.basename(path)} is missing; run with --write"
    with open(path, newline="", encoding="utf-8") as fh:
        if fh.read() != text:
            return f"{os.path.basename(path)} is stale; rerun with --write"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate features.csv and features.md")
    ap.add_argument("--check", action="store_true",
                    help="fail if either artifact is stale or a verdict is not allowlisted")
    a = ap.parse_args(argv)

    recs = build(scan_all(), registry())
    text_csv, text_md = render_csv(recs), render_md(recs)

    if a.check:
        problems = [p for p in (stale_or_missing(OUT_CSV, text_csv),
                                stale_or_missing(OUT_MD, text_md)) if p]
        # The key the report declares has to actually be one. Two rows sharing it means two
        # assertions have been merged into one line, and every count below them is then reading a
        # feature x oracle pair where it thinks it is reading a configuration.
        seen = collections.Counter(
            (r["dim"], r["feature"], r["claim_oracle"], r["claim_recipe"]) for r in recs)
        for k, n in sorted(seen.items()):
            if n > 1:
                problems.append(
                    f"{k[0]} {k[1]}: {n} rows share the key (oracle {k[2] or 'none'}, recipe "
                    f"{k[3] or 'none'}); one assertion is one row")
        for key in sorted({(r["dim"], r["family"]) for r in recs if r["verdict"] == "unscanned"}):
            if key not in KNOWN_UNSCANNED:
                problems.append(f"{key[0]} {key[1]}: no scanner regenerates its coverage artifact, "
                                f"and the gap is not recorded in KNOWN_UNSCANNED")
        for r in recs:
            if r["verdict"] in ("agree", "unscanned"):
                continue
            if (r["dim"], r["feature"], r["verdict"]) not in ALLOWLIST:
                problems.append(f"{r['dim']} {r['feature']}: verdict={r['verdict']} "
                                f"(claim {r['claim_status']}/{r['claim_oracle'] or 'no oracle'}, "
                                f"tree {r['scan_oracles'] or 'no oracle test'}) is not allowlisted")
        for p in problems:
            print("ERROR:", p)
        print(f"checked {len(recs)} rows: "
              f"{'clean' if not problems else str(len(problems)) + ' problem(s)'}")
        return 1 if problems else 0

    if a.write:
        # newline LF: text mode would emit CRLF on Windows, and LF is the repo standard
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
            fh.write(text_csv)
        with open(OUT_MD, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(text_md)
        print(f"wrote {OUT_CSV} and {OUT_MD} ({len(recs)} rows)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
