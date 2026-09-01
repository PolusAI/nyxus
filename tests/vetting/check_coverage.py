"""Validate tests/vetting/oracle_coverage.csv and regenerate coverage_report.md. Stdlib only."""
import csv, os, re, sys, argparse

COLUMNS = ["dim","feature","family","status","oracle","agreement","config_recipe",
           "tolerance","current_test","target_test","candidate_oracle","flag","source","notes",
           # SPEC 3: the assertion a row records is identified by the exact gtest case it runs
           # as and the benchmark it runs on. A row naming neither says a feature is covered
           # without saying by what, which is what the 3D NGLDM review asked to close.
           "test_name","benchmark"]
# SPEC 3 lists `invariant` among the outcome values; the checker never implemented it, so a
# cell backed by a path-equality assertion had nowhere to sit and was being written `vetted`,
# which claims an oracle. It is not a synonym for either: an invariant establishes a required
# relation between two Nyxus code paths and no external tool is involved.
ALLOWED_STATUS = {"vetted","regression","invariant","untested"}
ALLOWED_ORACLES = {"pyradiomics","radiomicsj","mirp","matlab","cellprofiler","mitk",
                   "feature2djava","wndcharm","imea","imagej","fraclac","ibsi","analytic","skimage",
                   "pydicom","opencv"}
# SPEC 3: where the VERDICT comes from -- a test in this repo, an offline harness run not migrated
# in, or an audit. Not where the numbers were generated; that belongs in notes.
ALLOWED_SOURCES = {"in-tree","tracker","audit"}
DEFAULT_REPORT = "tests/vetting/coverage_report.md"

def load_registry(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def benchmark_ids(path):
    """-> the ids defined in benchmarks.md, i.e. every '## `bench...`' or '## `name`' heading."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        return set(re.findall(r"^##\s+`([^`]+)`", fh.read(), re.M))


def validate_benchmarks(rows, benchmarks_md):
    """A benchmark id that is not defined is a pointer to nothing (SPEC 6.3)."""
    defined = benchmark_ids(benchmarks_md)
    if defined is None:
        return [f"{benchmarks_md} is missing; SPEC 6.3 requires it once any row names a benchmark"] \
            if any((r.get("benchmark") or "").strip() for r in rows) else []
    errs = []
    for r in rows:
        b = (r.get("benchmark") or "").strip()
        if b and b not in defined:
            errs.append(f"{r.get('feature','')}: benchmark {b!r} is not defined in "
                        f"{os.path.basename(benchmarks_md)}")
    return errs


def config_recipe_ids(path):
    """-> the recipe ids config_recipes.md defines, i.e. every '## <id>' heading."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        return set(re.findall(r"^##\s+([A-Za-z0-9_.]+)\s*$", fh.read(), re.M))


def validate_config_recipes(rows, recipes_md):
    """A config_recipe that is not defined is a pointer to nothing (SPEC 5).

    The third pointer column, and the last one to be checked. benchmark and test_name were
    validated while this one was not, so two imq ids and a whole glrlm id named sections that had
    never been written, and nothing in the tree could see it. A blank cell is allowed - a row that
    names no recipe claims nothing - but a cell that names one has to resolve."""
    defined = config_recipe_ids(recipes_md)
    named = [r for r in rows if (r.get("config_recipe") or "").strip()]
    if defined is None:
        return [f"{recipes_md} is missing; SPEC 5 requires it once any row names a recipe"] \
            if named else []
    errs = []
    for r in named:
        c = (r.get("config_recipe") or "").strip()
        if c not in defined:
            errs.append(f"{r.get('feature','')}: config_recipe {c!r} is not defined in "
                        f"{os.path.basename(recipes_md)}")
    return errs


def gtest_case_names(test_all_cc):
    """-> the "SUITE.CASE" names the gtest translation unit defines, or None if it is unreadable."""
    if not os.path.exists(test_all_cc):
        return None
    with open(test_all_cc, encoding="utf-8", errors="replace") as fh:
        return {f"{s}.{c}" for s, c in
                re.findall(r"^\s*TEST(?:_[PF])?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", fh.read(), re.M)}


def validate_test_names(rows, test_all_cc):
    """A test_name that no gtest case answers to names a test nobody runs (SPEC 3).

    Same rule as validate_benchmarks: the column exists so a row can say which assertion covers it,
    and a name that resolves to nothing says it without saying anything. Several names separated by
    ";" are allowed, matching how current_test lists more than one file."""
    defined = gtest_case_names(test_all_cc)
    named = [r for r in rows if (r.get("test_name") or "").strip()]
    if defined is None:
        return [f"{test_all_cc} is missing; a test_name cannot be resolved without it"] if named else []
    errs = []
    for r in named:
        for t in (r.get("test_name") or "").split(";"):
            t = t.strip()
            if t and t not in defined:
                errs.append(f"{r.get('feature','')}: test_name {t!r} is not a gtest case in "
                            f"{os.path.basename(test_all_cc)} - name it SUITE.CASE as declared there")
    return errs


def validate_rows(rows):
    errs = []
    if rows:
        missing = [c for c in COLUMNS if c not in rows[0]]
        if missing:
            errs.append(f"missing columns: {missing}")
            return errs
    for r in rows:
        f = (r.get("feature") or ""); st = (r.get("status") or "").strip(); ora = (r.get("oracle") or "").strip()
        # A row with more fields than the header means an unquoted comma inside one of them: csv
        # collects the overflow under the None key, every field after the comma is shifted, and the
        # last one silently drops out of the table. 3ROBUST_MEAN sat like that with a
        # "[P10,P90]" candidate_oracle -- its notes had become its source and its notes were gone.
        if None in r:
            errs.append(f"{f}: {len(r[None])} field(s) past the last column - an unquoted comma "
                        f"shifts every field after it; quote the field")
        if any(v is None for v in r.values()):
            errs.append(f"{f}: fewer fields than columns - the row is truncated")
        if st not in ALLOWED_STATUS:
            errs.append(f"{f}: bad status {st!r}")
        if ora and ora not in ALLOWED_ORACLES:
            errs.append(f"{f}: oracle token {ora!r} not in SPEC 4 allowed set")
        if st == "vetted" and not ora:
            errs.append(f"{f}: status=vetted but no oracle")
        if st != "vetted" and ora:
            errs.append(f"{f}: status={st} but has oracle {ora!r}")
        # SPEC 3 defines three source values and nothing checked the column, so an invented one
        # read as meaningful: six rows said source=generator, which no reader resolves. The
        # generator belongs in notes; the column says where the VERDICT comes from.
        src = (r.get("source") or "").strip()
        if src and src not in ALLOWED_SOURCES:
            errs.append(f"{f}: source {src!r} not in SPEC 3 allowed set "
                        f"{sorted(ALLOWED_SOURCES)}")
    return errs

def coverage_stats(rows):
    # The registry is one row per assertion, so several oracle rows may cover one feature. Roll up
    # by (dimension, feature) for the headline metric instead of counting assertion rows as features.
    # `invariant` ranks with `regression`: both say "an assertion covers this" and neither says an
    # oracle agreed, so neither may raise the headline. It is a separate bucket rather than folded
    # into regression because the two make different claims -- a snapshot pins a value, an invariant
    # pins a relation between two code paths -- and a reader who cannot tell them apart will read
    # path-equality as evidence of correctness.
    rank = {"untested": 0, "regression": 1, "invariant": 1, "vetted": 2}
    features = {}
    for r in rows:
        key = (r["dim"], r["feature"])
        if key not in features or rank[r["status"].strip()] > rank[features[key]["status"].strip()]:
            features[key] = r

    fam = {}
    tot = dict(total=0, vetted=0, regression=0, invariant=0, untested=0)
    for r in features.values():
        st = r["status"].strip()
        tot["total"] += 1; tot[st] = tot.get(st, 0) + 1
        f = fam.setdefault(r["family"],
                           dict(total=0, vetted=0, regression=0, invariant=0, untested=0))
        f["total"] += 1; f[st] = f.get(st, 0) + 1
    return dict(total=tot["total"], vetted=tot["vetted"], regression=tot["regression"],
                invariant=tot["invariant"], untested=tot["untested"], by_family=fam)

def render_report(rows):
    s = coverage_stats(rows)
    pct = (100.0 * s["vetted"] / s["total"]) if s["total"] else 0.0
    lines = ["# Nyxus Oracle-Vetting Coverage", "",
             "_Generated by check_coverage.py from oracle_coverage.csv. Do not edit by hand._", "",
             f"Features vetted by >=1 oracle: {s['vetted']}/{s['total']} ({pct:.0f}%)",
             f"regression: {s['regression']}  invariant: {s['invariant']}  "
             f"untested: {s['untested']}", "",
             "| family | total | vetted | regression | invariant | untested |",
             "|---|---|---|---|---|---|"]
    for name in sorted(s["by_family"]):
        f = s["by_family"][name]
        lines.append(f"| {name} | {f['total']} | {f['vetted']} | {f['regression']} | "
                     f"{f['invariant']} | {f['untested']} |")
    return "\n".join(lines) + "\n"

def drift_warnings(rows, tests_dir):
    warns = []
    for r in rows:
        tgt = (r.get("target_test") or "").strip()
        if tgt and not os.path.exists(os.path.join(tests_dir, tgt)):
            warns.append(f"{r.get('feature', '')}: target_test {tgt} not found in {tests_dir}")
    return warns

def report_staleness(rows, report_path, canonical):
    """[] if coverage_report.md matches what the registry renders to, one error if it does not.

    `canonical` says whether this is the committed report (the --report default) or one named on
    the command line. A missing canonical report is an error in its own right: deleting the file
    would otherwise be the one edit that passes, which is the same overstatement as a stale one
    reached by a shorter route. A missing ad-hoc report claims nothing -- the self-tests validate
    registries in tmp_path that have no report beside them -- so it stays clean."""
    if not os.path.exists(report_path):
        if canonical:
            return [f"{report_path} is missing: it is generated from the registry and committed "
                    f"beside it. Regenerate it with --write."]
        return []
    with open(report_path, newline="") as fh:
        on_disk = fh.read()
    if on_disk.replace("\r\n", "\n") == render_report(rows).replace("\r\n", "\n"):
        return []
    return [f"{report_path} is stale: it does not match what {len(rows)} registry rows render to. "
            f"Regenerate it with --write (it is generated, not hand-edited)."]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="tests/vetting/oracle_coverage.csv")
    # default None rather than the path itself, so --check can tell the committed report from one
    # named on the command line; only the former must exist
    ap.add_argument("--report", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    report = a.report if a.report is not None else DEFAULT_REPORT
    rows = load_registry(a.registry)
    errs = validate_rows(rows)
    errs += validate_benchmarks(rows, os.path.join(os.path.dirname(a.registry) or ".",
                                                   "benchmarks.md"))
    errs += validate_config_recipes(rows, os.path.join(os.path.dirname(a.registry) or ".",
                                                       "config_recipes.md"))
    errs += validate_test_names(rows, os.path.join(
        os.path.dirname(os.path.dirname(a.registry)) or ".", "test_all.cc"))
    if a.check:
        # The report is generated from the registry and says so in its own header, but until now
        # nothing checked that it still matched, so "re-run --write after editing the registry"
        # was remembered rather than enforced. It has been forgotten before: the ten GLCM matlab
        # rows demoted in #422 left the committed report claiming 118/118 glcm vetted against the
        # registry's 108, and the published headline overstated coverage by ten features until
        # someone happened to regenerate. Comparing the rendered text to the file on disk closes
        # that for good; every registry edit after this one has to bring its report with it.
        errs += report_staleness(rows, report, canonical=a.report is None)
        for e in errs: print("ERROR:", e)
        return 1 if errs else 0
    if a.write:
        if errs:
            for e in errs: print("ERROR:", e)
            return 1
        # newline LF: text mode would emit CRLF on Windows, and LF is the repo standard
        with open(report, "w", newline="\n") as fh: fh.write(render_report(rows))
        print(f"wrote {report}")
        return 0
    ap.print_help(); return 0

if __name__ == "__main__":
    sys.exit(main())
