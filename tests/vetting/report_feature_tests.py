#!/usr/bin/env python3
"""Report which test actually asserts each feature, and regenerate feature_test_report.md.

`oracle_coverage.csv` records the verdict a feature *carries*; this reports the assertions the
tree *contains*. They are not the same thing and are not meant to be: a row may be vetted from an
offline harness run (source=tracker) with `target_test` naming the file the assertion has not been
migrated into yet. Reading the registry alone cannot tell those apart, so every column here except
dim/family/feature is scanned from the tests.

Attribution has to follow three shapes or it badly under-counts:

  literal   the feature name appears in the test body       test_2d_gldm_sde_ibsi -> "GLDM_SDE"
  table     the test passes a golden table to a helper      test_2d_moments_shape_skimage passes
            that loops it, naming no feature itself         moments_2d_skimage_shape_ref_vals
  sweep     the 3D coverage suites enumerate features at    GLCM_UNVETTED_LOCAL_REGRESSION/...
            run time, so no source line names them

plus helper -> caller propagation, because SPEC 6.2 keeps assertions in `assert_*` helpers while
only the `test_*`/TEST() caller is the test: TEST(TEST_NYXUS, TEST_2D_GABOR_SKIMAGE) calls
assert_2d_gabor_skimage() and names no feature at all.

File and function kinds come from check_test_names.py, so oracle / regression / invariant /
mechanics here is the same taxonomy SPEC 6.1/6.2 enforces.

Usage (from the repo root):
    python tests/vetting/report_feature_tests.py --write        # regenerate the report
    python tests/vetting/report_feature_tests.py --write --csv  # ... plus the full-notes CSV
    python tests/vetting/report_feature_tests.py --check        # exit 1 if the report is stale
"""
import argparse
import collections
import csv
import importlib.util
import io
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("check_test_names", HERE / "check_test_names.py")
ctn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctn)

# Asserts about the vetting framework itself. Its planted fixtures quote feature names that no
# feature test computes, so counting them would report coverage that does not exist.
NOT_FEATURE_TESTS = {"test_vetting_mechanics.py"}

# imq names its own dimension (registry dim=IMQ), so these speak for IMQ rows only; every other
# file with no dim token may speak for any dim.
IMQ_FILES = {"test_imq_opencv.h", "test_imq_cellprofiler.h", "test_imq_regression.h"}

# The two parameterized suites of the 3D coverage sweep (test_3d_coverage_common.h). Their cases
# are built from the featureset at run time, so a feature they assert is named in no test body.
SWEEP_EMBEDDED = "Test3DFeature_WITH_3P_EMBEDDED_GT"
SWEEP_REGRESSION = "Test3DFeature_UNVETTED_LOCAL_REGRESSION"

CPP_FN = re.compile(r"^[ \t]*(?:static[ \t]+)?(?:inline[ \t]+)?(?:void|double|int|bool)[ \t]+"
                    r"((?:test|assert)_[A-Za-z0-9_]*)[ \t]*\(", re.M)
PY_FN = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z0-9_]+)[ \t]*\(", re.M)
# a table declaration: the brace often sits on the line after the name, so one newline is allowed
# but no more - an unbounded \s* here backtracks quadratically over the long blank runs that
# blanking a file's function bodies leaves behind (test_all.cc alone cost minutes).
# The optional "()" also catches an accessor wrapping a function-local static.
TABLE = re.compile(r"([A-Za-z_]\w*)[ \t]*(?:\([ \t]*\))?[ \t]*(?:=[ \t]*)?(?:\r?\n[ \t]*)?\{")
IDENT = re.compile(r"[A-Za-z_]\w*")
TOKEN = re.compile(r"[A-Za-z0-9_]+")            # feature names may start with a digit: 3GLCM_ACOR
ANGLE = re.compile(r"_(?:0|45|90|135)$")        # per-angle dataframe column: GLCM_CONTRAST_0
INSTANTIATE = re.compile(r"INSTANTIATE_TEST_SUITE_P\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)")
CPP_SKIP = re.compile(r'"(?:\\.|[^"\\])*"' r"|'(?:\\.|[^'\\])*'"
                      r"|(//[^\n]*)|(/\*.*?\*/)", re.S)
PY_SKIP = re.compile(r'"""(?:\\.|(?!""").)*"""' r"|'''(?:\\.|(?!''').)*'''"
                     r'|"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|(#[^\n]*)", re.S)

REPORT_COLUMNS = ["Dim", "Family", "FeatureName", "List_of_Oracles", "Test_Names",
                  "Regression", "Reg_Test_Name", "Notes"]


# ------------------------------------------------------------------- parsing ---
def blank_comments(txt, py=False):
    """Blank comment bodies, preserving offsets so spans stay valid. A commented-out golden row
    is not coverage, and offsets have to survive because the spans are cut from this text."""
    def sub(m):
        body = m.group(1) or (None if py else m.group(2))
        return re.sub(r"[^\n]", " ", body) if body else m.group(0)
    return (PY_SKIP if py else CPP_SKIP).sub(sub, txt)


def brace_map(txt):
    """{open offset: end offset (exclusive)} for every balanced {...}. One pass, so the per-table
    lookups stay linear in the file rather than rescanning to EOF per candidate."""
    ends, stack, i, n, instr = {}, [], 0, len(txt), None
    while i < n:
        ch = txt[i]
        if instr:
            if ch == "\\":
                i += 2
                continue
            if ch == instr:
                instr = None
        elif ch in "\"'":
            instr = ch
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            ends[stack.pop()] = i + 1
        i += 1
    for b in stack:
        ends[b] = n
    return ends


def cpp_spans(txt, ends):
    out = []
    for m in CPP_FN.finditer(txt):
        b = txt.find("{", m.end())
        if b >= 0:
            out.append((m.group(1), m.start(), ends.get(b, len(txt))))
    return out


def py_spans(txt):
    ms = list(PY_FN.finditer(txt))
    return [(m.group(1), m.start(), (ms[i + 1].start() if i + 1 < len(ms) else len(txt)))
            for i, m in enumerate(ms)]


def gtest_spans(txt):
    ms = list(ctn.GTEST_CASE.finditer(txt))
    return [(m.group(2), m.start(), (ms[i + 1].start() if i + 1 < len(ms) else len(txt)))
            for i, m in enumerate(ms)]


def file_dims(name, dim):
    if name in IMQ_FILES:
        return {"IMQ"}
    if dim == "2d":
        return {"2D"}
    if dim == "3d":
        return {"3D"}
    return {"2D", "3D", "IMQ"}


def file_kind(name):
    kind, _ = ctn.file_kind(name)
    if kind:
        return kind
    stem = name.rsplit(".", 1)[0]
    if stem.endswith("_common"):
        return "common"
    if stem.endswith("_coverage"):
        return "coverage"
    if name in ctn.FIXTURES:
        return "fixture"
    return "mixed" if name in ctn.GRANDFATHERED else "unknown"


def names_in(text, dims, feats, bare3):
    """Feature names referenced in `text`, restricted to the dims the file may speak for.
    One tokenize pass plus set membership - matching each of ~758 names by regex is quadratic."""
    found = set()
    for tok in set(TOKEN.findall(text)):
        stripped = ANGLE.sub("", tok)
        cands = [tok, stripped]
        if dims == {"3D"}:
            # a 3D file may name the feature through its enum (Feature3D::GLCM_ACOR), which drops
            # the registry's leading "3"; unambiguous because the file speaks only for 3D
            cands += [bare3.get(tok), bare3.get(stripped)]
        for n in cands:
            if n in feats and feats[n]["dim"] in dims:
                found.add(n)
    return found


# --------------------------------------------------------------- attribution ---
def load_registry(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def scan(tests_dir, rows):
    """-> (hits, helper_only, included, case_of_fn, called_fns).

    hits[feature] = {(file, test name, kind, how)} where `how` is literal/table/helper/sweep."""
    tests_dir = pathlib.Path(tests_dir)
    feats = {r["feature"]: r for r in rows}
    bare3 = {n[1:]: n for n in feats if n.startswith("3")}

    files = sorted([p for p in list(tests_dir.glob("*.h")) + list(tests_dir.glob("*.cc")) +
                    list((tests_dir / "python").glob("*.py"))
                    if p.name not in NOT_FEATURE_TESTS])

    def read(p):
        return p.read_text(encoding="utf-8", errors="surrogateescape")

    # which headers reach the binary at all: the #include closure of the gtest translation unit
    included, frontier = set(), ["test_all.cc"]
    while frontier:
        p = tests_dir / frontier.pop()
        if not p.exists():
            continue
        for inc in re.findall(r'#include\s+"([^"]+)"', read(p)):
            base = inc.split("/")[-1]
            if base.startswith("test_") and base not in included:
                included.add(base)
                frontier.append(base)

    all_cc = blank_comments(read(tests_dir / "test_all.cc"))
    case_of_fn, called_fns = {}, set()
    for case, s, e in gtest_spans(all_cc):
        for fn in re.findall(r"\b(test_[A-Za-z0-9_]*)\s*\(", all_cc[s:e]):
            called_fns.add(fn)
            case_of_fn.setdefault(fn, case)

    spans_all, tables_by_file, suites_by_file = [], {}, {}
    for p in files:
        kind = file_kind(p.name)
        dims = file_dims(p.name, ctn.file_dim(p.name)[0])
        txt = blank_comments(read(p), py=(p.suffix == ".py"))
        ends = brace_map(txt)
        spans = (gtest_spans(txt) if p.name == "test_all.cc" else
                 py_spans(txt) if p.suffix == ".py" else cpp_spans(txt, ends))

        # file scope = golden tables, module-level dicts, sweep instantiations
        scope = list(txt)
        for _, s, e in spans:
            for k in range(s, min(e, len(scope))):
                scope[k] = " "
        scope = "".join(scope)

        tables = {}
        for m in TABLE.finditer(scope):
            b = m.end() - 1
            blk = scope[b:ends.get(b, len(scope))]
            if len(blk) > 300000:                 # a whole-file span, not a golden table
                continue
            ns = names_in(blk, dims, feats, bare3)
            if ns:
                tables.setdefault(m.group(1), set()).update(ns)
        tables_by_file[p.name] = tables
        for m in INSTANTIATE.finditer(scope):
            suites_by_file.setdefault(p.name, {})[m.group(2)] = m.group(1)

        for fn, s, e in spans:
            body = txt[s:e]
            fkind = fn.rsplit("_", 1)[-1].lower()   # gtest case name is UPPER(function)
            fkind = fkind if fkind in ctn.KINDS else kind
            found = {(n, "literal") for n in names_in(body, dims, feats, bare3)}
            idents = set(IDENT.findall(body))
            for ident in idents & tables.keys():
                found |= {(n, "table") for n in tables[ident]}
            spans_all.append([p.name, fn, fkind, found, idents])

    # push each helper's features out to whoever calls it; helper names are unique tree-wide
    helper_feats = {fn: {n for n, _ in found}
                    for f, fn, k, found, ids in spans_all if fn.startswith("assert_")}
    for _ in range(3):                              # helpers call helpers; converges well inside 3
        grown = False
        for f, fn, k, found, ids in spans_all:
            add = set()
            for c in (ids & helper_feats.keys()) - {fn}:
                add |= {(n, "helper") for n in helper_feats[c]}
            if add - found:
                found |= add
                grown = True
                if fn.startswith("assert_"):
                    helper_feats[fn] |= {n for n, _ in add}
        if not grown:
            break

    hits, helper_only = collections.defaultdict(set), collections.defaultdict(set)
    for f, fn, k, found, ids in spans_all:
        for n, how in found:
            if fn.startswith("test_") or f == "test_all.cc":
                hits[n].add((f, fn, k, how))
            else:
                helper_only[n].add((f, fn))

    add_sweep_hits(rows, hits, tables_by_file, suites_by_file)
    return hits, helper_only, included, case_of_fn, called_fns


def add_sweep_hits(rows, hits, tables_by_file, suites_by_file):
    """The 3D coverage sweep asserts every public 3D feature through one of two parameterized
    suites, chosen at run time by whether an oracle table already holds the feature's name."""
    oracle_file = re.compile(r"test_3d_\w+_(%s)\.h" % "|".join(sorted(ctn.ORACLES)))
    embedded = set()
    for f, tabs in tables_by_file.items():
        if oracle_file.fullmatch(f):
            for ns in tabs.values():
                embedded |= ns
    for r in rows:
        if r["dim"] != "3D":
            continue
        cov = "test_3d_%s_coverage.h" % r["family"]
        suites = suites_by_file.get(cov)
        if not suites:
            continue
        name = r["feature"]
        if name in embedded:
            pre = suites.get(SWEEP_EMBEDDED)
            if pre:
                hits[name].add((cov, "%s/%s.PublicFeatureIsComputableAndHasEmbeddedOracle"
                                % (pre, SWEEP_EMBEDDED), "coverage", "sweep"))
        elif any(name in ns for ns in tables_by_file.get(cov, {}).values()):
            pre = suites.get(SWEEP_REGRESSION)
            if pre:
                hits[name].add((cov, "%s/%s.PublicFeatureIsComputableButHasNoEmbeddedOracleYet"
                                % (pre, SWEEP_REGRESSION), "regression", "sweep"))


def pick_entries(entries):
    """Keep the most precise attribution a file offers: a test that names the feature beats one
    that only passes the golden table, which beats one that only calls a shared helper."""
    for tier in ("literal", "table", "helper", "sweep"):
        best = [e for e in entries if e[2] == tier]
        if best:
            return sorted(set(best))
    return sorted(set(entries))


def test_label(file, fn, case_of_fn, included, called_fns):
    """The name the test runs under, plus why it does not run if it does not."""
    name = case_of_fn.get(fn, fn) if not file.endswith(".py") and "/" not in fn else fn
    tag = ""
    if file.endswith(".h") and file not in included:
        tag = " [ORPHANED: header not #included by test_all.cc]"
    elif (file.endswith((".h", ".cc")) and file != "test_all.cc"
          and fn.startswith("test_") and fn not in called_fns):
        tag = " [NOT RUN: no TEST() case calls it]"
    return "%s::%s%s" % (file, name, tag)


def build_rows(rows, hits, helper_only, included, case_of_fn, called_fns):
    out = []
    for r in rows:
        name = r["feature"]
        per_file = collections.defaultdict(list)
        for f, fn, k, how in hits.get(name, ()):
            per_file[f].append((fn, k, how))

        oracle_tests, reg_tests, other_tests, oracles = [], [], [], set()
        for f in sorted(per_file):
            for fn, k, how in pick_entries(per_file[f]):
                tag = test_label(f, fn, case_of_fn, included, called_fns)
                if k in ctn.ORACLES:
                    oracles.add(k)
                    oracle_tests.append(tag)
                elif k == "regression":
                    reg_tests.append(tag)
                else:
                    other_tests.append("%s (%s)" % (tag, k))

        claim = (r["status"] or "").strip()
        claimed_oracle = (r["oracle"] or "").strip()
        notes = []
        if claim == "vetted" and not oracle_tests:
            where = "source=%s" % (r.get("source") or "?")
            if (r.get("target_test") or "").strip():
                where += ", target_test=%s" % r["target_test"].strip()
            notes.append("registry claims vetted vs %s but no in-tree oracle test asserts it (%s)"
                         % (claimed_oracle or "?", where))
        elif claimed_oracle and oracle_tests and claimed_oracle not in oracles:
            notes.append("registry oracle=%s; in-tree oracle test(s): %s"
                         % (claimed_oracle, ",".join(sorted(oracles))))
        if oracle_tests and claim != "vetted":
            notes.append("registry status=%s but oracle-kind test(s) found: %s"
                         % (claim, ",".join(sorted(oracles))))
        if not (oracle_tests or reg_tests or other_tests):
            ho = sorted(f for f, _ in helper_only.get(name, ()))
            notes.append("golden value present in %s but no test function names it" % "; ".join(ho)
                         if ho else "NO TEST in the tree names this feature")
        if reg_tests and all(("ORPHANED" in t or "NOT RUN" in t) for t in reg_tests):
            notes.append("its only regression coverage is in a test that never executes")

        listed = sorted(oracles | ({claimed_oracle} if claimed_oracle else set()))
        out.append({
            "Dim": r["dim"], "Family": r["family"], "FeatureName": name,
            "List_of_Oracles": ", ".join(listed) if listed else "-",
            "Test_Names": " | ".join(oracle_tests + other_tests) or "-",
            "Regression": "Yes" if reg_tests else "No",
            "Reg_Test_Name": " | ".join(reg_tests) or "-",
            "Notes": " | ".join(notes),
            "registry_status": claim,
            "registry_oracle": claimed_oracle,
            "registry_notes": (r["notes"] or "").strip(),
            "n_oracle_tests": len(oracle_tests),
            "n_reg_tests": len(reg_tests),
            "n_other_tests": len(other_tests),
        })
    return out


# ------------------------------------------------------------------ rendering ---
def esc(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def render_report(out_rows):
    n = len(out_rows)
    def count(pred):
        return sum(1 for r in out_rows if pred(r))
    orc = count(lambda r: r["n_oracle_tests"])
    reg = count(lambda r: r["n_reg_tests"])
    gap = count(lambda r: r["registry_status"] == "vetted" and not r["n_oracle_tests"])
    dead = count(lambda r: "never executes" in r["Notes"])
    none = count(lambda r: not (r["n_oracle_tests"] or r["n_reg_tests"] or r["n_other_tests"]))

    L = ["# Nyxus feature x test report", "",
         "_Generated by report_feature_tests.py from oracle_coverage.csv and the test tree."
         " Do not edit by hand._", "",
         "One row per registry feature. `Dim`/`Family`/`FeatureName` come from"
         " `oracle_coverage.csv`; every other column is scanned from the tests, so this says what"
         " the tree asserts rather than what the registry claims. Where the two disagree the note"
         " says so - most often a row vetted from an offline harness run whose assertion has not"
         " been migrated into the tree yet (its `target_test` names where it belongs).", "",
         "| | count | share |", "|---|---:|---:|"]
    for label, v in [("features", n),
                     ("have an in-tree oracle test", orc),
                     ("have a regression test", reg),
                     ("have neither", none),
                     ("registry says `vetted`, no in-tree oracle test asserts it", gap),
                     ("regression coverage only in a test that never executes", dead)]:
        L += ["| %s | %d | %d%% |" % (label, v, round(100.0 * v / n) if n else 0)]
    L += [""]

    L += ["## By dim x family", "",
          "| Dim | Family | features | in-tree oracle | regression | oracles (registry + in-tree) |",
          "|---|---|---:|---:|---:|---|"]
    fam = collections.defaultdict(list)
    for r in out_rows:
        fam[(r["Dim"], r["Family"])].append(r)
    for key in sorted(fam):
        rs = fam[key]
        tools = sorted({t.strip() for r in rs for t in r["List_of_Oracles"].split(",")
                        if t.strip() and t.strip() != "-"})
        L += ["| %s | %s | %d | %d | %d | %s |"
              % (key[0], key[1], len(rs), sum(1 for r in rs if r["n_oracle_tests"]),
                 sum(1 for r in rs if r["n_reg_tests"]), ", ".join(tools) or "-")]
    L += [""]

    dead_files = collections.defaultdict(set)
    for r in out_rows:
        for cell in (r["Test_Names"], r["Reg_Test_Name"]):
            for part in cell.split(" | "):
                if "ORPHANED" in part or "NOT RUN" in part:
                    key = part.split("::")[0] + (" [NOT RUN]" if "NOT RUN" in part else "")
                    dead_files[key].add(r["FeatureName"])
    if dead_files:
        L += ["## Tests credited to features that never execute", "",
              "A header no `#include` chain reaches from `tests/test_all.cc` is not compiled into"
              " `runAllTests`; a `test_*` function no `TEST()` case calls is compiled but never"
              " run. Either way the assertion has never executed."
              " (`not_covered.md` sections B.1/B.2 track these.)", "",
              "| file | features crediting it |", "|---|---:|"]
        for f in sorted(dead_files):
            L += ["| `%s` | %d |" % (f, len(dead_files[f]))]
        L += [""]

    L += ["## Full table", "",
          "`Notes` carries only what the scan found. The registry's own note for a feature stays"
          " in `oracle_coverage.csv`; `feature_test_report.csv` (`--csv`) joins the two.", "",
          "| Dim | Family | FeatureName | List_of_Oracles | Test_Names | Regression |"
          " Reg_Test_Name | Notes |", "|---|---|---|---|---|---|---|---|"]
    order = {"2D": 0, "3D": 1, "IMQ": 2}
    for r in sorted(out_rows, key=lambda r: (order.get(r["Dim"], 9), r["Family"], r["FeatureName"])):
        L += ["| %s | %s | `%s` | %s | %s | %s | %s | %s |"
              % (r["Dim"], r["Family"], r["FeatureName"], esc(r["List_of_Oracles"]),
                 esc(r["Test_Names"]), r["Regression"], esc(r["Reg_Test_Name"]), esc(r["Notes"]))]
    return "\n".join(L) + "\n"


def render_csv(out_rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REPORT_COLUMNS + [
        "registry_status", "registry_oracle", "registry_notes",
        "n_oracle_tests", "n_reg_tests", "n_other_tests"], lineterminator="\n")
    w.writeheader()
    for r in out_rows:
        w.writerow(r)
    return buf.getvalue()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:   # LF, like the rest of the tree
        fh.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=os.path.join("tests", "vetting", "oracle_coverage.csv"))
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--report", default=os.path.join("tests", "vetting", "feature_test_report.md"))
    ap.add_argument("--csv", action="store_true",
                    help="also write feature_test_report.csv beside the report")
    ap.add_argument("--check", action="store_true", help="exit 1 if the report on disk is stale")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    rows = load_registry(a.registry)
    out_rows = build_rows(rows, *scan(a.tests, rows))
    text = render_report(out_rows)

    twin = os.path.splitext(a.report)[0] + ".csv"
    if a.check:
        # the csv twin is checked only when it is present: it is optional output, but once a copy
        # is on disk a stale one is read as current exactly like a stale report
        wanted = [(a.report, text)] + ([(twin, render_csv(out_rows))] if os.path.exists(twin) else [])
        stale = []
        for path, expected in wanted:
            try:
                with open(path, encoding="utf-8") as fh:
                    if fh.read() != expected:
                        stale.append(path)
            except OSError:
                stale.append(path)
        for path in stale:
            print("ERROR: %s is missing or stale - run with --write --csv" % path)
        return 1 if stale else 0
    if a.write:
        write_text(a.report, text)
        print("wrote %s" % a.report)
        if a.csv:
            twin = os.path.splitext(a.report)[0] + ".csv"
            write_text(twin, render_csv(out_rows))
            print("wrote %s" % twin)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
