"""Validate tests/vetting/oracle_coverage.csv and regenerate coverage_report.md. Stdlib only."""
import csv, os, sys, argparse
from pathlib import Path

COLUMNS = ["dim","feature","family","status","oracle","agreement","config_recipe",
           "tolerance","current_test","target_test","candidate_oracle","flag","source","notes"]
ALLOWED_STATUS = {"vetted","regression","untested"}
ALLOWED_ORACLES = {"pyradiomics","radiomicsj","mirp","matlab","cellprofiler","mitk",
                   "feature2djava","wndcharm","imea","imagej","fraclac","ibsi","analytic","skimage",
                   "pydicom","opencv"}
# SPEC §2 kind / oracle filename suffixes
_ORACLE_SUFFIXES = ALLOWED_ORACLES
_KIND_SUFFIXES = {"regression", "mechanics", "invariant"}


def load_registry(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _split_tests(s):
    if not s:
        return []
    return [x.strip() for x in s.replace("|", ";").split(";") if x.strip()]


def _classify_basename(name):
    """Return ('oracle', token) | ('kind', kind) | ('other', '')."""
    stem = Path(name).stem
    if not stem.startswith("test_"):
        return "other", ""
    body = stem[5:]
    parts = body.split("_")
    for n in range(1, 4):
        suf = "_".join(parts[-n:])
        if suf in _ORACLE_SUFFIXES:
            return "oracle", suf
        if suf in _KIND_SUFFIXES:
            return "kind", suf
    if "invariant" in body:
        return "kind", "invariant"
    return "other", ""


def _tests_index(tests_dir):
    """Map basename -> absolute path for every test_* file under tests_dir."""
    root = Path(tests_dir)
    idx = {}
    if not root.is_dir():
        return idx
    for p in root.rglob("test_*"):
        if p.is_file() and p.suffix in {".h", ".cc", ".cpp", ".py"}:
            idx[p.name] = p
    return idx


def _is_oracle_shell(text):
    """True if file is an empty oracle-named shell that only pulls in regression."""
    if "re-homes the oracle label" in text:
        return True
    import re
    includes = re.findall(r'#include\s+"([^"]+)"', text)
    reg_inc = [i for i in includes if "regression" in i]
    if not reg_inc:
        return False
    code = [
        ln.strip() for ln in text.splitlines()
        if ln.strip()
        and not ln.strip().startswith("//")
        and not ln.strip().startswith("/*")
        and not ln.strip().startswith("*")
        and not ln.strip().startswith("#pragma")
    ]
    non_inc = [c for c in code if not c.startswith("#include")]
    return not non_inc


def _file_evidences_feature(feat, dim, fname, text):
    """Dim-correct content check: oracle file must mention this feature (SPEC §1–§2)."""
    import re
    if _is_oracle_shell(text):
        return False
    is_3d_file = fname.startswith("test_3d_") or "3d_" in fname
    has_exact = bool(re.search(rf'"{re.escape(feat)}"', text))
    has_f2 = bool(re.search(rf"Feature2D::{re.escape(feat)}\b", text))
    has_fimq = bool(re.search(rf"FeatureIMQ::{re.escape(feat)}\b", text))
    if dim == "3D":
        bare = feat[1:] if feat.startswith("3") else feat
        has_f3 = bool(re.search(rf"Feature3D::{re.escape(bare)}\b", text)) or bool(
            re.search(rf"Feature3D::{re.escape(feat)}\b", text)
        )
        has_ave = False
        if bare.endswith("_AVE"):
            base = bare[:-4]
            has_ave = bool(re.search(rf"Feature3D::{re.escape(bare)}\b", text))
            has_ave = has_ave or bool(re.search(rf'"{re.escape("3" + base)}"', text))
            has_ave = has_ave or bool(re.search(rf"Feature3D::{re.escape(base)}\b", text))
        if not is_3d_file:
            return has_exact  # 2D file must use exact 3-prefixed name
        return has_exact or has_f3 or has_ave
    if is_3d_file and not has_exact and not has_f2:
        return False
    if has_exact or has_f2 or has_fimq:
        return True
    if feat.startswith("IH_"):
        short = feat[3:]
        if re.search(rf'"{re.escape(short)}"', text) and re.search(
            rf"IH_{re.escape(short)}", text
        ):
            return True
        if re.search(rf"F::{re.escape(feat)}\b", text):
            return True
    return False


def validate_rows(rows, tests_dir=None):
    """Structural + (optional) filesystem checks.

    When tests_dir is provided:
    - every current_test basename must exist under tests/
    - status=vetted requires an oracle-named file whose *body* evidences the feature
      (filename-only links and empty shells that #include regression are rejected)
    """
    errs = []
    if rows:
        missing = [c for c in COLUMNS if c not in rows[0]]
        if missing:
            errs.append(f"missing columns: {missing}")
            return errs
    idx = _tests_index(tests_dir) if tests_dir else None
    text_cache = {}
    for r in rows:
        f = (r.get("feature") or "")
        st = (r.get("status") or "").strip()
        ora = (r.get("oracle") or "").strip()
        dim = (r.get("dim") or "").strip()
        if st not in ALLOWED_STATUS:
            errs.append(f"{f}: bad status {st!r}")
        if ora and ora not in ALLOWED_ORACLES:
            errs.append(f"{f}: oracle token {ora!r} not in SPEC 4 allowed set")
        if st == "vetted" and not ora:
            errs.append(f"{f}: status=vetted but no oracle")
        if st != "vetted" and ora:
            errs.append(f"{f}: status={st} but has oracle {ora!r}")
        if idx is not None:
            curr = _split_tests(r.get("current_test") or "")
            for c in curr:
                if c not in idx:
                    errs.append(f"{f}: current_test {c!r} not found under {tests_dir}")
            if st == "vetted":
                evidenced = False
                for c in curr:
                    kind, tok = _classify_basename(c)
                    if kind != "oracle" or tok != ora:
                        continue
                    p = idx.get(c)
                    if p is None:
                        continue
                    if c not in text_cache:
                        text_cache[c] = p.read_text(encoding="utf-8", errors="replace")
                    text = text_cache[c]
                    if _is_oracle_shell(text):
                        errs.append(
                            f"{f}: oracle file {c!r} is an empty shell "
                            f"(#include regression only) — not SPEC-vetted evidence"
                        )
                        continue
                    if _file_evidences_feature(f, dim, c, text):
                        evidenced = True
                if not evidenced:
                    errs.append(
                        f"{f}: status=vetted but no dim-correct oracle body covers this "
                        f"feature under oracle={ora!r} (filename-only links do not count)"
                    )
    return errs


def coverage_stats(rows):
    fam = {}
    tot = dict(total=0, vetted=0, regression=0, untested=0)
    for r in rows:
        st = r["status"].strip()
        tot["total"] += 1
        tot[st] = tot.get(st, 0) + 1
        f = fam.setdefault(r["family"], dict(total=0, vetted=0, regression=0, untested=0))
        f["total"] += 1
        f[st] = f.get(st, 0) + 1
    return dict(total=tot["total"], vetted=tot["vetted"], regression=tot["regression"],
                untested=tot["untested"], by_family=fam)


def render_report(rows):
    s = coverage_stats(rows)
    pct = (100.0 * s["vetted"] / s["total"]) if s["total"] else 0.0
    lines = ["# Nyxus Oracle-Vetting Coverage", "",
             "_Generated by check_coverage.py from oracle_coverage.csv. Do not edit by hand._", "",
             f"Features vetted by >=1 oracle: {s['vetted']}/{s['total']} ({pct:.0f}%)",
             f"regression: {s['regression']}  untested: {s['untested']}", "",
             "| family | total | vetted | regression | untested |",
             "|---|---|---|---|---|"]
    for name in sorted(s["by_family"]):
        f = s["by_family"][name]
        lines.append(f"| {name} | {f['total']} | {f['vetted']} | {f['regression']} | {f['untested']} |")
    return "\n".join(lines) + "\n"


def drift_warnings(rows, tests_dir):
    """Non-fatal: missing target_test paths, multi-oracle token mismatches."""
    warns = []
    idx = _tests_index(tests_dir)
    for r in rows:
        feat = r.get("feature", "")
        for tgt in _split_tests(r.get("target_test") or ""):
            if tgt and tgt not in idx:
                warns.append(f"{feat}: target_test {tgt} not found under {tests_dir}")
        st = (r.get("status") or "").strip()
        ora = (r.get("oracle") or "").strip()
        if st == "vetted" and ora:
            curr = _split_tests(r.get("current_test") or "")
            otoks = sorted({_classify_basename(c)[1]
                            for c in curr if _classify_basename(c)[0] == "oracle"})
            if otoks and ora not in otoks:
                warns.append(
                    f"{feat}: registry oracle={ora!r} not among current_test oracle tokens {otoks}"
                )
    return warns


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="tests/vetting/oracle_coverage.csv")
    ap.add_argument("--report", default="tests/vetting/coverage_report.md")
    ap.add_argument("--tests-dir", default="tests")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    rows = load_registry(a.registry)
    # Resolve tests dir relative to registry when needed
    tests_dir = a.tests_dir
    if not os.path.isdir(tests_dir):
        alt = os.path.join(os.path.dirname(a.registry), "..")
        if os.path.isdir(alt):
            tests_dir = os.path.normpath(alt)
    errs = validate_rows(rows, tests_dir=tests_dir if a.check or a.write else None)
    if a.check:
        for e in errs:
            print("ERROR:", e)
        for w in drift_warnings(rows, tests_dir):
            print("WARN:", w)
        return 1 if errs else 0
    if a.write:
        # write still requires structural validity; filesystem checks optional for report regen
        struct = validate_rows(rows, tests_dir=None)
        if struct:
            for e in struct:
                print("ERROR:", e)
            return 1
        with open(a.report, "w") as fh:
            fh.write(render_report(rows))
        print(f"wrote {a.report}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
