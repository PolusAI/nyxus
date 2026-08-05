import csv, importlib.util, os, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "vetting" / "check_coverage.py"
_spec = importlib.util.spec_from_file_location("check_coverage", _SCRIPT)
cc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cc)

_NAMES_SCRIPT = _HERE.parent / "vetting" / "check_test_names.py"
_nspec = importlib.util.spec_from_file_location("check_test_names", _NAMES_SCRIPT)
ctn = importlib.util.module_from_spec(_nspec); _nspec.loader.exec_module(ctn)
_REPO = _HERE.parents[1]

def _write(tmp_path, rows):
    p = tmp_path / "reg.csv"
    cols = ["dim","feature","family","status","oracle","agreement","config_recipe",
            "tolerance","current_test","target_test","candidate_oracle","flag","source","notes"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return str(p)

def test_vetting_valid_registry_has_no_errors_mechanics(tmp_path):
    path = _write(tmp_path, [
        {"dim":"2D","feature":"GLCM_CONTRAST","family":"glcm","status":"vetted","oracle":"pyradiomics"},
        {"dim":"2D","feature":"ROUNDNESS","family":"morphology","status":"regression","oracle":""},
    ])
    assert cc.validate_rows(cc.load_registry(path)) == []

def test_vetting_bad_status_and_token_flagged_mechanics(tmp_path):
    path = _write(tmp_path, [
        {"dim":"2D","feature":"A","family":"x","status":"maybe","oracle":""},          # bad status
        {"dim":"2D","feature":"B","family":"x","status":"vetted","oracle":"mahotas"},   # token not allowed
        {"dim":"2D","feature":"C","family":"x","status":"vetted","oracle":""},          # vetted w/o oracle
        {"dim":"2D","feature":"D","family":"x","status":"regression","oracle":"ibsi"},  # non-vetted w/ oracle
    ])
    errs = cc.validate_rows(cc.load_registry(path))
    assert len(errs) == 4
    assert any("maybe" in e for e in errs) and any("mahotas" in e for e in errs)

def test_vetting_coverage_stats_and_report_mechanics(tmp_path):
    path = _write(tmp_path, [
        {"dim":"2D","feature":"A","family":"glcm","status":"vetted","oracle":"pyradiomics"},
        {"dim":"2D","feature":"B","family":"glcm","status":"regression","oracle":""},
        {"dim":"2D","feature":"C","family":"moments","status":"untested","oracle":""},
    ])
    rows = cc.load_registry(path)
    s = cc.coverage_stats(rows)
    assert s["total"] == 3 and s["vetted"] == 1 and s["regression"] == 1 and s["untested"] == 1
    assert s["by_family"]["glcm"] == {"total":2,"vetted":1,"regression":1,"untested":0}
    rep = cc.render_report(rows)
    assert rep.startswith("# Nyxus Oracle-Vetting Coverage")
    assert "Features vetted by >=1 oracle: 1/3" in rep

def test_vetting_drift_and_main_write_mechanics(tmp_path):
    path = _write(tmp_path, [
        {"dim":"2D","feature":"A","family":"glcm","status":"vetted","oracle":"pyradiomics",
         "target_test":"test_glcm_pyradiomics.h"},
    ])
    # target file does not exist -> one drift warning
    assert len(cc.drift_warnings(cc.load_registry(path), str(tmp_path))) == 1
    # --write emits coverage_report.md next to the registry
    rc = cc.main(["--write", "--registry", path, "--report", str(tmp_path / "coverage_report.md")])
    assert rc == 0 and (tmp_path / "coverage_report.md").exists()
    assert "Features vetted" in (tmp_path / "coverage_report.md").read_text()

def test_vetting_main_check_fails_on_bad_row_mechanics(tmp_path):
    path = _write(tmp_path, [{"dim":"2D","feature":"A","family":"x","status":"bad","oracle":""}])
    assert cc.main(["--check", "--registry", path]) == 1

# ---- SPEC 6.1/6.2 test-naming conventions (tests/vetting/check_test_names.py) ----

def test_vetting_test_tree_names_conform_to_spec_mechanics():
    """The whole test tree must satisfy the SPEC 6.1 file-name and 6.2 function-name rules.
    A new test with a name that does not end in a kind/oracle token fails here."""
    violations = ctn.check(_REPO)
    assert violations == [], "\n".join(violations)

def test_vetting_name_checker_flags_a_bad_name_mechanics(tmp_path):
    """The checker itself must reject a non-conforming name (guards against a no-op lint)."""
    (tmp_path / "tests" / "python").mkdir(parents=True)
    (tmp_path / "tests" / "test_all.cc").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_glcm_regression.h").write_text(
        "void test_glcm_contrast() {}\n", encoding="utf-8")     # missing the _regression suffix
    (tmp_path / "tests" / "test_glcm.h").write_text("", encoding="utf-8")  # missing the kind suffix
    errs = ctn.check(tmp_path)
    assert any("test_glcm.h" in e and "SPEC 6.1" in e for e in errs)
    assert any("test_glcm_contrast" in e and "SPEC 6.2" in e for e in errs)
