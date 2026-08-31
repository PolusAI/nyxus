import csv, importlib.util, os, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "vetting" / "check_coverage.py"
_spec = importlib.util.spec_from_file_location("check_coverage", _SCRIPT)
cc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cc)

_NAMES = _HERE.parent / "vetting" / "check_test_names.py"
_nspec = importlib.util.spec_from_file_location("check_test_names", _NAMES)
ctn = importlib.util.module_from_spec(_nspec); _nspec.loader.exec_module(ctn)
_REPO = _HERE.parents[1]

def _write(tmp_path, rows):
    p = tmp_path / "reg.csv"
    # the schema comes from check_coverage itself: a hardcoded copy here silently fails every
    # validate_rows() call the moment a column is added, which is what adding test_name/benchmark did
    cols = list(cc.COLUMNS)
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

def test_vetting_unquoted_comma_row_flagged_mechanics(tmp_path):
    """A row carrying an unquoted comma has more fields than the header: every field after the
    comma shifts left by one and the last one falls off the end entirely. 3ROBUST_MEAN sat that
    way with a "[P10,P90]" candidate_oracle, so its notes had become its source.

    Written field by field off cc.COLUMNS rather than as a row literal: a literal stops having one
    field too many the moment the schema grows, and then asserts nothing."""
    row = {c: "" for c in cc.COLUMNS}
    row.update(dim="3D", feature="3X", family="firstorder", status="vetted", oracle="matlab",
               agreement="agreed", current_test="t.h", target_test="t.h",
               candidate_oracle="octave (mean of [P10,P90])", source="tracker", notes="note")
    p = tmp_path / "reg.csv"
    with open(p, "w", newline="") as fh:
        fh.write(",".join(cc.COLUMNS) + "\n")
        fh.write(",".join(row[c] for c in cc.COLUMNS) + "\n")
    errs = cc.validate_rows(cc.load_registry(str(p)))
    assert any("past the last column" in e for e in errs), errs


def test_vetting_coverage_stats_and_report_mechanics(tmp_path):
    path = _write(tmp_path, [
        {"dim":"2D","feature":"A","family":"glcm","status":"vetted","oracle":"pyradiomics"},
        {"dim":"2D","feature":"A","family":"glcm","status":"vetted","oracle":"matlab"},
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
         "target_test":"test_2d_glcm_pyradiomics.h"},
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

def test_vetting_undefined_benchmark_flagged_mechanics(tmp_path):
    """A benchmark id is a pointer; an id benchmarks.md does not define points at nothing (SPEC 6.3).
    The registry gained the column so a row can say which fixture backs it - unchecked, the column
    would let a row name a fixture that never existed."""
    md = tmp_path / "benchmarks.md"
    md.write_text("# Benchmark registry\n\n## `bench_real_one`\n\nsomething\n", encoding="utf-8")
    rows = [{"feature": "A", "benchmark": "bench_real_one"},
            {"feature": "B", "benchmark": "bench_typo"}]
    errs = cc.validate_benchmarks(rows, str(md))
    assert len(errs) == 1 and "bench_typo" in errs[0], errs
    assert cc.benchmark_ids(str(md)) == {"bench_real_one"}

def test_vetting_undefined_config_recipe_flagged_mechanics(tmp_path):
    """The third pointer column. benchmark and test_name were validated while config_recipe was
    not, so a row could name a SPEC 5 recipe that had never been written and every checker stayed
    green - which is how two imq ids and one glrlm id survived. A blank cell claims nothing and is
    allowed; a cell that names a recipe has to resolve to a heading."""
    md = tmp_path / "config_recipes.md"
    md.write_text("# Config Recipes\n\n## glcm.ibsi_identity\n\nsomething\n", encoding="utf-8")
    rows = [{"feature": "A", "config_recipe": "glcm.ibsi_identity"},
            {"feature": "B", "config_recipe": "glcm.typo_that_was_never_written"},
            {"feature": "C", "config_recipe": ""}]
    errs = cc.validate_config_recipes(rows, str(md))
    assert len(errs) == 1 and "typo_that_was_never_written" in errs[0], errs
    assert cc.config_recipe_ids(str(md)) == {"glcm.ibsi_identity"}

    # The file-missing branch. It is a separate return path, and an untested branch is one a
    # refactor can turn into a silent pass -- with no config_recipes.md every id resolves to
    # nothing, so it has to report rather than wave the rows through.
    gone = tmp_path / "nowhere.md"
    assert cc.config_recipe_ids(str(gone)) is None
    assert len(cc.validate_config_recipes(rows, str(gone))) == 1
    assert cc.validate_config_recipes([{"feature": "C", "config_recipe": ""}], str(gone)) == []

def test_vetting_unresolvable_test_name_flagged_mechanics(tmp_path):
    """A test_name is a pointer too: SPEC 3 identifies an assertion by the gtest case it runs as,
    so a name no case answers to says a feature is covered by a test nobody runs."""
    cc_file = tmp_path / "test_all.cc"
    cc_file.write_text("TEST(TEST_NYXUS, TEST_3D_NGLDM_LDE_REGRESSION) {\n}\n", encoding="utf-8")
    rows = [{"feature": "A", "test_name": "TEST_NYXUS.TEST_3D_NGLDM_LDE_REGRESSION"},
            {"feature": "B", "test_name": "TEST_NYXUS.TEST_THAT_NEVER_EXISTED"},
            {"feature": "C", "test_name": ""}]
    errs = cc.validate_test_names(rows, str(cc_file))
    assert len(errs) == 1 and "TEST_THAT_NEVER_EXISTED" in errs[0], errs
    assert cc.gtest_case_names(str(cc_file)) == {"TEST_NYXUS.TEST_3D_NGLDM_LDE_REGRESSION"}

def test_vetting_check_fails_on_stale_report_mechanics(tmp_path):
    """coverage_report.md is generated from the registry and says so in its own header, so --check
    must reject one that no longer matches it. Unenforced, the published figure can overstate
    coverage for as long as nobody re-runs --write: the ten GLCM matlab rows demoted in #422 left
    the committed report claiming 118/118 glcm vetted against the registry's 108."""
    path = _write(tmp_path, [
        {"dim":"2D","feature":"A","family":"glcm","status":"vetted","oracle":"pyradiomics"},
    ])
    report = tmp_path / "coverage_report.md"
    assert cc.main(["--write", "--registry", path, "--report", str(report)]) == 0
    # freshly written -> clean
    assert cc.main(["--check", "--registry", path, "--report", str(report)]) == 0
    # registry moves on, report does not -> caught
    path2 = _write(tmp_path, [
        {"dim":"2D","feature":"A","family":"glcm","status":"regression","oracle":""},
    ])
    assert cc.main(["--check", "--registry", path2, "--report", str(report)]) == 1
    # an AD-HOC report (named on the command line) that is not there claims nothing, and the other
    # self-tests rely on that; the canonical one is covered by the next test
    assert cc.main(["--check", "--registry", path2,
                    "--report", str(tmp_path / "absent.md")]) == 0


def test_vetting_check_fails_on_missing_canonical_report_mechanics(tmp_path, monkeypatch):
    """Deleting coverage_report.md must not be the one edit that passes --check. The staleness rule
    compares rendered text to the file, so an absent file has nothing to disagree with; without this
    the enforcement SPEC 3.1 promises is one `rm` away from silent. Only the CANONICAL report -- the
    --report default -- is required to exist, because --check also runs against ad-hoc registries in
    tmp_path that have no report beside them."""
    monkeypatch.chdir(tmp_path)
    reg = tmp_path / "tests" / "vetting"
    reg.mkdir(parents=True)
    path = _write(reg, [
        {"dim":"2D","feature":"A","family":"glcm","status":"vetted","oracle":"pyradiomics"},
    ])
    canonical = reg / "coverage_report.md"
    assert cc.main(["--write", "--registry", path, "--report", str(canonical)]) == 0
    # the canonical path is cc.DEFAULT_REPORT relative to the cwd, so --check with no --report
    # resolves to the file just written -> clean
    assert cc.DEFAULT_REPORT == "tests/vetting/coverage_report.md"
    assert cc.main(["--check", "--registry", path]) == 0
    # delete it and the same command must now fail rather than pass by absence
    canonical.unlink()
    assert cc.main(["--check", "--registry", path]) == 1


def test_vetting_unknown_source_flagged_mechanics(tmp_path):
    """`source` is a closed SPEC 3 set -- in-tree / tracker / audit -- and nothing checked it, so an
    invented token read as meaningful: six rows said `generator`, naming where the numbers came from
    rather than where the verdict does. A blank cell claims nothing and stays allowed."""
    path = _write(tmp_path, [
        {"dim":"2D","feature":"A","family":"x","status":"regression","source":"in-tree"},
        {"dim":"2D","feature":"B","family":"x","status":"regression","source":"generator"},
        {"dim":"2D","feature":"C","family":"x","status":"regression","source":""},
    ])
    errs = [e for e in cc.validate_rows(cc.load_registry(path)) if "source" in e]
    assert len(errs) == 1 and "'generator'" in errs[0] and errs[0].startswith("B:"), errs


# ---- SPEC 6.1/6.2 test-naming conventions (tests/vetting/check_test_names.py) ----

def test_vetting_tree_names_conform_to_spec_mechanics():
    """Every test file, test function and gtest case must satisfy SPEC 6.1/6.2. A new test whose
    name does not end in a kind/oracle token fails here, at write time rather than at review time."""
    violations = ctn.check(_REPO)
    assert violations == [], chr(10).join(violations)


def test_vetting_name_checker_rejects_bad_names_mechanics(tmp_path):
    """The checker must actually fail on each defect class - a lint that cannot fail is not a lint.
    Plants one of each and requires all fourteen to be reported. Each planted file isolates a single
    defect, so an assertion failing here names the rule that stopped being enforced."""
    (tmp_path / "tests" / "python").mkdir(parents=True)
    # a TEST body with two callees: case = UPPER(function) has no single function to mirror
    # ... and a case whose only callee wears the helper prefix: nothing else in 6.2 inspects an
    # assert_*, so without this rule the case name is checked by suffix alone and the function not
    # at all.
    (tmp_path / "tests" / "test_all.cc").write_text(
        "TEST(TEST_NYXUS, TEST_2D_GLSZM_SZE_REGRESSION) {\n"
        "    test_2d_glszm_sze_regression();\n"
        "    test_2d_glszm_lze_regression();\n"
        "}\n"
        "TEST(TEST_NYXUS, TEST_2D_GABOR_SKIMAGE) {\n"
        "    assert_2d_gabor_skimage();\n"
        "}\n", encoding="utf-8")
    (tmp_path / "tests" / "test_2d_gabor_skimage.h").write_text(
        "void assert_2d_gabor_skimage() {}", encoding="utf-8")
    (tmp_path / "tests" / "test_2d_glszm_regression.h").write_text(
        "void test_2d_glszm_sze_regression() {}\nvoid test_2d_glszm_lze_regression() {}",
        encoding="utf-8")
    (tmp_path / "tests" / "test_2d_glcm.h").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_glcm_regression.h").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_2d_glcm_regression.h").write_text(
        "void test_2d_glcm_contrast() {}", encoding="utf-8")
    (tmp_path / "tests" / "test_2d_gldm_ibsi.h").write_text(
        "void test_2d_gldm_sde_regression() {}", encoding="utf-8")
    (tmp_path / "tests" / "test_2d_ngldm_ibsi.h").write_text(
        "void test_3d_ngldm_sde_ibsi() {}", encoding="utf-8")
    (tmp_path / "tests" / "test_2d_ngtdm_regression.h").write_text(
        "void test_helper(int x) {}", encoding="utf-8")
    # a golden table under the pre-6.3.1 naming, brace on the next line
    (tmp_path / "tests" / "test_2d_glrlm_regression.h").write_text(
        'static std::unordered_map<std::string, double> ibsi_reference_glrlm_golden_values\n'
        '{\n    {"GLRLM_SRE", 1.0},\n};\n'
        'void test_2d_glrlm_sre_regression() {}', encoding="utf-8")
    # the same defect written "name = {" on one line. Planted separately because the two spellings
    # reach the detector by different paths: the declaration regex once accepted "=" or "{" but not
    # both, so a table opened this way was never inspected and any name passed.
    (tmp_path / "tests" / "test_2d_ngtdm_ibsi.h").write_text(
        'static const std::unordered_map<std::string, double> oracle_3p_ngtdm_values = {\n'
        '    {"NGTDM_COARSENESS", 1.0},\n};\n'
        'void test_2d_ngtdm_coarseness_ibsi() {}', encoding="utf-8")
    # a table reached through an accessor wrapping a function-local static. Its name never appears
    # in a variable declaration, so the declaration regex cannot see it however it is spelled.
    (tmp_path / "tests" / "test_2d_glszm_matlab.h").write_text(
        'static const std::map<std::string, double>& matlab_glszm_shape_gt()\n'
        '{\n'
        '\tstatic const std::map<std::string, double> gt = {\n'
        '\t\t{ "GLSZM_SAE", 1.0 }\n'
        '\t};\n'
        '\treturn gt;\n}\n'
        'void test_2d_glszm_sae_matlab() {}', encoding="utf-8")

    # a reference table left in a shared header (SPEC 6.3.1: _common.h holds fixtures, not values).
    # The name conforms, so only the file it sits in makes it a defect.
    (tmp_path / "tests" / "test_2d_gldzm_common.h").write_text(
        'static ref_vals_map<double> gldzm_2d_ibsi_ref_vals\n'
        '{\n    {"GLDZM_SDE", 1.0},\n};\n', encoding="utf-8")

    # a table whose name and location both conform, declared as a raw container instead of a
    # 6.3.1 alias. Only its type makes it a defect, so this is what keeps the type rule honest.
    (tmp_path / "tests" / "test_2d_ngldm_regression.h").write_text(
        'static const std::unordered_map<std::string, double> ngldm_2d_regression_ref_vals = {\n'
        '    {"NGLDM_LDE", 1.0},\n};\n'
        'void test_2d_ngldm_lde_regression() {}', encoding="utf-8")

    # a table whose name, location and type all conform, declared without const. Only its
    # mutability makes it a defect -- which is what a mutable table costs: operator[] compiles, so a
    # missing key is default-inserted as 0 instead of failing the lookup.
    (tmp_path / "tests" / "test_2d_glszm_ibsi.h").write_text(
        'static ref_vals_map<double> glszm_2d_ibsi_ref_vals = {\n'
        '    {"GLSZM_SZE", 1.0},\n};\n'
        'void test_2d_glszm_sze_ibsi() {}', encoding="utf-8")

    # one family reaching into another family's fixture header. Both file names conform and the
    # header holds no values, so only the include makes it a defect (SPEC 6.3.1).
    (tmp_path / "tests" / "test_2d_glszm_common.h").write_text(
        "static void calculate_glszm_fixture() {}\n", encoding="utf-8")
    (tmp_path / "tests" / "test_2d_gldzm_regression.h").write_text(
        '#include "test_2d_glszm_common.h"\n'
        'void test_2d_gldzm_sde_regression() {}', encoding="utf-8")

    errs = ctn.check(tmp_path)
    assert any("test_2d_glcm.h" in e and "SPEC 6.1" in e for e in errs)       # file, no kind
    assert any("test_glcm_regression.h" in e and "dim token" in e for e in errs)  # file, no dim
    assert any("test_2d_glcm_contrast" in e and "kind/oracle" in e for e in errs)  # function, no kind
    assert any("test_2d_gldm_sde_regression" in e and "SPEC 2" in e for e in errs)  # wrong kind for file
    assert any("test_3d_ngldm_sde_ibsi" in e and "_2d_ dim token" in e for e in errs)  # wrong dim for file
    assert any("test_helper" in e and "assert_*" in e for e in errs)          # helper as test_
    assert any("TEST_2D_GLSZM_SZE_REGRESSION" in e and "calls 2 test_ functions" in e
               for e in errs)                                                # case mirrors no single fn
    assert any("TEST_2D_GABOR_SKIMAGE" in e and "assert_2d_gabor_skimage()" in e
               and "takes no arguments" in e for e in errs)                  # nullary helper IS the test
    assert any("ibsi_reference_glrlm_golden_values" in e and "SPEC 6.3.1" in e
               for e in errs)                                                # golden table off-convention
    assert any("oracle_3p_ngtdm_values" in e and "SPEC 6.3.1" in e
               for e in errs)                                                # ... declared as "name = {"
    assert any("matlab_glszm_shape_gt" in e and "SPEC 6.3.1" in e
               for e in errs)                                                # ... reached via an accessor
    assert any("gldzm_2d_ibsi_ref_vals" in e and "_common.h" in e
               for e in errs)                                                # conforming name, wrong file
    assert any("ngldm_2d_regression_ref_vals" in e and "raw container" in e
               for e in errs)                                                # conforming name, raw type
    assert any("test_2d_gldzm_regression.h" in e and "test_2d_glszm_common.h" in e
               and "SPEC 6.3.1" in e for e in errs)                          # includes another family
    assert any("glszm_2d_ibsi_ref_vals" in e and "without const" in e
               for e in errs)                                                # conforming name+type, mutable
    # and it must not fire on a const-qualified table -- a rule that flags everything is as useless
    # as one that flags nothing. ngldm_2d_regression_ref_vals above is const, so it earns the raw
    # container error and only that one.
    assert not any("ngldm_2d_regression_ref_vals" in e and "without const" in e for e in errs)
