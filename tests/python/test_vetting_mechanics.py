import csv, importlib.util, os, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "vetting" / "check_coverage.py"
_spec = importlib.util.spec_from_file_location("check_coverage", _SCRIPT)
cc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cc)

_NAMES = _HERE.parent / "vetting" / "check_test_names.py"
_nspec = importlib.util.spec_from_file_location("check_test_names", _NAMES)
ctn = importlib.util.module_from_spec(_nspec); _nspec.loader.exec_module(ctn)
_REPO = _HERE.parents[1]

_REPORT = _HERE.parent / "vetting" / "report_feature_tests.py"
_rspec = importlib.util.spec_from_file_location("report_feature_tests", _REPORT)
rft = importlib.util.module_from_spec(_rspec); _rspec.loader.exec_module(rft)

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

# ---- SPEC 6.1/6.2 test-naming conventions (tests/vetting/check_test_names.py) ----

def test_vetting_tree_names_conform_to_spec_mechanics():
    """Every test file, test function and gtest case must satisfy SPEC 6.1/6.2. A new test whose
    name does not end in a kind/oracle token fails here, at write time rather than at review time."""
    violations = ctn.check(_REPO)
    assert violations == [], chr(10).join(violations)


def test_vetting_name_checker_rejects_bad_names_mechanics(tmp_path):
    """The checker must actually fail on each defect class - a lint that cannot fail is not a lint.
    Plants one of each and requires all twelve to be reported. Each planted file isolates a single
    defect, so an assertion failing here names the rule that stopped being enforced."""
    (tmp_path / "tests" / "python").mkdir(parents=True)
    # a TEST body with two callees: case = UPPER(function) has no single function to mirror
    (tmp_path / "tests" / "test_all.cc").write_text(
        "TEST(TEST_NYXUS, TEST_2D_GLSZM_SZE_REGRESSION) {\n"
        "    test_2d_glszm_sze_regression();\n"
        "    test_2d_glszm_lze_regression();\n"
        "}\n", encoding="utf-8")
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

    errs = ctn.check(tmp_path)
    assert any("test_2d_glcm.h" in e and "SPEC 6.1" in e for e in errs)       # file, no kind
    assert any("test_glcm_regression.h" in e and "dim token" in e for e in errs)  # file, no dim
    assert any("test_2d_glcm_contrast" in e and "kind/oracle" in e for e in errs)  # function, no kind
    assert any("test_2d_gldm_sde_regression" in e and "SPEC 2" in e for e in errs)  # wrong kind for file
    assert any("test_3d_ngldm_sde_ibsi" in e and "_2d_ dim token" in e for e in errs)  # wrong dim for file
    assert any("test_helper" in e and "assert_*" in e for e in errs)          # helper as test_
    assert any("TEST_2D_GLSZM_SZE_REGRESSION" in e and "calls 2 test_ functions" in e
               for e in errs)                                                # case mirrors no single fn
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


# ---- the feature x test report (tests/vetting/report_feature_tests.py) ----

def _plant_tree(tmp_path, registry_rows, files):
    (tmp_path / "tests" / "python").mkdir(parents=True)
    for name, text in files.items():
        (tmp_path / "tests" / name).write_text(text, encoding="utf-8")
    (tmp_path / "tests" / "vetting").mkdir(exist_ok=True)
    path = _write(tmp_path, registry_rows)
    rows = rft.load_registry(path)
    return rows, rft.build_rows(rows, *rft.scan(tmp_path / "tests", rows))


def test_vetting_report_finds_each_attribution_shape_mechanics(tmp_path):
    """The three shapes the tree actually uses. Matching only literal feature names in a test body
    misses two of them: the moments oracle tests name a table and no feature, and the 3D coverage
    sweeps build their cases from the featureset at run time. Both must land as coverage."""
    rows, out = _plant_tree(tmp_path, [
        {"dim": "2D", "feature": "GLDM_SDE", "family": "gldm", "status": "vetted", "oracle": "ibsi"},
        {"dim": "2D", "feature": "HU_M1", "family": "moments", "status": "vetted", "oracle": "skimage"},
        {"dim": "3D", "feature": "3GLCM_ENERGY_AVE", "family": "glcm", "status": "regression"},
    ], {
        "test_all.cc":
            "TEST(TEST_NYXUS, TEST_2D_GLDM_SDE_IBSI) { test_2d_gldm_sde_ibsi(); }\n"
            "TEST(TEST_NYXUS, TEST_2D_MOMENTS_SHAPE_SKIMAGE) { test_2d_moments_shape_skimage(); }\n"
            '#include "test_2d_gldm_ibsi.h"\n#include "test_2d_moments_skimage.h"\n'
            '#include "test_3d_glcm_coverage.h"\n',
        # literal: the test names the feature
        "test_2d_gldm_ibsi.h": 'void test_2d_gldm_sde_ibsi() { assert_x("GLDM_SDE"); }\n',
        # table: the test names a golden table and no feature
        "test_2d_moments_skimage.h":
            'static ref_vals_list<G> moments_2d_skimage_shape_ref_vals\n'
            '{\n\t{Feature2D::HU_M1, "HU_M1", 1.0},\n};\n'
            'void test_2d_moments_shape_skimage() { assert_g(moments_2d_skimage_shape_ref_vals); }\n',
        # sweep: nothing names the feature; the suite enumerates it at run time
        "test_3d_glcm_coverage.h":
            "INSTANTIATE_TEST_SUITE_P(\n\tGLCM_UNVETTED_LOCAL_REGRESSION,\n"
            "\tTest3DFeature_UNVETTED_LOCAL_REGRESSION,\n\ttesting::ValuesIn(x), y);\n"
            'static ref_vals_map<std::vector<double>> glcm_3d_regression_coverage_ref_vals\n'
            '{\n\t{ "3GLCM_ENERGY_AVE", { 0.217 } },\n};\n',
    })
    by = {r["FeatureName"]: r for r in out}
    assert by["GLDM_SDE"]["Test_Names"] == "test_2d_gldm_ibsi.h::TEST_2D_GLDM_SDE_IBSI"
    assert by["HU_M1"]["Test_Names"] == "test_2d_moments_skimage.h::TEST_2D_MOMENTS_SHAPE_SKIMAGE"
    assert by["3GLCM_ENERGY_AVE"]["Regression"] == "Yes"
    assert rft.SWEEP_REGRESSION in by["3GLCM_ENERGY_AVE"]["Reg_Test_Name"]


def test_vetting_report_propagates_helper_assertions_to_callers_mechanics(tmp_path):
    """SPEC 6.2 keeps assertions in assert_* helpers, so the test that runs one may name no
    feature at all - TEST_2D_GABOR_SKIMAGE just calls assert_2d_gabor_skimage(). Without
    helper -> caller propagation an entire oracle file reads as uncovered."""
    rows, out = _plant_tree(tmp_path, [
        {"dim": "2D", "feature": "GABOR", "family": "gabor", "status": "vetted", "oracle": "skimage"},
    ], {
        "test_all.cc": "TEST(TEST_NYXUS, TEST_2D_GABOR_SKIMAGE) { assert_2d_gabor_skimage(false); }\n"
                       '#include "test_2d_gabor_skimage.h"\n',
        "test_2d_gabor_skimage.h":
            "void assert_2d_gabor_skimage() { check(fvals[(int)Feature2D::GABOR]); }\n",
    })
    r = out[0]
    assert r["Test_Names"] == "test_all.cc::TEST_2D_GABOR_SKIMAGE"
    assert r["List_of_Oracles"] == "skimage" and r["Notes"] == ""


def test_vetting_report_flags_claims_and_tests_that_never_run_mechanics(tmp_path):
    """The two disagreements the report exists to surface: a vetted verdict no in-tree assertion
    backs, and coverage credited to a header test_all.cc never includes, so it never compiled."""
    rows, out = _plant_tree(tmp_path, [
        {"dim": "2D", "feature": "AREA_UM2", "family": "morphology", "status": "vetted",
         "oracle": "matlab", "source": "tracker", "target_test": "test_2d_morphology_matlab.h"},
        {"dim": "3D", "feature": "3GLCM_DIS", "family": "glcm", "status": "regression"},
    ], {
        "test_all.cc": "TEST(TEST_NYXUS, TEST_2D_MORPHOLOGY_BASIC_REGRESSION)"
                       " { test_2d_morphology_basic_regression(); }\n"
                       '#include "test_2d_morphology_regression.h"\n',
        "test_2d_morphology_regression.h":
            'void test_2d_morphology_basic_regression() { assert_m("AREA_UM2"); }\n',
        # never included from test_all.cc -> compiled into nothing
        "test_3d_glcm_regression.h":
            'void test_3d_glcm_dis_regression() { assert_g("3GLCM_DIS"); }\n',
    })
    by = {r["FeatureName"]: r for r in out}
    # regression only, so the regression test stands in, starred - the column names a test whenever
    # one exists, and the star is what keeps that from reading as an oracle assertion
    assert (by["AREA_UM2"]["Test_Names"]
            == "*test_2d_morphology_regression.h::TEST_2D_MORPHOLOGY_BASIC_REGRESSION")
    assert by["AREA_UM2"]["n_oracle_tests"] == 0
    assert "no in-tree oracle test asserts it" in by["AREA_UM2"]["Notes"]
    assert "target_test=test_2d_morphology_matlab.h" in by["AREA_UM2"]["Notes"]
    assert "ORPHANED" in by["3GLCM_DIS"]["Reg_Test_Name"]
    assert "never executes" in by["3GLCM_DIS"]["Notes"]


def test_vetting_report_numbers_rows_and_counts_vetted_ones_mechanics(tmp_path):
    """`#` and `Vetted` are what make the report countable by eye. `#` has to run 1..n over the
    order the report prints, and `Vetted` has to advance only on a vetted row - a counter that
    also ticked on a regression row would read as a vetted total that the registry never claims."""
    rows, out = _plant_tree(tmp_path, [
        {"dim": "3D", "feature": "3GLCM_DIS", "family": "glcm", "status": "regression"},
        {"dim": "2D", "feature": "ROUNDNESS", "family": "morphology", "status": "regression"},
        {"dim": "2D", "feature": "GLDM_SDE", "family": "gldm", "status": "vetted", "oracle": "ibsi"},
        {"dim": "2D", "feature": "AREA_UM2", "family": "morphology", "status": "vetted",
         "oracle": "matlab"},
    ], {"test_all.cc": "\n"})
    assert [(r["#"], r["Vetted"], r["FeatureName"]) for r in out] == [
        (1, 1, "GLDM_SDE"),         # 2D sorts before 3D, gldm before morphology
        (2, 2, "AREA_UM2"),
        (3, "", "ROUNDNESS"),       # regression: no verdict to count
        (4, "", "3GLCM_DIS"),
    ]


def test_vetting_report_on_disk_is_current_mechanics():
    """A generated file that drifts from its source is worse than none - it is read as current.
    Regenerate with: python tests/vetting/report_feature_tests.py --write"""
    rc = rft.main(["--check",
                   "--registry", str(_HERE.parent / "vetting" / "oracle_coverage.csv"),
                   "--tests", str(_HERE.parent),
                   "--report", str(_HERE.parent / "vetting" / "feature_test_report.md")])
    assert rc == 0, "tests/vetting/feature_test_report.md is stale - run --write"
