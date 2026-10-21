"""Scan the 3D GLDZM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_gldzm3d_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.

A pytest case has no gtest case to name in `test_name`. A row of a recipe in PYTEST_RECIPES leaves it
empty and is identified by its recipe instead: RECIPE_READER has to resolve it to exactly one
function, defined in a file `current_test` names, asserting the row's feature against the row's
oracle. Every other row still needs a `test_name` (pytest_identity_problems below).
"""
import os
import re
import sys

import scanlib

# The family's whole in-tree footprint. test_3d_gldzm_common.h is the shared fixture and asserts no
# feature, so it is not scanned. test_3d_coverage_common.h reads both pin tables, but only for their
# key sets -- it asserts a count and a has-a-pin property, never a feature value -- so it covers
# nothing per-feature either, matching every other 3D scanner here. The pytest file is VOLMIN's.
SOURCES = ["test_3d_gldzm_mirp.h", "test_3d_gldzm_regression.h"]

ORACLE_SUFFIX = {"mirp": "mirp"}

# The volmin table is a Python dict in a pytest file. One declaration reads one table dialect and the
# family's own tables are C++ initialisers, so the pytest file has a declaration of its own and its
# scan joins the family's in collect() below.
VOLMIN = scanlib.Family(
    dim="3D", family="gldzm",
    sources=[os.path.join("python", "test_3d_gldzm_mirp.py")],
    oracle_suffix=ORACLE_SUFFIX,
    table_owner={"GLDZM_3D_VOLMIN_MIRP": "test_3d_gldzm_volmin_mirp"},
    table_dialect="python",
)

# Only the two features no tool reaches are named. The other 16 are vetted against MIRP on
# bench_compat_gldzm_3d and their measurements live in audit/gldzm_3d_mirp_vetting_report.md;
# repeating them here would put a second copy of that table in a generated file.
NOTE = {
    "3GLDZM_GLM": "no counterpart in any tool (MIRP's GLDZM emits no dzm_gl_mean column) and none "
                  "in IBSI, so it is a drift guard at every recipe",
    "3GLDZM_ZDM": "no counterpart in any tool (MIRP's GLDZM emits no dzm_zd_mean column) and none "
                  "in IBSI, so it is a drift guard at every recipe",
}

# recipe -> the function that asserts AT that recipe. The family has five config cells and two of
# them are asserted against the same oracle on the same fixture, so feature, kind and oracle are
# identical between them and only the function name says which cell a row records.
RECIPE_READER = {
    "gldzm3d.mirp_compat_phantom": re.compile(
        r"^test_3d_gldzm_(?!volmin_)[a-z0-9]+_(mirp|compat_regression)$"),
    "gldzm3d.mirp_compat_phantom_radiomics": re.compile(
        r"^test_3d_gldzm_[a-z0-9]+_radiomics_(mirp|regression)$"),
    "gldzm3d.regression_ut_phantom": re.compile(r"^test_3d_gldzm_[a-z0-9]+_regression$"),
    "gldzm3d.mirp_fbn20_negdepth": re.compile(r"^test_3d_gldzm_negative_greydepth_mirp$"),
    "gldzm3d.mirp_volmin_fbn8": re.compile(r"^test_3d_gldzm_volmin_mirp$"),
}

# the recipes whose assertion is a pytest case, identified by RECIPE_READER rather than by test_name
PYTEST_RECIPES = {"gldzm3d.mirp_volmin_fbn8"}


def collect(fam, feat_re):
    """-> the five coverage maps of the family's sources and of VOLMIN's, joined."""
    maps = scanlib.collect(fam, feat_re)
    for into, more in zip(maps, scanlib.collect(VOLMIN, feat_re)):
        for key, val in more.items():
            if isinstance(val, set):
                into.setdefault(key, set()).update(val)
            else:
                into[key] = val
    return maps


def pytest_identity_problems(fam, cov):
    """scanlib's missing_test_name, with a pytest row identified by its recipe (module docstring)."""
    out = []
    for r in cov.rows:
        if r["test_name"].strip():
            continue
        f, recipe = r["feature"], r["config_recipe"].strip()
        if recipe not in PYTEST_RECIPES:
            out.append(f"{f}: no test_name, so current_test names an assertion nothing identifies")
            continue
        claimed = {t for t in r["current_test"].split(";") if t}
        fns = sorted(fn for fn in cov.asserted.get(f, set())
                     if RECIPE_READER[recipe].match(fn) and cov.where[fn] in claimed
                     and scanlib.oracle_token(fam, fn) == r["oracle"].strip())
        if len(fns) != 1:
            out.append(f"{f}: config_recipe {recipe} resolves to {fns or 'no function'} asserting "
                       f"it against {r['oracle'] or 'no oracle'} in {sorted(claimed)}; a pytest row "
                       f"needs one")
    return out


FAMILY = scanlib.Family(
    dim="3D", family="gldzm",
    sources=SOURCES,
    # MIRP is the only mainstream oracle for this family -- PyRadiomics implements no GLDZM at all.
    # The vetted cell is gldzm3d.mirp_compat_phantom, where neither tool discretises, so what the
    # comparison measures is the GLDZM rather than the binning the two disagree about on a CT-like
    # fixture (audit/gldzm_3d_mirp_vetting_report.md).
    oracle_suffix=ORACLE_SUFFIX,
    table_owner={"gldzm_3d_mirp_negdepth_ref_vals": "test_3d_gldzm_negative_greydepth_mirp"},
    notes=NOTE,
    enum_dim_prefix=True,
    other_note="asserted",
    loop_tables=True,
    # missing_test_name is pytest_identity_problems, which reads a pytest row by its recipe
    checks=scanlib.CORE_CHECKS | (scanlib.IDENTITY_CHECKS - {"missing_test_name"}),
    extra_problems=pytest_identity_problems,
    recipe_reader=RECIPE_READER,
    collect_override=collect,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
