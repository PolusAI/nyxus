"""Scan the 3D NGLDM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_ngldm3d_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.

A pytest case has no gtest case to name in `test_name`. A row of a recipe in PYTEST_RECIPES leaves it
empty and is identified by its recipe instead: the pattern there has to resolve to exactly one
function, defined in a file `current_test` names, asserting the row's feature against the row's
oracle. Every other row still needs a `test_name` (pytest_identity_problems below). The patterns stay
out of the declaration's `recipe_reader`: the family's gtest recipes declare none, and a partial map
would read every one of them as unreadable.
"""
import os
import re
import sys

import scanlib

SOURCES = [
    "test_3d_ngldm_mechanics.h",
    "test_3d_ngldm_mirp.h",
    "test_3d_ngldm_regression.h",
    os.path.join("python", "test_nyxus.py"),
]

ORACLE_SUFFIX = {"mirp": "mirp"}

# The volmin table is a Python dict in a pytest file. One declaration reads one table dialect and the
# family's own tables are C++ initialisers, so the pytest file has a declaration of its own and its
# scan joins the family's in collect() below.
VOLMIN = scanlib.Family(
    dim="3D", family="ngldm",
    sources=[os.path.join("python", "test_3d_ngldm_mirp.py")],
    oracle_suffix=ORACLE_SUFFIX,
    table_owner={"NGLDM_3D_VOLMIN_MIRP": "test_3d_ngldm_volmin_mirp"},
    table_dialect="python",
)

# recipe -> the pytest function that asserts AT that recipe; see the module docstring
PYTEST_RECIPES = {"ngldm3d.mirp_volmin_fbn8": re.compile(r"^test_3d_ngldm_volmin_mirp$")}

NOTE = {
    "3NGLDM_DCP": "no oracle row: Nyxus hard-codes f_DCP = 1 and MIRP returns 1 on any input where "
                  "every voxel has a same-level neighbour, so the agreement cannot fail",
    "3NGLDM_GLM": "no counterpart in any tool (MIRP's NGLDM emits no gl_mean column)",
    "3NGLDM_DCM": "no counterpart in any tool (MIRP's NGLDM emits no dc_mean column)",
}

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
                     if PYTEST_RECIPES[recipe].match(fn) and cov.where[fn] in claimed
                     and scanlib.oracle_token(fam, fn) == r["oracle"].strip())
        if len(fns) != 1:
            out.append(f"{f}: config_recipe {recipe} resolves to {fns or 'no function'} asserting "
                       f"it against {r['oracle'] or 'no oracle'} in {sorted(claimed)}; a pytest row "
                       f"needs one")
    return out


FAMILY = scanlib.Family(
    dim="3D", family="ngldm",
    sources=SOURCES,
    # The family's oracle is MIRP at ngldm3d.mirp_samelevels, where both tools evaluate the NGLDM
    # over one grey-level ladder (oracles/gen_ngldm3d_mirp.py, test_3d_ngldm_mirp.h). 16 of the 19
    # features carry a vetted row there; 3NGLDM_DCP is a non-discriminating agreement and
    # 3NGLDM_GLM / 3NGLDM_DCM have no MIRP counterpart, so those three stay regression-only.
    oracle_suffix=ORACLE_SUFFIX,
    table_owner={"ngldm_3d_mirp_negdepth_ref_vals": "test_3d_ngldm_negative_greydepth_mirp"},
    notes=NOTE,
    enum_dim_prefix=True,
    enum_alias="NGLDM",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
    # missing_test_name is pytest_identity_problems, which reads a pytest row by its recipe
    checks=scanlib.CORE_CHECKS | (scanlib.IDENTITY_CHECKS - {"missing_test_name"}),
    extra_problems=pytest_identity_problems,
    collect_override=collect,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
