"""Regenerate gldzm_3d_coverage.csv by scanning the 3D GLDZM tests. Stdlib only.

    python tests/vetting/audit/scan_gldzm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import sys

import scanlib

# The family's whole in-tree footprint. test_3d_gldzm_common.h is the shared fixture and asserts no
# feature, so it is not scanned. test_3d_coverage_common.h reads both pin tables, but only for their
# key sets -- it asserts a count and a has-a-pin property, never a feature value -- so it covers
# nothing per-feature either, matching every other 3D scanner here.
SOURCES = ["test_3d_gldzm_mirp.h", "test_3d_gldzm_regression.h"]

# Only the two features no tool reaches are named. The other 16 are vetted against MIRP on
# bench_compat_gldzm_3d and their measurements live in audit/gldzm_3d_mirp_vetting_report.md;
# repeating them here would put a second copy of that table in a generated file.
NOTE = {
    "3GLDZM_GLM": "no counterpart in any tool (MIRP's GLDZM emits no dzm_gl_mean column) and none "
                  "in IBSI, so it is a drift guard at every recipe",
    "3GLDZM_ZDM": "no counterpart in any tool (MIRP's GLDZM emits no dzm_zd_mean column) and none "
                  "in IBSI, so it is a drift guard at every recipe",
}

FAMILY = scanlib.Family(
    dim="3D", family="gldzm", out="gldzm_3d_coverage.csv",
    sources=SOURCES,
    # MIRP is the only mainstream oracle for this family -- PyRadiomics implements no GLDZM at all.
    # The vetted cell is gldzm3d.mirp_compat_phantom, where neither tool discretises, so what the
    # comparison measures is the GLDZM rather than the binning the two disagree about on a CT-like
    # fixture (audit/gldzm_3d_mirp_vetting_report.md).
    oracle_suffix={"mirp": "mirp"},
    notes=NOTE,
    enum_dim_prefix=True,
    other_note="asserted",
    loop_tables=True,
    checks=scanlib.CORE_CHECKS | scanlib.IDENTITY_CHECKS,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
