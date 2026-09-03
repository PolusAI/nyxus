"""Regenerate gldzm_3d_coverage.csv by scanning the 3D GLDZM tests. Stdlib only.

    python tests/vetting/audit/scan_gldzm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import sys

import scanlib

# The family's whole in-tree footprint. test_3d_coverage_common.h also reads this file's pin table,
# but only for its key set -- it asserts a count and a has-a-pin property, never a feature value --
# so it covers nothing per-feature and is not scanned, matching every other 3D scanner here.
SOURCES = ["test_3d_gldzm_regression.h"]

# Ratios are Nyxus / MIRP on the ut_mask57 phantom, from audit/gldzm_3d_mirp_vetting_report.md.
# Only the two features with no counterpart anywhere and the three widest divergences are named --
# the report carries all 16, and repeating them here would put a second copy of that table in a
# generated file.
NOTE = {
    "3GLDZM_GLM": "no counterpart in any tool (MIRP's GLDZM emits no dzm_gl_mean column)",
    "3GLDZM_ZDM": "no counterpart in any tool (MIRP's GLDZM emits no dzm_zd_mean column); its pin "
                  "was 222 with no test function and no registration until this family's audit, so "
                  "it could not fail -- Nyxus computes 15.31",
    "3GLDZM_LDHGLE": "67.5x from MIRP, the family's widest measured divergence",
    "3GLDZM_LDE": "28.0x from MIRP",
    "3GLDZM_ZDV": "24.6x from MIRP",
}

FAMILY = scanlib.Family(
    dim="3D", family="gldzm", out="gldzm_3d_coverage.csv",
    sources=SOURCES,
    # Deliberately empty: no oracle test exists for this family, and none should be added on the
    # current implementation. MIRP is config-matched and reproducible (oracles/gen_gldzm3d_mirp.py)
    # but disagrees with Nyxus on all 16 features it computes, by up to 67.5x, and an independent
    # from-definition recomputation reproduces MIRP to rel 3.2e-16 -- so the gap is Nyxus'. Add
    # {"mirp": "mirp"} once the zone-map defects in the audit report are fixed.
    oracle_suffix={},
    notes=NOTE,
    enum_dim_prefix=True,
    other_note="asserted",
    loop_tables=True,
    checks=scanlib.CORE_CHECKS | scanlib.IDENTITY_CHECKS,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
