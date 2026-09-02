"""Regenerate ngldm_3d_coverage.csv by scanning the 3D NGLDM tests. Stdlib only.

    python tests/vetting/audit/scan_ngldm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_3d_ngldm_regression.h",
    os.path.join("python", "test_nyxus.py"),
]

NOTE = {
    "3NGLDM_DCP": "was status=vetted/oracle=mirp from an offline run; demoted -- it agrees with MIRP "
                  "only at the degenerate value 1.0, while every other comparable feature diverges",
    "3NGLDM_GLM": "no counterpart in any tool (MIRP's NGLDM emits no gl_mean column)",
    "3NGLDM_DCM": "no counterpart in any tool (MIRP's NGLDM emits no dc_mean column)",
    "3NGLDM_GLNU": "26.5x from MIRP -- consistent with background voxels piling into one grey row",
    "3NGLDM_DCENE": "49.9x from MIRP, the family's widest measured divergence",
    "3NGLDM_HDE": "9.3x from MIRP -- background voxels each see ~24 identical neighbours",
}

FAMILY = scanlib.Family(
    dim="3D", family="ngldm", out="ngldm_3d_coverage.csv",
    sources=SOURCES,
    # Deliberately empty: no oracle test exists for this family. MIRP is config-matched and
    # reproducible (oracles/gen_ngldm3d_mirp.py) but disagrees with Nyxus on 16 of 17 features for
    # implementation reasons, so nothing here is vetted and no function carries an oracle suffix.
    # Add {"mirp": "mirp"} once the defects in the audit report are fixed.
    oracle_suffix={},
    notes=NOTE,
    enum_dim_prefix=True,
    enum_alias="NGLDM",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
    checks=scanlib.CORE_CHECKS | scanlib.IDENTITY_CHECKS,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
