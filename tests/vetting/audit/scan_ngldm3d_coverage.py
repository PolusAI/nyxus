"""Scan the 3D NGLDM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_ngldm3d_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_3d_ngldm_mirp.h",
    "test_3d_ngldm_regression.h",
    os.path.join("python", "test_nyxus.py"),
]

NOTE = {
    "3NGLDM_DCP": "no oracle row: Nyxus hard-codes f_DCP = 1 and MIRP returns 1 on any input where "
                  "every voxel has a same-level neighbour, so the agreement cannot fail",
    "3NGLDM_GLM": "no counterpart in any tool (MIRP's NGLDM emits no gl_mean column)",
    "3NGLDM_DCM": "no counterpart in any tool (MIRP's NGLDM emits no dc_mean column)",
}

FAMILY = scanlib.Family(
    dim="3D", family="ngldm",
    sources=SOURCES,
    # The family's oracle is MIRP at ngldm3d.mirp_samelevels, where both tools evaluate the NGLDM
    # over one grey-level ladder (oracles/gen_ngldm3d_mirp.py, test_3d_ngldm_mirp.h). 16 of the 19
    # features carry a vetted row there; 3NGLDM_DCP is a non-discriminating agreement and
    # 3NGLDM_GLM / 3NGLDM_DCM have no MIRP counterpart, so those three stay regression-only.
    oracle_suffix={"mirp": "mirp"},
    notes=NOTE,
    enum_dim_prefix=True,
    enum_alias="NGLDM",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
    checks=scanlib.CORE_CHECKS | scanlib.IDENTITY_CHECKS,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
