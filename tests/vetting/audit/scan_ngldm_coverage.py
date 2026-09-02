"""Regenerate ngldm_2d_coverage.csv by scanning the 2D NGLDM tests. Stdlib only.

    python tests/vetting/audit/scan_ngldm_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_2d_ngldm_ibsi.h",
    "test_2d_ngldm_mirp.h",
    "test_2d_ngldm_regression.h",
]

# The 17 IBSI features are deliberately asserted twice, by test_2d_ngldm_ibsi.h and
# test_2d_ngldm_mirp.h. That is not redundancy: the IBSI consensus values are published to three
# significant figures and fix the DEFINITION (rel=1e-2), while mirp reproduces Nyxus to 2.9e-16 and
# fixes the DIGITS (rel=1e-9). Dropping either weakens the family - see
# audit/ngldm_2d_mirp_vetting_report.md.
DUAL_ORACLE = ("asserted against both oracles by design: IBSI fixes the definition at its published "
               "3-significant-figure precision, mirp fixes the digits at 2.9e-16")

NOTE = {
    "NGLDM_GLM": "no IBSI NGLDM counterpart and no mirp column; regression drift guard only",
    "NGLDM_DCM": "no IBSI NGLDM counterpart and no mirp column; regression drift guard only",
}

FAMILY = scanlib.Family(
    dim="2D", family="ngldm", out="ngldm_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"mirp": "mirp", "ibsi": "ibsi"},
    notes=scanlib.dual_oracle_notes(NOTE, DUAL_ORACLE),
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
