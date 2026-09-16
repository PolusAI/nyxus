"""Scan the 2D GLSZM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_glszm_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `test_output.csv`. This file is
the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_2d_glszm_ibsi.h",
    "test_2d_glszm_mirp.h",
    "test_2d_glszm_regression.h",
]

# All sixteen features are deliberately asserted twice, by test_2d_glszm_ibsi.h and
# test_2d_glszm_mirp.h. That is not redundancy: the IBSI consensus values are published to three
# significant figures and fix the DEFINITION (rel=1e-2), while mirp agrees with Nyxus to within
# 2.0e-16 and fixes the DIGITS (SPEC 7's exact tier, an absolute 1e-9 band). Dropping either weakens
# the family - see audit/glszm_2d_mirp_vetting_report.md.
DUAL_ORACLE = ("asserted against both oracles by design: IBSI fixes the definition at its published "
               "3-significant-figure precision, mirp fixes the digits at 2.0e-16")

NOTE = {
    "GLSZM_ZE": ("dual-oracle like the rest, but asserted against mirp at rel=4e-3 rather than at "
                 "the exact tier: Nyxus takes the logarithm through fast_log10, a float-precision "
                 "approximation, costing 2.5e-3 per slice"),
}

FAMILY = scanlib.Family(
    dim="2D", family="glszm",
    sources=SOURCES,
    oracle_suffix={"mirp": "mirp", "ibsi": "ibsi"},
    notes=scanlib.dual_oracle_notes(NOTE, DUAL_ORACLE),
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
