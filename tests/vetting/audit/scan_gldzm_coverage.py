"""Regenerate gldzm_2d_coverage.csv by scanning the 2D GLDZM tests. Stdlib only.

    python tests/vetting/audit/scan_gldzm_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_2d_gldzm_ibsi.h",
    "test_2d_gldzm_mirp.h",
    "test_2d_gldzm_regression.h",
]

# The 14 published IBSI features are deliberately asserted twice, by test_2d_gldzm_ibsi.h and
# test_2d_gldzm_mirp.h. That is not redundancy: the IBSI consensus values are published to three
# significant figures and fix the DEFINITION (rel=1e-2), while mirp reproduces Nyxus to 1.3e-15
# absolute and fixes the DIGITS (SPEC 7 exact tier, abs=1e-9). Dropping either weakens the family -
# see audit/gldzm_2d_mirp_vetting_report.md.
DUAL_ORACLE = ("asserted against both oracles by design: IBSI fixes the definition at its published "
               "3-significant-figure precision, mirp fixes the digits at 1.3e-15 absolute")

NOTE = {
    "GLDZM_GLM": "no IBSI GLDZM counterpart and no mirp column; regression drift guard only",
    "GLDZM_ZDM": "no IBSI GLDZM counterpart and no mirp column; regression drift guard only",
    "GLDZM_SDLGLE": "vetted by mirp only; the IBSI file never held a published value for it",
    "GLDZM_ZDV": "vetted by mirp only; the IBSI file never held a published value for it",
}

FAMILY = scanlib.Family(
    dim="2D", family="gldzm", out="gldzm_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"mirp": "mirp", "ibsi": "ibsi"},
    notes=scanlib.dual_oracle_notes(NOTE, DUAL_ORACLE),
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
