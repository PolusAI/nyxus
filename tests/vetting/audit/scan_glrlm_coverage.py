"""Scan the 2D GLRLM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_glrlm_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_2d_glrlm_ibsi.h",
    "test_2d_glrlm_mirp.h",
    "test_2d_glrlm_pyradiomics.h",
    "test_2d_glrlm_regression.h",
]

AVE = ("angle average; the tools report one value over the 4 directions, so it shares its base "
       "feature's golden")

FAMILY = scanlib.Family(
    dim="2D", family="glrlm",
    sources=SOURCES,
    oracle_suffix={"ibsi": "ibsi", "mirp": "mirp", "pyradiomics": "pyradiomics"},
    notes=lambda feature, cov: AVE if feature.endswith("_AVE") else "",
    # the two family-wide oracle tests loop a golden table keyed by feature name and never spell a
    # feature on an assertion line
    table_owner={
        "glrlm_2d_mirp_ref_vals": "test_2d_glrlm_family_mirp",
        "glrlm_2d_pyradiomics_ref_vals": "test_2d_glrlm_family_pyradiomics",
    },
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
