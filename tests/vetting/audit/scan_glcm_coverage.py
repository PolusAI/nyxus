"""Scan the 2D GLCM tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_glcm_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_2d_glcm_ibsi.h",
    "test_2d_glcm_mirp.h",
    "test_2d_glcm_pyradiomics.h",
    "test_2d_glcm_regression.h",
    "test_2d_glcm_mechanics.h",
    os.path.join("python", "test_2d_glcm_pyradiomics.py"),
]

ALIAS = {
    "GLCM_ASM": "same quantity as GLCM_ENERGY",
    "GLCM_DIS": "equals GLCM_DIFAVE; PyRadiomics dropped Dissimilarity as that duplicate, MIRP "
                "reports it natively",
    "GLCM_ENERGY": "same quantity as GLCM_ASM (PyRadiomics JointEnergy / MIRP cm_energy)",
    "GLCM_ENTROPY": "same quantity as GLCM_JE (joint entropy)",
    "GLCM_HOM1": "same quantity as GLCM_ID (PyRadiomics Id / MIRP cm_inv_diff)",
    "GLCM_HOM2": "same quantity as GLCM_IDM (PyRadiomics Idm / MIRP cm_inv_diff_mom); no _AVE twin "
                 "exists",
    "GLCM_SUMVARIANCE": "equals GLCM_CLUTEND; PyRadiomics dropped SumVariance as that duplicate, "
                        "MIRP reports it natively",
    "GLCM_VARIANCE": "same quantity as GLCM_JVAR (PyRadiomics SumSquares = joint variance)",
}

# the features that sum over logarithms, base names; each _AVE twin shares the tolerance
LOG_BASED = {"GLCM_DIFENTRO", "GLCM_ENTROPY", "GLCM_INFOMEAS1", "GLCM_INFOMEAS2", "GLCM_JE",
             "GLCM_SUMENTROPY"}
LOG = ("log-based: Nyxus sums through fast_log10 with an EPSILON guard, so it lands 1e-3..3e-3 from "
       "both tools instead of on them; asserted at rel=5e-3")
AVE = ("angle average; the tools report one value over the angle set, so it shares its base "
       "feature's golden")


def note(feature, cov):
    base = feature[:-len("_AVE")] if feature.endswith("_AVE") else feature
    parts = [ALIAS.get(feature, ""),
             LOG if base in LOG_BASED else "",
             AVE if feature != base else ""]
    return " | ".join(p for p in parts if p)


FAMILY = scanlib.Family(
    dim="2D", family="glcm",
    sources=SOURCES,
    oracle_suffix={"ibsi": "ibsi", "mirp": "mirp", "pyradiomics": "pyradiomics"},
    notes=note,
    # test_2d_glcm_acor_family_pyradiomics loops a dict of goldens rather than naming each feature
    py_loop_tables=True,
    # the two family-wide oracle tests loop a golden table keyed by feature name and never spell a
    # feature on an assertion line
    table_owner={
        "glcm_2d_mirp_ref_vals": "test_2d_glcm_family_mirp",
        "glcm_2d_pyradiomics_ref_vals": "test_2d_glcm_family_pyradiomics",
    },
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
