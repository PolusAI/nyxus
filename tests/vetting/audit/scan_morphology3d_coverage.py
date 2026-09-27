"""Scan the 3D morphology tests for the feature -> test mapping. Stdlib only.

    python tests/vetting/audit/scan_morphology3d_coverage.py --check

The mapping is read out of the test sources rather than written by hand, so it cannot drift from the
tree. `--check` runs the acceptance checks against `oracle_coverage.csv`. The coverage rule and the
checks live in scanlib.py; `report_features.py` joins the scan into `report_output.csv`. This file is
the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_3d_morphology_analytic.h",
    "test_3d_morphology_matlab.h",
    "test_3d_morphology_mirp.h",
    "test_3d_morphology_regression.h",
]

MESH = ("IBSI area (mesh) / volume (mesh) and the ratios of them: Nyxus and MIRP integrate the same "
        "marching-cubes surface, so the residual is MIRP's float32 storage of the mesh (<=2.4e-07) "
        "and the pin is rel=1e-6")

NOTE = {
    "3AREA": "IBSI area (mesh) = MIRP morph_area_mesh 46739.0195, pinned at rel=1e-6; also pinned "
             "against the octahedron and bevelled-box closed forms in the analytic file",
    "3AREA_2_VOLUME": MESH,
    "3COMPACTNESS1": MESH,
    "3COMPACTNESS2": MESH,
    "3SPHERICITY": MESH,
    "3SPHERICAL_DISPROPORTION": MESH,
    "3MESH_VOLUME": "IBSI volume (mesh) = MIRP morph_volume 274338.3438, pinned at rel=1e-6; also "
                    "pinned against closed-form solids in the analytic file. MATLAB regionprops3 "
                    "carries no mesh volume, so it holds no row for this feature",
    "3VOLUME_CONVEXHULL": "discrete voxel hull (480655.17) vs MIRP's triangulated qhull volume, "
                          "measured 3.28%; MATLAB regionprops3 separately asserts ConvexVolume "
                          "497824 at 3.45%; "
                          "the two oracles agree with each other to 0.17%",
    "3VOXEL_VOLUME": "MIRP morph_vol_approx and MATLAB Volume both report 274432; each separately "
                     "asserts Nyxus 274431.358260 within rel=1e-3 (2.338e-04% residual)",
    "3MAJOR_AXIS_LEN": "the eigenvalue-order defect that once made LEAST>MAJOR is guarded by the MIRP "
                       "pins and by the generator's identity check",
    "3FLATNESS": "was >1 before the eigenvalue-order fix, which is structurally impossible",
    "3LEAST_AXIS_LEN": "was >3MAJOR_AXIS_LEN before the eigenvalue-order fix, structurally impossible",
}

FAMILY = scanlib.Family(
    dim="3D", family="morphology",
    sources=SOURCES,
    oracle_suffix={"analytic": "analytic", "matlab": "matlab", "mirp": "mirp"},
    notes=NOTE,
    enum_dim_prefix=True,
    enum_alias="MORPHOLOGY",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
    # the registry carries several oracle rows per feature here, and the scan is the feature
    # rollup, so the summary counts features rather than the rows behind them
    count_noun="features",
    # ... and for the same reason a covering file is measured against the union of the feature's
    # rows, not each row alone, with each row additionally required to name its own oracle's file
    current_scope="feature",
    checks=scanlib.DEFAULT_CHECKS | scanlib.ORACLE_FILE_CHECK,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
