"""Regenerate morphology_3d_coverage.csv by scanning the 3D morphology tests. Stdlib only.

    python tests/vetting/audit/scan_morphology3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import sys

import scanlib

SOURCES = [
    "test_3d_morphology_matlab.h",
    "test_3d_morphology_mirp.h",
    "test_3d_morphology_regression.h",
]

NOTE = {
    "3AREA": "Nyxus counts exposed voxel faces (59992); MIRP/pyradiomics integrate a marching-cubes "
             "mesh (46739), a 28% convention difference -- regression-only until it is settled",
    "3AREA_2_VOLUME": "inherits the 3AREA surface-area convention difference",
    "3COMPACTNESS1": "inherits the 3AREA surface-area convention difference",
    "3COMPACTNESS2": "inherits the 3AREA surface-area convention difference",
    "3SPHERICITY": "inherits the 3AREA surface-area convention difference",
    "3SPHERICAL_DISPROPORTION": "inherits the 3AREA surface-area convention difference",
    "3MESH_VOLUME": "Nyxus aliases this to the convex-hull volume rather than integrating the mesh; "
                    "MIRP and MATLAB separately assert the hull quantity at a 5% band",
    "3VOLUME_CONVEXHULL": "discrete voxel hull (479997.83) vs MIRP's triangulated qhull volume, "
                          "measured 3.41%; MATLAB regionprops3 separately asserts ConvexVolume "
                          "497824 at 3.58%; "
                          "the two oracles agree with each other to 0.17%",
    "3VOXEL_VOLUME": "MIRP morph_vol_approx and MATLAB Volume both report 274432; each separately "
                     "asserts Nyxus 274431.358260 within rel=1e-3 (2.338e-04% residual)",
    "3MAJOR_AXIS_LEN": "the eigenvalue-order defect that once made LEAST>MAJOR is guarded by the MIRP "
                       "pins and by the generator's identity check",
    "3FLATNESS": "was >1 before the eigenvalue-order fix, which is structurally impossible",
    "3LEAST_AXIS_LEN": "was >3MAJOR_AXIS_LEN before the eigenvalue-order fix, structurally impossible",
}

FAMILY = scanlib.Family(
    dim="3D", family="morphology", out="morphology_3d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"matlab": "matlab", "mirp": "mirp"},
    notes=NOTE,
    enum_dim_prefix=True,
    enum_alias="MORPHOLOGY",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
    # the registry carries several oracle rows per feature here, and this artifact is the feature
    # rollup, so the summary counts what it emitted rather than the rows behind it
    count_noun="features",
    # ... and for the same reason a covering file is measured against the union of the feature's
    # rows, not each row alone, with each row additionally required to name its own oracle's file
    current_scope="feature",
    checks=scanlib.DEFAULT_CHECKS | scanlib.ORACLE_FILE_CHECK,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
