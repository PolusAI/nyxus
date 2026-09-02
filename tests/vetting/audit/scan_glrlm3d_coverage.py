"""Regenerate glrlm_3d_coverage.csv by scanning the 3D GLRLM tests. Stdlib only.

    python tests/vetting/audit/scan_glrlm3d_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The coverage rule, the checks and the rendering all live in scanlib.py; this file
is the family's declaration.
"""
import os
import sys

import scanlib

SOURCES = [
    "test_3d_glrlm_pyradiomics.h",
    "test_3d_glrlm_regression.h",
    os.path.join("python", "test_nyxus.py"),
]

NOTE = {
    "3GLRLM_RP": "exceeds its mathematical bound of 1 at positive GLRLM_GREYDEPTH values; in range "
                 "at the binCount binning both tests use",
    "3GLRLM_RP_AVE": "registry read oracle=mirp while the base row read oracle=pyradiomics for the "
                     "same quantity; no mirp run for 3D GLRLM has ever existed in the tree",
    "3GLRLM_RE": "the family's only sum over logarithms; fast_log10 puts it 3.9e-4 from "
                 "PyRadiomics, so it is asserted at rel=5e-3 where the rest are at rel=1e-9",
    "3GLRLM_RE_AVE": "same mirp mislabel as 3GLRLM_RP_AVE, and the same log tolerance as 3GLRLM_RE",
}

FAMILY = scanlib.Family(
    dim="3D", family="glrlm", out="glrlm_3d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"pyradiomics": "pyradiomics"},
    notes=NOTE,
    enum_dim_prefix=True,
    # the _AVE test aliases the enum (`using F = Nyxus::Feature3D;`), so cover `F::` too
    enum_alias="GLRLM",
    other_note="asserted",
    scan_helpers=True, loop_tables=True,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
