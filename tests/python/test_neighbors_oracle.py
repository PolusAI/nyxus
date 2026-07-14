"""Regression / bug-exposure tests for 2D feature defects found during oracle validation
(2026-06). These exercise the PRODUCTION featurize() path on ROIs *with background* and at
the DEFAULT settings - the conditions the C++ unit tests miss.

This module covers the neighbor closest-distance/angle (fix #12) and PERCENT_TOUCHING
dedupe/adjacency (fix #13) defects.
"""
import numpy as np
import nyxus


# ----------------------------- helpers --------------------------------------
def _run(features, inten, label, **kw):
    nyx = nyxus.Nyxus(features=features, n_feature_calc_threads=1, **kw)
    df = nyx.featurize(inten.astype(np.float64), label.astype(np.uint32),
                       intensity_names=["i"], label_names=["l"])
    return df  # one row per label


# ============================ NEIGHBORS =====================================
def test_neighbor_distance_and_percent_touching():
    """Fixes #12 (closest-neighbor dist/ang need CENTROID, now forced when neighbor features are
    requested - reduce_trivial_rois.cpp) and #13 (PERCENT_TOUCHING deduped + adjacency-based, was
    >100% because per-neighbor touch counts were summed - neighbors.cpp). Two 4x4 squares separated
    by a 2px gap: they ARE neighbors within neighbor_distance=5 but do NOT touch (adjacency)."""
    H, W = 12, 20
    inten = np.zeros((H, W), np.uint32)
    label = np.zeros((H, W), np.uint32)
    inten[4:8, 3:7] = 100;  label[4:8, 3:7] = 1
    inten[4:8, 9:13] = 100; label[4:8, 9:13] = 2
    df = _run(["*ALL_NEIGHBOR*"], inten, label, neighbor_distance=5)
    assert (df["NUM_NEIGHBORS"] >= 1).all()
    assert (df["CLOSEST_NEIGHBOR1_DIST"] > 0).all()        # was 0 (CENTROID not computed) == bug #12
    assert (df["PERCENT_TOUCHING"] <= 100.0).all()         # was >100% == bug #13
    assert (df["PERCENT_TOUCHING"] == 0.0).all()           # 2px gap -> not touching (adjacency-based)


def test_percent_touching_fully_enclosed_is_100():
    """exact_min_sqdist touch fix: a ROI completely surrounded by face-adjacent neighbors has
    EVERY contour pixel 8-adjacent to a neighbor, so PERCENT_TOUCHING must be exactly 100. The approximate
    min_sqdist v2 (hill-descent on the closed contour) overestimated some pixels min distance and reported
    < 100. Center 3x3 (label 1) is boxed in on all four sides by directly-touching 3x3 blocks."""
    H, W = 12, 12
    inten = np.zeros((H, W), np.uint32)
    label = np.zeros((H, W), np.uint32)
    inten[4:7, 4:7] = 100; label[4:7, 4:7] = 1   # center (fully enclosed)
    inten[1:4, 4:7] = 100; label[1:4, 4:7] = 2   # above, touching
    inten[7:10, 4:7] = 100; label[7:10, 4:7] = 3 # below, touching
    inten[4:7, 1:4] = 100; label[4:7, 1:4] = 4   # left, touching
    inten[4:7, 7:10] = 100; label[4:7, 7:10] = 5 # right, touching
    df = _run(["*ALL_NEIGHBOR*"], inten, label, neighbor_distance=2)
    center = df[df["label"] == 1] if "label" in df.columns else df.iloc[[0]]
    assert (df["PERCENT_TOUCHING"] <= 100.0).all()
    assert center["PERCENT_TOUCHING"].iloc[0] == 100.0     # was < 100 (approximate min_sqdist missed touches)
