"""Whole-slide (single-ROI) contour and edge-intensity behaviour, pinned.

Kind: *regression* per tests/vetting/SPEC.md 2 -- these pin CURRENT behaviour and establish no
vetting. They exist because `SINGLEROI=true` is a reachable production configuration for these
features (the Nyxus whole-slide mode) and SPEC 5.1 requires every cell of the config matrix to carry
a disposition rather than sit unmeasured. This cell is VALID-BUT-PRODUCTION-ONLY: real, reachable,
and reproduced by no external tool -- so it is a snapshot, never an oracle row. See
tests/vetting/matrix/morphology.md.

What the cell computes. `ContourFeature::calculate()` branches on `SINGLEROI` and, when it is set,
calls `buildWholeSlideContour()`, which does not trace a boundary at all: it pushes the four AABB
corner pixels, each carrying `r.aux_max` as its intensity. Every edge statistic is therefore a
statistic of four synthetic corner points at one value, and is degenerate by construction:

    EDGE_MIN == EDGE_MAX == EDGE_MEAN == the image maximum
    EDGE_STDDEV                        == 0            (four identical values)
    EDGE_INTEGRATED                    == 4 * maximum
    PERIMETER                          == the AABB perimeter, not the object's

That is a different quantity from the per-object edge statistics the segmented cell computes and
CellProfiler vets (test_2d_morphology_cellprofiler.h), which is why the vetted rows state their
scope as the segmented in-RAM path only. Pinning the degeneracy here is what stops a reader assuming
the CellProfiler evidence carries over to whole-slide mode, and makes any change to
`buildWholeSlideContour()` visible.
"""
import os

import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

# 64x64 image, one disk ROI, intensities 1 + x + 7y inside it -> maximum 397 at the far corner of
# the disk. The same fixture the out-of-core tests use, so the three matrix cells are measured on
# one shape and their numbers are directly comparable.
IMAGE_MAX = 397.0
SEGMENTED = {
    "PERIMETER": 131.88225099390849,
    "MASS_DISPLACEMENT": 2.7526140113386943,
    "EDGE_MEAN_INTENSITY": 257.0,
    "EDGE_STDDEV_INTENSITY": 98.12659593009853,
    "EDGE_MAX_INTENSITY": 397.0,
    "EDGE_MIN_INTENSITY": 117.0,
    "EDGE_INTEGRATED_INTENSITY": 28784.0,
}
WHOLE_SLIDE = {
    "PERIMETER": 256.0,                 # 4 * 64, the AABB walk
    "MASS_DISPLACEMENT": 3.3453118163885427,
    "EDGE_MEAN_INTENSITY": 397.0,
    "EDGE_STDDEV_INTENSITY": 0.0,
    "EDGE_MAX_INTENSITY": 397.0,
    "EDGE_MIN_INTENSITY": 397.0,
    "EDGE_INTEGRATED_INTENSITY": 1588.0,  # 4 * 397
}

FEATURES = ["*ALL_MORPHOLOGY*", "*BASIC_MORPHOLOGY*"]


def _fixture(tmp_path):
    Y = X = 64
    yy, xx = np.mgrid[0:Y, 0:X]
    mask = (((yy - 32) ** 2 + (xx - 32) ** 2) <= 20 * 20).astype(np.uint32)
    inten = ((1 + xx + yy * 7) * mask).astype(np.uint32)
    assert inten.max() == IMAGE_MAX, "fixture maximum moved; the pins below are keyed to it"
    ip = tmp_path / "img.tif"
    sp = tmp_path / "seg.tif"
    tifffile.imwrite(str(ip), inten)
    tifffile.imwrite(str(sp), mask)
    return str(ip), str(sp)


def _featurize(ip, sp, single_roi):
    return nyxus.Nyxus(FEATURES).featurize_files([ip], [sp], single_roi)


def test_2d_morphology_whole_slide_edge_intensity_is_degenerate_regression(tmp_path):
    """The whole-slide cell returns corner-walk statistics, and they are pinned as such.

    Asserted as identities against the image maximum rather than only as literals, so the pins say
    WHY each value is what it is: a fix that made whole-slide mode trace the real slide boundary
    would break all of them, which is the intended signal.
    """
    ip, sp = _fixture(tmp_path)
    df = _featurize(ip, sp, True)
    got = {c: float(df[c].iloc[0]) for c in WHOLE_SLIDE}

    assert got["PERIMETER"] == pytest.approx(WHOLE_SLIDE["PERIMETER"], rel=1e-9)
    assert got["MASS_DISPLACEMENT"] == pytest.approx(WHOLE_SLIDE["MASS_DISPLACEMENT"], rel=1e-9)
    assert got["EDGE_MEAN_INTENSITY"] == pytest.approx(WHOLE_SLIDE["EDGE_MEAN_INTENSITY"], rel=1e-9)
    assert got["EDGE_STDDEV_INTENSITY"] == pytest.approx(WHOLE_SLIDE["EDGE_STDDEV_INTENSITY"], rel=1e-9)
    assert got["EDGE_MAX_INTENSITY"] == pytest.approx(WHOLE_SLIDE["EDGE_MAX_INTENSITY"], rel=1e-9)
    assert got["EDGE_MIN_INTENSITY"] == pytest.approx(WHOLE_SLIDE["EDGE_MIN_INTENSITY"], rel=1e-9)
    assert got["EDGE_INTEGRATED_INTENSITY"] == pytest.approx(WHOLE_SLIDE["EDGE_INTEGRATED_INTENSITY"], rel=1e-9)

    # the degeneracy stated as relations, not just numbers
    assert got["EDGE_MIN_INTENSITY"] == got["EDGE_MAX_INTENSITY"] == got["EDGE_MEAN_INTENSITY"]
    assert got["EDGE_MEAN_INTENSITY"] == pytest.approx(IMAGE_MAX, rel=1e-9)
    assert got["EDGE_STDDEV_INTENSITY"] == 0.0
    assert got["EDGE_INTEGRATED_INTENSITY"] == pytest.approx(4.0 * IMAGE_MAX, rel=1e-9)


def test_2d_morphology_whole_slide_differs_from_segmented_regression(tmp_path):
    """Whole-slide and segmented are different quantities, and nothing should read across.

    The CellProfiler vetting in test_2d_morphology_cellprofiler.h covers the segmented in-RAM cell.
    This asserts the two cells actually disagree, so a future reader cannot quietly treat that
    evidence as covering whole-slide mode -- and so a change that accidentally made whole-slide take
    the segmented path shows up here rather than silently widening what the vetted rows claim.
    """
    ip, sp = _fixture(tmp_path)
    seg = _featurize(ip, sp, False)
    ws = _featurize(ip, sp, True)

    assert float(seg["PERIMETER"].iloc[0]) == pytest.approx(SEGMENTED["PERIMETER"], rel=1e-9)
    assert float(seg["MASS_DISPLACEMENT"].iloc[0]) == pytest.approx(SEGMENTED["MASS_DISPLACEMENT"], rel=1e-9)
    assert float(seg["EDGE_MEAN_INTENSITY"].iloc[0]) == pytest.approx(SEGMENTED["EDGE_MEAN_INTENSITY"], rel=1e-9)
    assert float(seg["EDGE_STDDEV_INTENSITY"].iloc[0]) == pytest.approx(SEGMENTED["EDGE_STDDEV_INTENSITY"], rel=1e-9)
    assert float(seg["EDGE_MAX_INTENSITY"].iloc[0]) == pytest.approx(SEGMENTED["EDGE_MAX_INTENSITY"], rel=1e-9)
    assert float(seg["EDGE_MIN_INTENSITY"].iloc[0]) == pytest.approx(SEGMENTED["EDGE_MIN_INTENSITY"], rel=1e-9)
    assert float(seg["EDGE_INTEGRATED_INTENSITY"].iloc[0]) == pytest.approx(SEGMENTED["EDGE_INTEGRATED_INTENSITY"], rel=1e-9)

    same = [c for c in SEGMENTED
            if float(seg[c].iloc[0]) == pytest.approx(float(ws[c].iloc[0]), rel=1e-6)]
    # EDGE_MAX coincides here only because the disk's brightest pixel is also the image maximum,
    # which is a property of this fixture rather than of the two paths.
    assert same == ["EDGE_MAX_INTENSITY"], (
        "expected only EDGE_MAX_INTENSITY to coincide between the whole-slide and segmented cells, "
        "got %r" % (same,)
    )
