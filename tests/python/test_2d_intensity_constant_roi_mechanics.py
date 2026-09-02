"""Mechanics tests for a populated ROI sitting at the slide's own intensity minimum.

The load-time offset map puts the slide's all-pixel minimum on grey level 0, so an ROI that lies
entirely at that minimum -- an all-air CT ROI at -1013 HU, say -- reaches the intensity family with
aux_min == aux_max == 0. A blank-ROI guard written as `aux_min == 0 && aux_max == 0` cannot tell that
apart from an ROI with no pixels at all, and turned it away: the histogram never ran, so UNIFORMITY
kept its default 0 rather than the 1 a single-valued ROI has, ENTROPY was 0 by default rather than by
computation, and the per-bin HISTOGRAM came back empty. The guard now tests emptiness, and the
histogram treats a zero-width value range as the single bin it is.

Both the in-RAM and the out-of-core paths carry their own copy of that guard, so both are exercised
here. Kind: *mechanics* per tests/vetting/SPEC.md 2 -- these pin loader/guard plumbing rather than
feature values against a reference, and establish no vetting.

ram_limit is a process-global in Nyxus, so the out-of-core case sets it explicitly and puts it back.
"""
import numpy as np
import pytest
import nyxus

tifffile = pytest.importorskip("tifffile")

# Air in the reporter's own CT: the slide minimum, and the value the whole of ROI 1 sits at.
AIR_HU = -1013
TISSUE_HU = 200

# Big enough that ram_limit=1 MB pushes it through the out-of-core path.
H, W = 700, 700

RAM_LIMIT_LARGE_MB = 64

FEATURES = ["MIN", "MAX", "MEAN", "MEDIAN", "RANGE", "STANDARD_DEVIATION", "UNIFORMITY",
            "ENTROPY", "HISTOGRAM"]


def _write_pair(tmp_path):
    """A signed CT-like slide whose ROI 1 is entirely at the slide's minimum.

    ROI 2 holds a different value so the slide itself has a range -- without it the offset map would
    be recording the minimum of a constant slide, which is a different case (test_hu_analytic.h).
    """
    inten = np.full((H, W), AIR_HU, np.int16)
    inten[600:660, 600:660] = TISSUE_HU

    lab = np.zeros((H, W), np.uint16)
    lab[0:500, 0:500] = 1        # entirely inside the air region -> constant at the slide minimum
    lab[600:660, 600:660] = 2    # the tissue patch

    ip = tmp_path / "ct.tif"
    sp = tmp_path / "seg.tif"
    tifffile.imwrite(str(ip), inten)
    tifffile.imwrite(str(sp), lab)
    return [str(ip)], [str(sp)]


def _set_ram_limit_mb(nyx, mb):
    """Set the process-global ram_limit and verify it was accepted.

    Nyxus refuses a limit above the RAM currently available and keeps the previous value without
    raising, which would silently run the "out-of-core" case in RAM."""
    nyx.set_params(ram_limit=mb)
    got = nyx.get_params("ram_limit")["ram_limit"]
    assert got == mb, (
        "ram_limit=%d MB was not accepted (still %d MB) -- Nyxus refuses a limit above available "
        "RAM." % (mb, got)
    )


def _assert_constant_roi_at_slide_minimum(row):
    # Reported in the slide's own domain, not in the grey levels the offset map stored.
    assert row["MIN"] == pytest.approx(float(AIR_HU))
    assert row["MAX"] == pytest.approx(float(AIR_HU))
    assert row["MEAN"] == pytest.approx(float(AIR_HU))
    assert row["MEDIAN"] == pytest.approx(float(AIR_HU))
    assert row["RANGE"] == pytest.approx(0.0)
    assert row["STANDARD_DEVIATION"] == pytest.approx(0.0)

    # The distribution of a single-valued ROI: one occupied bin. This is what the blank-ROI guard
    # used to suppress -- UNIFORMITY stayed at its 0 default instead of reaching 1.
    assert row["UNIFORMITY"] == pytest.approx(1.0)
    assert row["ENTROPY"] == pytest.approx(0.0, abs=1e-9)


def test_2d_intensity_constant_roi_at_slide_minimum_mechanics(tmp_path):
    ifiles, sfiles = _write_pair(tmp_path)

    nyx = nyxus.Nyxus(features=FEATURES, n_feature_calc_threads=1)
    _set_ram_limit_mb(nyx, RAM_LIMIT_LARGE_MB)
    df = nyx.featurize_files(ifiles, sfiles, False)

    air = df[df["ROI_label"] == 1].iloc[0]
    _assert_constant_roi_at_slide_minimum(air)

    # The per-bin histogram is emitted rather than left empty.
    hist_cols = [c for c in df.columns if c.startswith("HISTOGRAM")]
    assert len(hist_cols) > 0
    assert air[hist_cols].sum() == pytest.approx(500.0 * 500.0)

    # The other ROI is unaffected: a plain, non-minimum ROI still reports its own value.
    tissue = df[df["ROI_label"] == 2].iloc[0]
    assert tissue["MIN"] == pytest.approx(float(TISSUE_HU))
    assert tissue["MAX"] == pytest.approx(float(TISSUE_HU))


def test_2d_intensity_constant_roi_at_slide_minimum_ooc_mechanics(tmp_path):
    """The out-of-core path carries its own copy of the guard and must agree."""
    ifiles, sfiles = _write_pair(tmp_path)

    nyx = nyxus.Nyxus(features=FEATURES, n_feature_calc_threads=1)
    try:
        _set_ram_limit_mb(nyx, 1)
        df = nyx.featurize_files(ifiles, sfiles, False)
    finally:
        _set_ram_limit_mb(nyx, RAM_LIMIT_LARGE_MB)

    air = df[df["ROI_label"] == 1].iloc[0]
    _assert_constant_roi_at_slide_minimum(air)
