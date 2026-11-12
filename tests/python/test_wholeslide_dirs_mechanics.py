"""Whole-slide detection from the two directory arguments of featurize_directory.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- runs of one dataset under different spellings of
its directory are compared with each other, and nothing is asserted against a reference.

featurize_directory runs whole-slide (one ROI per image) when the label directory is the intensity
directory. "The same directory" is a filesystem question, not a string one: a trailing separator,
a leading "./", an absolute path and a symbolic link all name it. A spelling the detection misses
runs segmented with the intensity image as its own mask, one ROI per distinct intensity, and
nothing in the output says so.
"""
import os

import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")


def _slide(tmp_path):
    """One 32 x 32 slide whose 200 distinct intensities would read as 200 ROIs if used as a mask."""
    d = tmp_path / "wsi"
    d.mkdir()
    img = (1 + np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) % 200)
    tifffile.imwrite(str(d / "s.tif"), img)
    return d


def _volume(tmp_path):
    """One 4 x 16 x 16 volume with 200 distinct intensities."""
    d = tmp_path / "vol"
    d.mkdir()
    vol = (1 + np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16) % 200)
    # photometric="minisblack" keeps a leading axis of 4 as Z rather than RGBA samples
    tifffile.imwrite(str(d / "v.ome.tif"), vol, photometric="minisblack", metadata={"axes": "ZYX"})
    return d


def _spellings(d):
    """Every way of writing directory d as the label directory, relative to its parent as cwd."""
    names = {
        "trailing slash": d.name + "/",
        "trailing os.sep": d.name + os.sep,
        "dot prefix": "./" + d.name,
        "dot-dot round trip": os.path.join(d.name, "..", d.name),
        "absolute": str(d.resolve()),
    }
    link = d.parent / (d.name + "_link")
    try:
        os.symlink(str(d), str(link), target_is_directory=True)
        names["symlink"] = link.name
    except (OSError, NotImplementedError):
        pass  # no symlink privilege on this host (Windows without developer mode)
    return names


# The control first: the same string twice is whole-slide, one row. Then every other spelling of the
# same directory must give that row too, value for value. What this discriminates: comparing the two
# arguments as strings, which reads "wsi" against "wsi/" as two directories and returns 200 rows.
def test_wholeslide_dirs_2d_any_spelling_is_one_roi_mechanics(tmp_path, monkeypatch):
    d = _slide(tmp_path)
    monkeypatch.chdir(tmp_path)

    ref = nyxus.Nyxus(["MEAN"]).featurize_directory("wsi", "wsi", ".*")
    assert len(ref) == 1, "the control: the same string twice is one whole-slide row"

    for what, seg in _spellings(d).items():
        df = nyxus.Nyxus(["MEAN"]).featurize_directory("wsi", seg, ".*")
        assert len(df) == 1, "%s (%r): %d rows, not one whole-slide row" % (what, seg, len(df))
        assert df["MEAN"].iloc[0] == ref["MEAN"].iloc[0], what


# The same detection on the 3D path, which decides between the whole-volume and the segmented
# workflow by it.
def test_wholeslide_dirs_3d_any_spelling_is_one_roi_mechanics(tmp_path, monkeypatch):
    d = _volume(tmp_path)
    monkeypatch.chdir(tmp_path)

    ref = nyxus.Nyxus3D(["3MEAN"]).featurize_directory("vol", "vol", ".*")
    assert len(ref) == 1, "the control: the same string twice is one whole-volume row"

    for what, seg in _spellings(d).items():
        df = nyxus.Nyxus3D(["3MEAN"]).featurize_directory("vol", seg, ".*")
        assert len(df) == 1, "%s (%r): %d rows, not one whole-volume row" % (what, seg, len(df))
        assert df["3MEAN"].iloc[0] == ref["3MEAN"].iloc[0], what


# Two directories that differ only in case are two directories on a case-sensitive filesystem: the
# second holds a mask, and the run is segmented. Where the filesystem folds case they are one
# directory and there is nothing to test.
def test_wholeslide_dirs_case_differing_dirs_stay_segmented_mechanics(tmp_path, monkeypatch):
    d = _slide(tmp_path)
    other = tmp_path / "WSI"
    if other.exists():
        pytest.skip("this filesystem folds case: WSI and wsi are one directory")
    other.mkdir()
    mask = np.zeros((32, 32), np.uint16)
    mask[4:12, 4:12] = 1
    mask[20:28, 20:28] = 2
    tifffile.imwrite(str(other / "s.tif"), mask)
    monkeypatch.chdir(tmp_path)

    df = nyxus.Nyxus(["MEAN"]).featurize_directory("wsi", "WSI", ".*")
    assert sorted(df["ROI_label"].tolist()) == [1, 2]


# The refusal: the backend compares the two directories on the filesystem, so a path that is not a
# directory is refused rather than compared as a string. The Python wrapper refuses a missing path
# first, so the backend is called directly to reach the C++ check; a file passes the wrapper's
# existence check and is refused by the backend through the wrapper itself.
@pytest.mark.parametrize("bad", ["missing", "file.tif"])
@pytest.mark.parametrize("which", ["intensity", "label"])
def test_wholeslide_dirs_missing_dir_refused_mechanics(tmp_path, monkeypatch, which, bad):
    _slide(tmp_path)
    (tmp_path / "file.tif").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    from nyxus.backend import featurize_directory_imp, featurize_directory_3D_imp

    intdir, segdir = (bad, "wsi") if which == "intensity" else ("wsi", bad)
    refused = "cannot access directory " + bad

    if bad == "missing":
        with pytest.raises(IOError, match="does not exist"):
            nyxus.Nyxus(["MEAN"]).featurize_directory(intdir, segdir, ".*")
    else:
        with pytest.raises(RuntimeError, match=refused):
            nyxus.Nyxus(["MEAN"]).featurize_directory(intdir, segdir, ".*")

    n2 = nyxus.Nyxus(["MEAN"])
    with pytest.raises(RuntimeError, match=refused):
        featurize_directory_imp(id(n2), intdir, segdir, ".*", "pandas", "")

    n3 = nyxus.Nyxus3D(["3MEAN"])
    with pytest.raises(RuntimeError, match=refused):
        featurize_directory_3D_imp(id(n3), intdir, segdir, ".*", "pandas", "")
