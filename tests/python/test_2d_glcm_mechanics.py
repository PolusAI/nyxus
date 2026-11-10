"""IBSI-mode 2D GLCM sizes its matrix by the ROI's maximum grey level. A ROI whose matrix cannot be
held gets the noval substitute for itself and a message naming it; the other ROIs of the image are
computed as if it were not there, in RAM and out of core alike.
"""
import numpy as np
import pytest
import nyxus

# the Python API has no noval option, so the substitute is the default
NOVAL = 0.0

# the largest level count whose matrix SimpleMatrix can index, plus one
FIRST_REFUSED_MAX = 46341

COMPUTED_PATCH = np.array([[10, 20, 30, 40],
                           [40, 10, 20, 30],
                           [30, 40, 10, 20],
                           [20, 30, 40, 200]], np.uint32)


def _refused_patch(roi_max):
    patch = np.full((4, 4), 1, np.uint32)
    patch[::2, ::2] = roi_max
    patch[1::2, 1::2] = roi_max
    return patch


def _featurize(inten, label):
    nyx = nyxus.Nyxus(["*ALL_GLCM*"], n_feature_calc_threads=1, ibsi=True)
    df = nyx.featurize(inten, label, intensity_names=["i"], label_names=["l"])
    return df.sort_values("ROI_label").reset_index(drop=True)


def _glcm_values(row):
    return row[[c for c in row.index if c.startswith("GLCM_")]].astype(float)


def _image(patches):
    """Patches side by side on background, labelled 1, 2, ... from the left."""
    inten = np.zeros((6, 6 * len(patches)), np.uint32)
    label = np.zeros_like(inten)
    for i, patch in enumerate(patches):
        inten[1:5, 1 + 6 * i:5 + 6 * i] = patch
        label[1:5, 1 + 6 * i:5 + 6 * i] = i + 1
    return inten, label


@pytest.mark.parametrize("refused_label", [1, 2])
@pytest.mark.parametrize("roi_max", [FIRST_REFUSED_MAX, 65535])
def test_2d_glcm_ibsi_refused_roi_is_isolated_mechanics(roi_max, refused_label, capfd):
    patches = [COMPUTED_PATCH, COMPUTED_PATCH]
    patches[refused_label - 1] = _refused_patch(roi_max)
    computed_label = 3 - refused_label

    df = _featurize(*_image(patches))
    err = capfd.readouterr().err

    solo = _glcm_values(_featurize(*_image([COMPUTED_PATCH])).iloc[0])
    computed = _glcm_values(df[df["ROI_label"] == computed_label].iloc[0])
    assert (solo != NOVAL).any()
    assert computed.to_numpy() == pytest.approx(solo.to_numpy(), rel=1e-12, abs=1e-12)

    refused = _glcm_values(df[df["ROI_label"] == refused_label].iloc[0])
    assert (refused == NOVAL).all(), refused[refused != NOVAL].to_dict()
    assert f"ROI {refused_label}" in err and "GLCM" in err, err


def _tile(patch, side=500):
    return np.tile(patch, (side // patch.shape[0], side // patch.shape[1]))


@pytest.mark.parametrize("refused_label", [1, 2])
def test_2d_glcm_ibsi_refused_oversized_roi_is_isolated_mechanics(tmp_path, refused_label, capfd):
    tifffile = pytest.importorskip("tifffile")

    # each 500x500 ROI is well above the 1 MB limit, so ram_limit=1 takes the out-of-core path
    tiles = [_tile(COMPUTED_PATCH), _tile(COMPUTED_PATCH)]
    tiles[refused_label - 1] = _tile(_refused_patch(65535))
    inten = np.hstack(tiles).astype(np.uint16)
    label = np.hstack([np.full(t.shape, i + 1, np.uint16) for i, t in enumerate(tiles)])

    intdir, segdir = tmp_path / "int", tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), label)

    def run(ram_limit):
        nyx = nyxus.Nyxus(["*ALL_GLCM*"], n_feature_calc_threads=1, ibsi=True)
        nyx.set_params(ram_limit=ram_limit)
        df = nyx.featurize_directory(str(intdir) + "/", str(segdir) + "/")
        return df.sort_values("ROI_label").reset_index(drop=True)

    in_ram = run(1000)
    capfd.readouterr()
    ooc = run(1)
    err = capfd.readouterr().err

    assert f"ROI {refused_label}" in err and "GLCM" in err, err
    computed_label = 3 - refused_label
    computed_ooc = _glcm_values(ooc[ooc["ROI_label"] == computed_label].iloc[0])
    computed_ram = _glcm_values(in_ram[in_ram["ROI_label"] == computed_label].iloc[0])
    assert (computed_ram != NOVAL).any()
    assert computed_ooc.to_numpy() == pytest.approx(computed_ram.to_numpy(), rel=1e-12, abs=1e-12)
    assert (_glcm_values(ooc[ooc["ROI_label"] == refused_label].iloc[0]) == NOVAL).all()


def test_2d_glcm_ibsi_8bit_rois_not_refused_mechanics(capfd):
    df = _featurize(*_image([COMPUTED_PATCH, _refused_patch(255)]))
    assert "GLCM" not in capfd.readouterr().err

    for label in (1, 2):
        values = _glcm_values(df[df["ROI_label"] == label].iloc[0])
        assert (values != NOVAL).any()
