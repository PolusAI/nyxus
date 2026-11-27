"""Mechanics tests for real-valued NIfTI volumes (FLOAT32 / FLOAT64) read end to end.

A real-valued NIfTI takes the quantization a real-valued TIFF or OME-Zarr takes: the volume's range
(or the ``min_intensity`` / ``max_intensity`` window) is spread over ``dynamic_range`` grey levels,
and the intensity family applies the recorded inverse, so MIN / MAX / MEAN come back in the volume's
own units to within one quantization step, ``(max - min) / dynamic_range``. A volume normalised to
[0, 1] keeps its fraction; narrowed to whole units instead it would read back as all zeros, and a
[0, 1000] volume would report its MEAN about half a unit low.

The volumes are random, written at test time with the header of ``ct3d_int16.nii`` (8x8x8) patched
to the float datatype, as tests/data/hounsfield/README.md describes for the committed fixtures, and
paired with ``mask3d.nii`` (all ones). The reference is numpy over the same values.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- these pin the loader's load-time map rather than
feature values against an independent reference, and establish no vetting.
"""
import os
import pathlib
import struct

import numpy as np
import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "hounsfield"
HEADER_SOURCE = DATA / "ct3d_int16.nii"
MASK = str(DATA / "mask3d.nii")

pytestmark = pytest.mark.skipif(
    not (HEADER_SOURCE.exists() and os.path.exists(MASK)),
    reason="NIfTI fixtures not present in tests/data/hounsfield",
)

FEATS = ["3MIN", "3MAX", "3MEAN"]
DYNAMIC_RANGE = 10000.0  # the Nyxus3D default

# NIfTI-1 datatype codes and the numpy dtype each stores
FLOAT_TYPES = {16: np.float32, 64: np.float64}


def _write_float_nifti(path, values, datatype):
    """Writes an 8x8x8 real-valued NIfTI with no header rescale."""
    h = bytearray(HEADER_SOURCE.read_bytes()[:352])  # 348 header + 4 extender
    dtype = FLOAT_TYPES[datatype]
    struct.pack_into("<h", h, 70, datatype)
    struct.pack_into("<h", h, 72, np.dtype(dtype).itemsize * 8)
    struct.pack_into("<f", h, 112, 0.0)  # scl_slope 0: no rescale
    struct.pack_into("<f", h, 116, 0.0)
    path.write_bytes(bytes(h) + np.ascontiguousarray(values, dtype="<" + np.dtype(dtype).char).tobytes())


@pytest.mark.parametrize("datatype", sorted(FLOAT_TYPES))
@pytest.mark.parametrize("hi", [1000.0, 1.0])
def test_3d_nifti_float_keeps_fraction_mechanics(tmp_path, datatype, hi):
    """MIN, MAX and MEAN of a random [0, hi) volume match numpy to within one quantization step.

    hi = 1 is the volume that read back as all zeros when narrowed to whole units; hi = 1000 is
    the one whose MEAN came out about 0.5 low.
    """
    rng = np.random.default_rng(0)
    v = rng.uniform(0.0, hi, 512).astype(FLOAT_TYPES[datatype])
    inten = tmp_path / "inten.nii"
    _write_float_nifti(inten, v, datatype)

    df = nyxus.Nyxus3D(FEATS).featurize_files([str(inten)], [MASK], False)
    f = {c: float(df[c].iloc[0]) for c in FEATS}

    ref = v.astype(np.float64)
    step = (ref.max() - ref.min()) / DYNAMIC_RANGE
    assert f["3MIN"] == pytest.approx(ref.min(), abs=1e-9 * hi)
    assert f["3MAX"] == pytest.approx(ref.max(), abs=step)
    assert f["3MEAN"] == pytest.approx(ref.mean(), abs=step)
    assert f["3MAX"] > f["3MIN"], "the volume collapsed onto one grey level"
