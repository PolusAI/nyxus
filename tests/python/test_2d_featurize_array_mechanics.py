"""Mechanics: what Nyxus.featurize and ImageQuality.featurize do to the 2D image arrays they are
given, and how their Arrow/Parquet branch reports a backend error.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- these assert argument handling and error
plumbing, not a feature value against a reference.

Nyxus.featurize converts intensities to uint32, shifting an array whose minimum is negative so
that the minimum becomes 0. The shifted values go into a new array: the caller's array keeps its
bytes, and a call on the array gives the same features as a call on a copy of it. A single 2D
image is wrapped into a fresh stack before the shift, so only a stack of images can reach the
caller's memory; the cases below use stacks for that reason and a single image for completeness.

With output_type "arrowipc" or "parquet" the binding returns a one-element tuple holding the
error text, and a non-empty text is raised as RuntimeError carrying it. The binding raises on
every error it detects, so that branch is reached here through a stand-in binding.
"""
import importlib

import numpy as np
import pytest

import nyxus

nyxus_module = importlib.import_module("nyxus.nyxus")

FEATURES = ["MIN", "MAX", "MEAN"]


def square_roi_labels(shape):
    """-> a uint32 label array of `shape` with one square ROI per image."""
    labels = np.zeros(shape, dtype=np.uint32)
    labels[..., 2:12, 3:13] = 1
    return labels


def negative_stack(dtype):
    """-> a 2 x 16 x 16 stack of `dtype` with negative values; a float stack holds fractions too."""
    stack = (np.arange(2 * 16 * 16).reshape(2, 16, 16) % 23 - 7).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        stack += 0.5
    return stack


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float64])
def test_2d_featurize_array_negative_stack_left_unchanged_mechanics(dtype):
    intens = negative_stack(dtype)
    before = intens.copy()
    labels = square_roi_labels(intens.shape)

    df = nyxus.Nyxus(FEATURES).featurize(intens, labels)

    assert intens.dtype == before.dtype
    assert intens.tobytes() == before.tobytes()

    reference = nyxus.Nyxus(FEATURES).featurize(before.copy(), labels)
    assert df[FEATURES].values.tolist() == reference[FEATURES].values.tolist()
    # the shifted grey levels start at 0 in the image holding the stack's minimum
    assert df["MIN"].min() == 0.0


def test_2d_featurize_array_negative_image_left_unchanged_mechanics():
    intens = negative_stack(np.int32)[0]
    before = intens.copy()

    nyxus.Nyxus(FEATURES).featurize(intens, square_roi_labels(intens.shape))

    assert intens.tobytes() == before.tobytes()


def test_2d_featurize_array_uint32_passed_through_mechanics(monkeypatch):
    # A non-negative uint32 stack needs no conversion, so the binding receives the caller's
    # array itself rather than a copy of it.
    intens = np.abs(negative_stack(np.int32)).astype(np.uint32)
    labels = square_roi_labels(intens.shape)
    received = []

    def binding(instid, I, M, *rest):
        received.append(I)
        return ("",)

    monkeypatch.setattr(nyxus_module, "featurize_montage_imp", binding)
    monkeypatch.setattr(nyxus_module, "get_arrow_file_imp", lambda instid: "out.arrow")
    nyxus.Nyxus(FEATURES).featurize(intens, labels, output_type="arrowipc")

    assert received[0] is intens


@pytest.mark.parametrize("output_type", ["arrowipc", "parquet"])
@pytest.mark.parametrize("cls", [nyxus.Nyxus, nyxus.ImageQuality])
def test_2d_featurize_array_arrow_error_raised_mechanics(monkeypatch, cls, output_type):
    intens = np.abs(negative_stack(np.int32)).astype(np.uint32)
    labels = square_roi_labels(intens.shape)

    monkeypatch.setattr(nyxus_module, "featurize_montage_imp",
                        lambda *args: ("backend refused the output file",))
    nyx = cls(["FOCUS_SCORE"] if cls is nyxus.ImageQuality else FEATURES)

    with pytest.raises(RuntimeError, match="Error calculating features: backend refused the output file"):
        nyx.featurize(intens, labels, output_type=output_type)
    assert nyx.error_message == "backend refused the output file"
