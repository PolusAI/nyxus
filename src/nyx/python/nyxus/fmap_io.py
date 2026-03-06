"""Utilities for saving feature-map (fmaps) results to image formats.

Feature-map results from Nyxus/Nyxus3D are returned as a list of dicts,
one per parent ROI, each containing numpy arrays of spatial feature maps.
These utilities write those arrays to TIFF stacks or NIfTI volumes so
they can be consumed by downstream imaging and ML pipelines.
"""

import os
import numpy as np
from typing import List, Dict, Optional


def save_fmaps_to_tiff(
    fmaps: List[Dict],
    output_dir: str,
    prefix: str = "fmap",
    features: Optional[List[str]] = None,
):
    """Save 2D feature maps to multi-page TIFF files.

    For each parent ROI, creates one TIFF per feature. If the feature map
    is 3D (D, H, W), each z-slice becomes a page in the TIFF stack.

    Requires ``tifffile`` (``pip install tifffile``).

    Parameters
    ----------
    fmaps : list[dict]
        Feature-map results from ``Nyxus.featurize_directory()`` or similar,
        obtained with ``fmaps=True``.
    output_dir : str
        Directory to write TIFF files into. Created if it does not exist.
    prefix : str, optional
        Filename prefix. Default ``"fmap"``.
    features : list[str], optional
        Subset of feature names to save. Default saves all features.

    Returns
    -------
    list[str]
        Paths of all TIFF files written.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for TIFF export. Install it with: pip install tifffile"
        )

    os.makedirs(output_dir, exist_ok=True)
    written = []

    for roi_result in fmaps:
        roi_label = roi_result["parent_roi_label"]
        feat_dict = roi_result["features"]

        names = features if features else list(feat_dict.keys())

        for feat_name in names:
            if feat_name not in feat_dict:
                continue
            arr = feat_dict[feat_name]

            # Ensure float32 for compatibility
            arr = np.asarray(arr, dtype=np.float32)

            fname = f"{prefix}_roi{roi_label}_{feat_name}.tif"
            fpath = os.path.join(output_dir, fname)

            if arr.ndim == 2:
                tifffile.imwrite(fpath, arr)
            elif arr.ndim == 3:
                # (D, H, W) -> multi-page TIFF
                tifffile.imwrite(fpath, arr)
            else:
                raise ValueError(
                    f"Unexpected array dimensions {arr.ndim} for feature '{feat_name}'"
                )

            written.append(fpath)

    return written


def save_fmaps_to_nifti(
    fmaps: List[Dict],
    output_dir: str,
    prefix: str = "fmap",
    features: Optional[List[str]] = None,
    voxel_size: tuple = (1.0, 1.0, 1.0),
):
    """Save 3D feature maps to NIfTI-1 (.nii.gz) volumes.

    For each parent ROI, creates one NIfTI file per feature.
    The affine is set so the origin matches the ROI's global coordinates.

    Requires ``nibabel`` (``pip install nibabel``).

    Parameters
    ----------
    fmaps : list[dict]
        Feature-map results from ``Nyxus3D.featurize_directory()`` or similar,
        obtained with ``fmaps=True``.
    output_dir : str
        Directory to write NIfTI files into. Created if it does not exist.
    prefix : str, optional
        Filename prefix. Default ``"fmap"``.
    features : list[str], optional
        Subset of feature names to save. Default saves all features.
    voxel_size : tuple of float, optional
        Voxel dimensions (x, y, z) in physical units. Default ``(1.0, 1.0, 1.0)``.

    Returns
    -------
    list[str]
        Paths of all NIfTI files written.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI export. Install it with: pip install nibabel"
        )

    os.makedirs(output_dir, exist_ok=True)
    written = []

    for roi_result in fmaps:
        roi_label = roi_result["parent_roi_label"]
        feat_dict = roi_result["features"]
        origin_x = roi_result.get("origin_x", 0)
        origin_y = roi_result.get("origin_y", 0)
        origin_z = roi_result.get("origin_z", 0)

        names = features if features else list(feat_dict.keys())

        for feat_name in names:
            if feat_name not in feat_dict:
                continue
            arr = feat_dict[feat_name]
            arr = np.asarray(arr, dtype=np.float32)

            # For 2D maps, add a singleton z-dimension
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]

            if arr.ndim != 3:
                raise ValueError(
                    f"Unexpected array dimensions {arr.ndim} for feature '{feat_name}'"
                )

            # NIfTI expects (x, y, z) axis order; our arrays are (D, H, W)
            # Transpose to (W, H, D) = (x, y, z) for NIfTI convention
            arr_nifti = np.transpose(arr, (2, 1, 0))

            # Build affine: diagonal scaling + translation to ROI origin
            affine = np.eye(4)
            affine[0, 0] = voxel_size[0]
            affine[1, 1] = voxel_size[1]
            affine[2, 2] = voxel_size[2]
            affine[0, 3] = origin_x * voxel_size[0]
            affine[1, 3] = origin_y * voxel_size[1]
            affine[2, 3] = origin_z * voxel_size[2]

            img = nib.Nifti1Image(arr_nifti, affine)

            fname = f"{prefix}_roi{roi_label}_{feat_name}.nii.gz"
            fpath = os.path.join(output_dir, fname)
            nib.save(img, fpath)

            written.append(fpath)

    return written
