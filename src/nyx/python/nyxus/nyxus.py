from .backend import initialize_environment, process_data
import os
import numpy as np
import pandas as pd
from typing import Optional, List


class Nyxus:
    """Nyxus image feature extraction library

    Scalably extracts features from images.

    Parameters
    ----------
    features : list[str]
        List of features to be calculated. Individual features can be
        provided or pre-specified feature groups. Valid groups include:
            *ALL*
            *ALL_INTENSITY*
            *ALL_MORPHOLOGY*
            *BASIC_MORPHOLOGY*
            *ALL_GLCM*
            *ALL_GLRM*
            *ALL_GLSZM*
            *ALL_GLDM*
            *ALL_NGTDM*
            *ALL_BUT_GABOR*
            *ALL_BUT_GLCM*
        Both individual features and feature groups are case sensitive.
    neighbor_distance: float (optional, default 5.0)
        Any two objects separated by a Euclidean distance (pixel units) greater than this
        value will not be be considered neighbors. This cutoff is used by all features which
        rely on neighbor identification.
    pixels_per_micron: float (optional, default 1.0)
        Specify the image resolution in terms of pixels per micron for unit conversion
        of non-unitless features.
    n_feature_calc_threads: int (optional, default 4)
        Number of threads to use for feature calculation parallelization purposes.
    n_loader_threads: int (optional, default 1)
        Number of threads to use for loading image tiles from disk. Note: image loading
        multithreading is very memory intensive. You should consider optimizing
        `n_feature_calc_threads` before increasing `n_loader_threads`.
    """

    def __init__(
        self,
        features: List[str],
        neighbor_distance: float = 5.0,
        pixels_per_micron: float = 1.0,
        n_feature_calc_threads: int = 4,
        n_loader_threads: int = 1,
    ):
        if neighbor_distance <= 0:
            raise ValueError("Neighbor distance must be greater than zero.")

        if pixels_per_micron <= 0:
            raise ValueError("Pixels per micron must be greater than zero.")

        if n_feature_calc_threads < 1:
            raise ValueError("There must be at least one feature calculation thread.")

        if n_loader_threads < 1:
            raise ValueError("There must be at least one loader thread.")

        initialize_environment(
            features,
            neighbor_distance,
            pixels_per_micron,
            n_feature_calc_threads,
            n_loader_threads,
        )

    def featurize(
        self,
        intensity_dir: str,
        label_dir: Optional[str] = None,
        file_pattern: Optional[str] = ".*",
    ):
        """Extract features from provided images.

        Extracts all the requested features _at the image level_ from the images
        present in `intensity_dir`. If `label_dir` is specified, features will be
        extracted for each unique label present in the label images. The file names
        of the label images are expected to match those of the intensity images.

        Parameters
        ----------
        intensity_dir : str
            Path to directory containing intensity images.
        label_dir : str (optional, default None)
            Path to directory containing label images.
        file_pattern: str (optional, default ".*")
            Regular expression used to filter the images present in both
            `intensity_dir` and `label_dir`

        Returns
        -------
        df : pd.DataFrame
            Pandas DataFrame containing the requested features with one row per label
            per image.
        """
        if not os.path.exists(intensity_dir):
            raise IOError(
                f"Provided intensity image directory '{intensity_dir}' does not exist."
            )

        if label_dir is not None and not os.path.exists(label_dir):
            raise IOError(
                f"Provided label image directory '{label_dir}' does not exist."
            )

        if label_dir is None:
            label_dir = intensity_dir

        header, string_data, numeric_data = process_data(
            intensity_dir, label_dir, file_pattern
        )

        df = pd.concat(
            [
                pd.DataFrame(string_data, columns=header[: string_data.shape[1]]),
                pd.DataFrame(numeric_data, columns=header[string_data.shape[1] :]),
            ],
            axis=1,
        )

        # Labels should always be uint.
        if "label" in df.columns:
            df["label"] = df.label.astype(np.uint32)

        return df
