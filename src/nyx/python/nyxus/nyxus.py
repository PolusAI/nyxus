from .backend import initialize_environment, process_data
import os
import numpy as np
import pandas as pd
from typing import List, Optional


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
    """

    def __init__(self, features: List[str]):
        initialize_environment(features)

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
