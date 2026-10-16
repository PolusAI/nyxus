from .backend import (
    initialize_environment,
    release_environment_imp,
    featurize_directory_imp,
    featurize_directory_3D_imp,
    featurize_montage_imp,
    featurize_fname_lists_imp,
    featurize_fname_lists_3D_imp,
    findrelations_imp,
    use_gpu,
    gpu_available_imp,
    blacklist_roi_imp,
    clear_roi_blacklist_imp,
    roi_blacklist_get_summary_imp,
    customize_gabor_feature_imp,
    set_if_ibsi_imp,
    set_environment_params_imp,
    get_params_imp,
    arrow_is_enabled_imp,
    get_arrow_file_imp, 
    get_parquet_file_imp,
    set_metaparam_imp,
    get_metaparam_imp)

import os
import numpy as np
import pandas as pd
from typing import Optional, List

class Nyxus:
    r"""Extract 2D features from images and their ROI masks.

    Each instance holds its own settings. Features are computed for every
    labelled ROI of each intensity/mask image pair, from numpy arrays
    (:meth:`featurize`), a directory (:meth:`featurize_directory`) or lists of
    files (:meth:`featurize_files`).

    Parameters
    ----------
    features : list of str
        Feature names and feature groups to compute. 2D groups:
        ``*ALL*``, ``*WHOLESLIDE*``, ``*ALL_INTENSITY*``, ``*ALL_MORPHOLOGY*``,
        ``*BASIC_MORPHOLOGY*``, ``*ALL_GLCM*``, ``*ALL_GLRLM*``, ``*ALL_GLDZM*``,
        ``*ALL_GLSZM*``, ``*ALL_GLDM*``, ``*ALL_NGLDM*``, ``*ALL_NGTDM*``,
        ``*ALL_NEIGHBOR*``, ``*ALL_IH*``, ``*GEOMOMS*``, ``*IGEOMOMS*`` and
        ``*SGEOMOMS*``.
        ``*ALL_IH*`` (the Intensity Histogram family) is computed only when
        `ibsi` is True; without it the IH features are dropped silently.
        Names are case-sensitive and must be upper case: ``"MEAN"`` is accepted,
        ``"mean"`` raises ValueError. (The command-line tool upper-cases its
        ``--features`` list; the Python API does not.) Entries are applied left to
        right; a leading ``-`` removes a feature or group enabled by an earlier
        entry (except ``-COV`` and ``-*ALL*``, which add) and a leading ``+``
        adds it, e.g. ``["*ALL_INTENSITY*", "-MEAN"]``. An empty list computes
        no features (the command line instead falls back to ``*ALL*``).
    neighbor_distance : int, optional
        Default 5. Two ROIs farther apart than this Euclidean distance, in
        pixels, are not neighbors. Used by every neighbor-based feature.
        Must be positive.
    pixels_per_micron : float, optional
        Default 1.0. Must be positive. Stored and reported by
        :meth:`get_params`; no feature currently depends on it
        (``AREA_UM2`` is reported as 0, a known defect,
        `#496 <https://github.com/PolusAI/nyxus/issues/496>`__).
    coarse_gray_depth : int, optional
        Default 64. Must be positive. Number of grey levels intensities are
        binned to for the texture features, and the number of bins of the
        ``ENTROPY``, ``UNIFORMITY``, ``HISTOGRAM`` and ``IH_*`` histograms.
        With ``ibsi=True`` the texture features are not binned and read the
        image's own grey levels; the histograms still follow this depth.
    n_feature_calc_threads : int, optional
        Default 4. Number of feature calculation threads. Must be at least 1.
    use_gpu_device : int, optional
        Default -1 (CPU). Any non-negative value turns the GPU on for erosion
        and the 2D moments (leave it off when computing ``GABOR`` and when
        passing arrays to :meth:`featurize`, where it can end the process); the
        computation runs on CUDA device 0 whatever the value (see
        :func:`get_gpu_properties` and the GPU page). A
        non-negative id raises ValueError when no GPU is available; a value
        below -1 raises RuntimeError when this build of Nyxus has no GPU
        support.
    ibsi : bool, optional
        Default False. Compute the IBSI-compliant variant of the features that
        have one, and enable the Intensity Histogram (``IH_*``) family. The
        texture features then skip grey-level binning, whatever
        ``coarse_gray_depth`` is.
    mergerois : bool, optional
        Default False. Merge all nonzero mask labels into one whole-foreground
        ROI (background excluded) instead of one ROI per label.
    gabor_kersize : int, optional
        Default 16. Side of the Gabor filter kernel, in pixels; at least 2
        (1 is accepted, but its values are arbitrary).
    gabor_gamma : float, optional
        Default 0.1. Aspect ratio of the Gaussian factor.
    gabor_sig2lam : float, optional
        Default 0.8. Spatial frequency bandwidth (sigma to lambda ratio).
    gabor_f0 : float, optional
        Default 0.1. Angular frequency of the baseline low-pass filter, in
        radians per pixel; its wavelength is :math:`2\pi/f_0` pixels.
    gabor_thold : float, optional
        Default 0.025. Lower threshold of the filtered-image to baseline
        ratio.
    gabor_thetas : list of float, optional
        Default [0, 45, 90, 135]. Orientations of the Gaussian, in degrees,
        conventionally 0-180 (not enforced), one per entry of `gabor_freqs`.
    gabor_freqs : list of float, optional
        Default [4, 16, 32, 64]. Angular frequencies of the filters' harmonic
        factor, in radians per pixel (wavelength :math:`2\pi/f` pixels), one
        per entry of `gabor_thetas`.
    dynamic_range : float, optional
        Default 10000. Number of grey levels a floating-point image is
        rescaled to.
    min_intensity : float, optional
        Default 0.0. Intensity of a floating-point image mapped to the lowest
        grey level.
    max_intensity : float, optional
        Default 1.0. Intensity of a floating-point image mapped to
        `dynamic_range`.
    preserve_hu : bool, optional
        Default False. Not needed for CT: Hounsfield units are preserved and
        reported either way, despite the option's name. What it does: carries
        a floating-point slide on the offset-preserving map (1 grey level = 1
        intensity unit, offset by the slide's floored minimum) instead of the
        min-max rescaling controlled by `dynamic_range`, `min_intensity` and
        `max_intensity`. A DICOM or integer NIfTI slide takes it instead of
        the stored map, which keeps the stored integers as grey levels and
        carries RescaleSlope/RescaleIntercept (scl_slope/scl_inter) in the
        recorded inverse, so a fractional slope loses no stored level. Other
        integer slides take the offset map regardless of this flag. The DICOM
        and NIfTI rescale reaches the reported values either way, so CT
        Hounsfield units survive the load and negative values never wrap.

        Whichever map a slide takes is recorded and undone on the way out:

        * Every intensity feature -- location (mean, median, mode,
          percentiles, min, max), dispersion, the sum/energy family, and the
          Intensity Histogram (IH_*) features -- is reported in the slide's
          own intensity domain. A CT read in Hounsfield units is reported in
          Hounsfield units, negatives included.
        * Shift-invariant intensity features (variance, standard deviation,
          skewness, kurtosis, range, interquartile range) and all
          shape/texture features are unaffected.
        * Sub-unit precision survives on the stored map, which reports a
          fractional rescaled value exactly. Elsewhere grey levels are
          integers, so a value that is not integer-valued is reported as the
          grey level it was stored as: truncated, or rounded to nearest under
          `preserve_hu`. Hounsfield units are integer-valued, so CT is
          unaffected.

        Applies to images read from files; arrays passed to ``featurize`` do
        not go through the file loaders.
    ram_limit : int, optional
        Memory budget in megabytes. A ROI read from files with a mask whose
        estimated footprint reaches it is processed out of core. Arrays passed
        to :meth:`featurize` are never processed out of core (the call raises
        RuntimeError), and a whole-slide image that reaches it is refused.
        Default -1 keeps the built-in limit, half
        of the physical memory available when the instance is created. A
        value larger than the available memory is rejected with a message on
        standard error and the previous limit is kept.
    verbose : int, optional
        Default 0 (silent). Level of diagnostic output. Must be non-negative.
    anisotropy_x : float, optional
        Default 1.0. Size of a pixel along X relative to Y. Must be positive.
        The geometric features are computed on the pixels resampled by the
        anisotropy; intensity and texture features on the pixels as acquired,
        so it does not change them. The intensity-weighted geometric features
        -- the EDGE_* statistics, WEIGHTED_CENTROID_X/Y, MASS_DISPLACEMENT, the
        intensity moments IMOM_*, the radial distribution and ZERNIKE2D -- are
        computed on the pixels as acquired too, each pixel at its position
        scaled by the anisotropy and weighted by its intensity times its area,
        with distances to the contour and the Zernike unit disk in the same
        units. Applies to ``featurize_directory`` and ``featurize_files``;
        ``featurize`` on in-memory arrays does not apply anisotropy.
    anisotropy_y : float, optional
        Default 1.0. Y-dimension counterpart of `anisotropy_x`. Must be
        positive.

    Other Parameters
    ----------------
    use_physical_spacing : bool, optional
        Default False. Accepted for parity with :class:`Nyxus3D`; it only
        affects 3D volumes.
    channel_signature, parent_channel, child_channel, aggregate
        Accepted without a warning and ignored. Use :class:`Nested` for
        parent-child analysis.

    Raises
    ------
    ValueError
        If a feature name or group is not recognized, or a numeric parameter
        is out of range, or `use_gpu_device` is non-negative and no GPU is
        available.
    RuntimeError
        If `use_gpu_device` is below -1 and this build has no GPU support.

    Notes
    -----
    Unknown keyword arguments print a warning and are ignored.

    The enabled feature set and the per-feature settings derived from
    `coarse_gray_depth` are fixed when the instance is constructed. Changing
    `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
    :meth:`get_params` reports but not the features computed by this
    instance; construct a new instance to change them.

    Gabor parameters are process-wide, not per instance: every constructor
    call that accepts ``gabor_*`` arguments sets them (to the defaults when
    omitted), and :meth:`set_gabor_feature_params` changes them for all
    instances.

    Examples
    --------
    >>> import numpy as np
    >>> from nyxus import Nyxus
    >>> intens = np.zeros((32, 32), dtype=np.uint16)
    >>> labels = np.zeros((32, 32), dtype=np.uint16)
    >>> intens[4:12, 4:12] = 100
    >>> labels[4:12, 4:12] = 1
    >>> intens[16:28, 16:28] = 200
    >>> labels[16:28, 16:28] = 2
    >>> nyx = Nyxus(["MEAN", "AREA_PIXELS_COUNT"])
    >>> df = nyx.featurize(intens, labels)
    >>> df[["ROI_label", "MEAN", "AREA_PIXELS_COUNT"]].values.tolist()
    [[1.0, 100.0, 64.0], [2.0, 200.0, 144.0]]

    Every keyword with its default value:

    >>> nyx = Nyxus(
    ...     ["*ALL_INTENSITY*", "*ALL_GLCM*"],
    ...     neighbor_distance=5,
    ...     pixels_per_micron=1.0,
    ...     coarse_gray_depth=64,
    ...     n_feature_calc_threads=4,
    ...     use_gpu_device=-1,
    ...     ibsi=False,
    ...     mergerois=False,
    ...     gabor_kersize=16,
    ...     gabor_gamma=0.1,
    ...     gabor_sig2lam=0.8,
    ...     gabor_f0=0.1,
    ...     gabor_thold=0.025,
    ...     gabor_thetas=[0, 45, 90, 135],
    ...     gabor_freqs=[4, 16, 32, 64],
    ...     dynamic_range=10000,
    ...     min_intensity=0.0,
    ...     max_intensity=1.0,
    ...     preserve_hu=False,
    ...     ram_limit=-1,
    ...     verbose=0,
    ...     anisotropy_x=1.0,
    ...     anisotropy_y=1.0,
    ... )
    >>> nyx.get_params("coarse_gray_depth")
    {'coarse_gray_depth': 64}

    Feature names are case-sensitive:

    >>> Nyxus(["mean"])
    Traceback (most recent call last):
        ...
    ValueError: Error: 'mean' is not a valid 2D feature name
    """

    def __del__(self):
        # The backend keys this instance's settings on id(self), which is the object's
        # address -- and CPython hands that same address to the next object it allocates.
        # An entry left behind after this object dies is inherited wholesale by that next
        # instance, which would then silently start with our ram_limit, feature list, gpu
        # flag or ibsi mode. Drop it here rather than leave the address dirty.
        try:
            release_environment_imp(id(self))
        except Exception:
            pass    # interpreter shutdown may already have torn the backend module down

    def __init__(
        self,
        features: List[str],
        **kwargs
        ):
        valid_keys = {
            'neighbor_distance', 'pixels_per_micron', 'coarse_gray_depth',
            'n_feature_calc_threads', 'use_gpu_device', 'ibsi',
            'gabor_kersize', 'gabor_gamma', 'gabor_sig2lam', 'gabor_f0',
            'gabor_thold', 'gabor_thetas', 'gabor_freqs', 'channel_signature', 
            'parent_channel', 'child_channel', 'aggregate', 'dynamic_range', 'min_intensity',
            'max_intensity', 'ram_limit', 'verbose',
            'anisotropy_x', 'anisotropy_y', 'mergerois', 'preserve_hu', 'use_physical_spacing'
        }

        # Check for unexpected keyword arguments
        invalid_keys = set(kwargs.keys()) - valid_keys
        if invalid_keys:
            print(f"Warning: unexpected keyword argument(s): {', '.join(invalid_keys)}")
        
        # parse kwargs
        features = features
        neighbor_distance = kwargs.get('neighbor_distance', 5)
        pixels_per_micron = kwargs.get('pixels_per_micron', 1.0)
        coarse_gray_depth = kwargs.get('coarse_gray_depth', 64)
        n_feature_calc_threads = kwargs.get('n_feature_calc_threads', 4)
        use_gpu_device = kwargs.get('use_gpu_device', -1)
        ibsi = kwargs.get('ibsi', False)
        gabor_kersize = kwargs.get('gabor_kersize', 16)
        gabor_gamma = kwargs.get('gabor_gamma', 0.1)
        gabor_sig2lam = kwargs.get('gabor_sig2lam', 0.8)
        gabor_f0 = kwargs.get('gabor_f0', 0.1)
        gabor_thold = kwargs.get('gabor_thold', 0.025)
        gabor_thetas = kwargs.get('gabor_thetas', [0, 45, 90, 135])
        gabor_freqs = kwargs.get('gabor_freqs', [4, 16, 32, 64])
        dynamic_range = kwargs.get('dynamic_range', 10000)
        min_intensity = kwargs.get('min_intensity', 0.0)
        max_intensity = kwargs.get('max_intensity', 1.0)
        preserve_hu = kwargs.get('preserve_hu', False)   # CT/HU: preserve absolute Hounsfield values
        use_physical_spacing = kwargs.get('use_physical_spacing', False)   # opt-in OME PhysicalSize* voxel spacing (3D)
        ram_limit = kwargs.get('ram_limit', -1)
        verb_lvl = kwargs.get('verbose', 0)
        aniso_x = kwargs.get('anisotropy_x', 1.0)
        aniso_y = kwargs.get('anisotropy_y', 1.0)
        
        if neighbor_distance <= 0:
            raise ValueError("Neighbor distance must be greater than zero.")

        if pixels_per_micron <= 0:
            raise ValueError("Pixels per micron must be greater than zero.")

        if coarse_gray_depth <= 0:
            raise ValueError("Custom number of grayscale levels (parameter coarse_gray_depth, default=64) must be positive.")

        if n_feature_calc_threads < 1:
            raise ValueError("There must be at least one feature calculation thread.")

        if use_gpu_device > -1 and not gpu_available_imp(id(self)):
            raise ValueError ("No need to set GPU device ID because GPU is unavailable")

        if verb_lvl < 0:
            raise ValueError ("verbosity must be non-negative")

        if aniso_x <= 0:
            raise ValueError ("anisotropy_x must be positive")

        if aniso_y <= 0:
            raise ValueError ("anisotropy_y must be positive")

        aniso_z = 1.0   # not used in 2D
        mergerois = kwargs.get('mergerois', False)

        initialize_environment(
            id(self),
            2, # 2D
            features,
            neighbor_distance,
            pixels_per_micron,
            coarse_gray_depth,
            n_feature_calc_threads,
            use_gpu_device,
            ibsi,
            dynamic_range,
            min_intensity,
            max_intensity,
            False,
            ram_limit,
            verb_lvl,
            aniso_x,
            aniso_y,
            aniso_z,
            mergerois,
            preserve_hu,
            use_physical_spacing)

        self.set_gabor_feature_params(
            kersize = gabor_kersize,
            gamma = gabor_gamma,
            sig2lam = gabor_sig2lam,
            f0 = gabor_f0,
            thold = gabor_thold,
            thetas = gabor_thetas,
            freqs = gabor_freqs
        )
        
        # list of valid outputs that are used throughout featurize functions
        self._valid_output_types = ['pandas', 'arrowipc', 'parquet']


    def set_metaparam(
        self, 
        paramval: str):
        r"""Set a feature-specific parameter.

        Parameters
        ----------
        paramval : str
            Assignment of the form ``"<feature>/<parameter>=<value>"``, for
            example ``"glcm/greydepth=32"``.
            Recognized names: ``glcm/greydepth``, ``glcm/offset``,
            ``3glcm/greydepth``, ``3glcm/offset``, ``3glcm/numang``,
            ``3glcm/sparseintensities``, ``3gldm/greydepth``, ``3ngtdm/greydepth``,
            ``3ngtdm/radius``, ``3glrlm/greydepth`` and ``3glszm/greydepth``. The
            value is an integer, except ``3glcm/sparseintensities`` which takes
            ``true`` or ``false``.

        Raises
        ------
        ValueError
            If the string is malformed, the feature or parameter is not
            recognized, or the value cannot be parsed.

        Notes
        -----
        A name without a ``/`` (e.g. ``"greydepth=8"``) is accepted and has no
        effect. ``glcm/greydepth`` does not change the 2D co-occurrence
        matrix, which is binned with `coarse_gray_depth`; it sets only the grey
        depth of the constant-ROI test, so it decides whether a ROI whose
        minimum and maximum fall into one bin returns the missing-value
        substitute.

        See Also
        --------
        get_metaparam

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["*ALL_GLCM*"])
        >>> nyx.set_metaparam("glcm/greydepth=32")
        >>> nyx.get_metaparam("glcm/greydepth")
        32.0
        """
        ok,errordetails = set_metaparam_imp (id(self), paramval)
        if ok==False:
            raise  ValueError (f'Invalid metaparameter value {paramval}: {errordetails}')

    def get_metaparam (
        self, 
        paramname: str):
        r"""Get a feature-specific parameter.

        Parameters
        ----------
        paramname : str
            Name of the form ``"<feature>/<parameter>"``, for example
            ``"glcm/greydepth"``. See :meth:`set_metaparam` for the names.

        Returns
        -------
        float
            Current value.

        Raises
        ------
        NameError
            If the feature or parameter is not recognized.

        Notes
        -----
        The ``3glcm/*`` names read the 2D ``glcm`` values, not the ones set
        through ``3glcm/*``, a known defect
        (`#513 <https://github.com/PolusAI/nyxus/issues/513>`__). ``3glszm/greydepth`` is read under the name
        ``3glsz/greydepth``. A name without a ``/`` returns an undefined value.

        See Also
        --------
        set_metaparam

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["*ALL_GLCM*"])
        >>> nyx.get_metaparam("glcm/greydepth")
        64.0
        """
        val,errordetails = get_metaparam_imp (id(self), paramname)
        if len(errordetails) > 0:
            raise  NameError (f'Invalid metaparameter name {paramname}: {errordetails}')
        return val

    def featurize_directory(
        self,
        intensity_dir: str,
        label_dir: Optional[str] = None,
        file_pattern: Optional[str] = ".*",
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
    ):
        r"""Extract features from the images in a directory.

        Pairs each intensity image in `intensity_dir` with the mask image of the
        same file name in `label_dir` and computes the features of every ROI.
        Without `label_dir` each intensity image is processed as a single ROI.

        Parameters
        ----------
        intensity_dir : str
            Directory of intensity images.
        label_dir : str, optional
            Directory of mask (label) images. Default None processes each
            intensity image as one ROI covering the whole image.
        file_pattern : str, optional
            Default ".*". Regular expression selecting the files. It must match
            the whole file name, case-sensitively, in both directories.
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        OSError
            If `intensity_dir` or `label_dir` does not exist.
        ValueError
            If `output_type` is not recognized or `file_pattern` is not a valid
            pattern.
        RuntimeError
            If the dataset cannot be read.

        Notes
        -----
        ROIs on the ROI blacklist (:meth:`blacklist_roi`) are skipped.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["*ALL_INTENSITY*"])
        >>> df = nyx.featurize_directory("int", "seg")
        >>> df.shape
        (139, 45)
        >>> df = nyx.featurize_directory("int", "seg", file_pattern=r".*_c0\.ome\.tif")
        >>> df.shape
        (66, 45)
        >>> nyx.featurize_directory("int", "seg", output_type="arrowipc",
        ...                         output_path="out/features.arrow")
        'out/features.arrow'
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

        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}.')
            
        if (output_type == 'pandas'):
            
            header, string_data, numeric_data = featurize_directory_imp (id(self), intensity_dir, label_dir, file_pattern, output_type, "")

            df = pd.concat(
                [
                    pd.DataFrame(string_data, columns=header[: string_data.shape[1]]),
                    pd.DataFrame(numeric_data, columns=header[string_data.shape[1] :]),
                ],
                axis=1,
            )

            # Labels should always be uint
            if "ROI_label" in df.columns:
                df.ROI_label = df.ROI_label.astype(np.uint32)

            return df
        
        else:
            
            featurize_directory_imp (id(self), intensity_dir, label_dir, file_pattern, output_type, output_path)
            
            return get_arrow_file_imp (id(self)) # return path to file


    def featurize(
        self,
        intensity_images: np.ndarray,
        label_images: np.ndarray,
        intensity_names: list = [],
        label_names: list = [],
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
        
    ):
        r"""Extract features from images held in numpy arrays.

        Computes the features of every labelled ROI of one image pair (2D arrays)
        or of a stack of image pairs (3D arrays, first axis = image).

        Parameters
        ----------
        intensity_images : numpy.ndarray
            One 2D intensity image, or a 3D stack of them.
        label_images : numpy.ndarray
            Mask (label) images matching `intensity_images` in shape; 0 is
            background.
        intensity_names : list of str, optional
            Names reported in the ``intensity_image`` column, one per image.
            Default ``Intensity0``, ``Intensity1``, ...
        label_names : list of str, optional
            Names reported in the ``mask_image`` column, one per image. Default
            ``Segmentation0``, ``Segmentation1``, ...
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        ValueError
            If an argument is not a numpy array, the arrays are not both 2D or
            both 3D, their shapes differ, the number of names does not match the
            number of images, or `output_type` is not recognized.
        RuntimeError
            If a ROI's estimated footprint reaches `ram_limit`: arrays are not
            processed out of core.

        Notes
        -----
        Intensities are converted to ``uint32`` before featurization: an array
        whose minimum is negative is first shifted so the minimum becomes 0, and
        fractional values are truncated (1.7 becomes 1). A 3D array is shifted in
        place, so the caller's array changes. Labels are converted to ``uint32``.
        The file-loader options (`dynamic_range`, `min_intensity`,
        `max_intensity`, `preserve_hu`) do not apply to arrays, and the ROI
        blacklist is not applied.

        Examples
        --------
        >>> import numpy as np
        >>> from nyxus import Nyxus
        >>> intens = np.zeros((2, 16, 16), dtype=np.uint16)
        >>> labels = np.zeros((2, 16, 16), dtype=np.uint16)
        >>> intens[:, 2:8, 2:8] = [[[10]], [[20]]]
        >>> labels[:, 2:8, 2:8] = 1
        >>> nyx = Nyxus(["MEAN"])
        >>> df = nyx.featurize(intens, labels, intensity_names=["a", "b"],
        ...                    label_names=["a_mask", "b_mask"])
        >>> df[["intensity_image", "mask_image", "MEAN"]].values.tolist()
        [['a', 'a_mask', 10.0], ['b', 'b_mask', 20.0]]
        """
        if (output_type != "" and output_type not in self._valid_output_types):
            raise ValueError(f'Invalid output type: {output_type}. Valid options are: {self._valid_output_types}')
        
        # verify argument types
        if not isinstance(intensity_images, np.ndarray):
            raise ValueError("intensity_images parameter must be numpy.ndarray")

        if not isinstance(label_images, np.ndarray):
            raise ValueError("label_images parameter must be numpy.ndarray")
        
        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}.')
        
        # verify dimensions of images are the same
        if(intensity_images.ndim == 2):
            if(label_images.ndim != 2):
                raise ValueError("Both intensity and label arrays must be the same dimension")
            
            intensity_images = np.array([intensity_images])
            label_images = np.array([label_images])
            
        elif(intensity_images.ndim == 3):
            if(label_images.ndim != 3):
                raise ValueError("Both intensity and label arrays must be the same dimension")
            
        else:
            raise ValueError("Intensity and label arrays must be 2D or 3D")
            
        
        if(intensity_images.shape != label_images.shape):
            raise ValueError("Intensity and label image arrays must have the same number of images with matching dimensions")
        
        if (intensity_names == []):
            int_name = "Intensity"
            
            # Rebind to a fresh list (do NOT mutate the shared default arg)
            intensity_names = [int_name + str(i) for i in range(intensity_images.shape[0])]
       
        if (label_names == []):
            seg_name = "Segmentation"
            
            # Rebind to a fresh list (do NOT mutate the shared default arg)
            label_names = [seg_name + str(i) for i in range(label_images.shape[0])]
                
        if (intensity_images.shape[0] != len(intensity_names)):
            raise ValueError ("Number of _intensity image names_ (" + str(len(intensity_names)) + ") must be the same as the number of intensity images (" + str(intensity_images.shape[0]) + ")")
        
        if (label_images.shape[0] != len(label_names)):
            raise ValueError("Number of segmentation names must be the same as the number of images.")
        
        # (1) check if the intensities are represented in Hounsfeld type scale, and (2) adjust them into the unsigned integer scale
        I = intensity_images
        min_raw_I = np.min (intensity_images)
        if (min_raw_I < 0):
            I -= min_raw_I
        if I.dtype != np.uint32:
            I = I.astype(np.uint32)

        # cast mask data to unsigned integer, too
        M = label_images.astype (np.uint32)

        # featurize
        if (output_type == 'pandas'):
            header, string_data, numeric_data, error_message = featurize_montage_imp (id(self), I, M, intensity_names, label_names, output_type, "")
            self.error_message = error_message
            if error_message != '':
                print (error_message)

            df = pd.concat(
                [
                    pd.DataFrame(string_data, columns=header[: string_data.shape[1]]),
                    pd.DataFrame(numeric_data, columns=header[string_data.shape[1] :]),
                ],
                axis=1,
            )

            # labels should always be uint
            if "ROI_label" in df.columns:
                df.ROI_label = df.ROI_label.astype(np.uint32)

            return df
            
        else:
            ret = featurize_montage_imp (id(self), I, M, intensity_names, label_names, output_type, output_path)
            self.error_message = ret[0]
            if(self.error_message != ''):
                raise RuntimeError('Error calculating features: ' + error_message[0])
            
            return get_arrow_file_imp (id(self)) # return path to file
                
    
    def use_gpu_device (self, gpu_device_id: int):
        r"""Switch GPU computation on or off for this instance.

        Parameters
        ----------
        gpu_device_id : int
            Interpreted as a flag: nonzero switches GPU computation on, 0
            switches it off. It does not select a device: from Python the
            computation runs on CUDA device 0.

        Raises
        ------
        RuntimeError
            If this build of Nyxus has no GPU support.

        Examples
        --------
        >>> import nyxus
        >>> nyx = nyxus.Nyxus(["MEAN"])
        >>> if nyxus.gpu_is_available(id(nyx)):
        ...     nyx.use_gpu_device(1)
        """
        use_gpu (id(self), gpu_device_id)

    def featurize_files (
        self,
        intensity_files: list,
        mask_files: list,
        single_roi: bool,
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
    ):
        r"""Extract features from lists of image files.

        Parameters
        ----------
        intensity_files : list of str
            Paths of intensity images.
        mask_files : list of str
            Paths of mask (label) images, paired with `intensity_files` by
            position. May be None when `single_roi` is True.
        single_roi : bool
            True processes each intensity image as one ROI covering the whole
            image and ignores `mask_files`; False pairs `intensity_files` with
            `mask_files`.
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        OSError
            If `intensity_files` is None, or `mask_files` is None while
            `single_roi` is False.
        ValueError
            If `output_type` is not recognized.
        RuntimeError
            If `intensity_files` is empty, a file does not exist, the two
            lists differ in length while `single_roi` is False, or
            featurization fails.

        Notes
        -----
        ROIs on the ROI blacklist (:meth:`blacklist_roi`) are skipped.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["*ALL_INTENSITY*"])
        >>> df = nyx.featurize_files(["int/p0_y1_r1_c0.ome.tif"],
        ...                          ["seg/p0_y1_r1_c0.ome.tif"], False)
        >>> df.shape
        (66, 45)
        >>> nyx.featurize_files(["int/p0_y1_r1_c0.ome.tif"], None, True).shape
        (1, 45)
        """

        if intensity_files is None:
            raise IOError ("The list of intensity file paths is empty")

        if mask_files is None and not single_roi:
            raise IOError ("The list of segment file paths is empty. Supply mask images or set single_roi to True")

        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}')

        mask_files_arg = mask_files if mask_files is not None else []

        if (output_type == 'pandas'):
            
            header, string_data, numeric_data = featurize_fname_lists_imp (id(self), intensity_files, mask_files_arg, single_roi, output_type, "")

            df = pd.concat(
                [
                    pd.DataFrame(string_data, columns=header[: string_data.shape[1]]),
                    pd.DataFrame(numeric_data, columns=header[string_data.shape[1] :]),
                ],
                axis=1,
            )

            # Labels should always be uint
            if "ROI_label" in df.columns:
                df.ROI_label = df.ROI_label.astype(np.uint32)

            return df

        else:
            
            featurize_fname_lists_imp (id(self), intensity_files, mask_files_arg, single_roi, output_type, output_path)
            return get_arrow_file_imp (id(self))


    def blacklist_roi(self, blacklist:str):
        r"""Set the ROI blacklist.

        Blacklisted ROI labels are skipped during featurization of files
        (:meth:`featurize_directory`, :meth:`featurize_files`) and do not appear in
        the output. Each call replaces the previous blacklist.

        Parameters
        ----------
        blacklist : str
            Either a global list of labels, ``"27,28,30"``, or per-file lists
            separated by semicolons, ``"file1.ome.tif:5,6,7;file2.ome.tif:11,12"``.
            A per-file list names the mask file without its directory; a name
            with a directory part is accepted but matches no file.

        Raises
        ------
        OSError
            If `blacklist` is None or blank.
        RuntimeError
            If `blacklist` cannot be parsed.

        See Also
        --------
        clear_roi_blacklist, roi_blacklist_get_summary

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> nyx.blacklist_roi("27,28,30")
        >>> nyx.roi_blacklist_get_summary()
        'global 27 28 30'
        """
        if blacklist is None:
            raise IOError ("ROI blacklist argument is empty")

        if len(blacklist.strip()) == 0:
            raise IOError ("ROI blacklist argument is non-informative")

        blacklist_roi_imp (id(self), blacklist)

    def clear_roi_blacklist(self):
        r"""Clear the ROI blacklist.

        See Also
        --------
        blacklist_roi

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> nyx.blacklist_roi("5")
        >>> nyx.clear_roi_blacklist()
        >>> nyx.roi_blacklist_get_summary() is None
        True
        """
        clear_roi_blacklist_imp (id(self))

    def roi_blacklist_get_summary (self):
        r"""Describe the ROI blacklist.

        Returns
        -------
        str or None
            Human-readable summary of the blacklist, or None when it is empty.

        See Also
        --------
        blacklist_roi

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> nyx.blacklist_roi("file1.ome.tif:5,6;file2.ome.tif:7")
        >>> nyx.roi_blacklist_get_summary()
        'file1.ome.tif : 5 6 file2.ome.tif : 7'
        """
        s = roi_blacklist_get_summary_imp (id(self))
        s = s.strip()
        if len(s) == 0:
            return None
        return s
    
    def set_gabor_feature_params (self, **kwargs):
        r"""Set the parameters of the GABOR feature.

        Only the parameters passed are changed.

        Parameters
        ----------
        kersize : int, optional
            Side of the filter kernel, in pixels.
        gamma : float, optional
            Aspect ratio of the Gaussian factor.
        sig2lam : float, optional
            Spatial frequency bandwidth (sigma to lambda ratio).
        f0 : float, optional
            Angular frequency of the baseline low-pass filter, in radians per
            pixel; its wavelength is :math:`2\pi/f_0` pixels.
        thold : float, optional
            Lower threshold of the filtered-image to baseline ratio.
        thetas : list of float, optional
            Orientations of the Gaussian, in degrees, conventionally 0-180
            (not enforced), one per entry of `freqs`. Must be a list; a
            comma-separated string is rejected.
        freqs : list of float, optional
            Angular frequencies of the filters' harmonic factor, in radians per
            pixel (wavelength :math:`2\pi/f` pixels), one per entry of `thetas`.
            Must be a list.

        Raises
        ------
        ValueError
            If a keyword is not one of the above, or a value is invalid (for
            example `thetas` and `freqs` of different lengths).
        OSError
            If no keyword is passed.

        Notes
        -----
        Gabor parameters are process-wide: the change applies to every instance,
        and the next :class:`Nyxus` constructor call resets them.

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["GABOR"])
        >>> nyx.set_gabor_feature_params(kersize=16, gamma=0.1, sig2lam=0.8,
        ...                              f0=0.1, thold=0.025,
        ...                              thetas=[0, 90], freqs=[4, 16])
        >>> nyx.get_params("gabor_thetas", "gabor_freqs")
        {'gabor_freqs': [4.0, 16.0], 'gabor_thetas': [0.0, 90.0]}
        """

        params = [
            'kersize',
            'gamma',
            'sig2lam',
            'f0',
            'thold',
            'thetas',
            'freqs'
        ]
        
        for key in kwargs:
            if key not in params:
                raise ValueError(f"Invalid Gabor parameter {key}. The valid parameters are: {params}")
    
        kersize = str(kwargs.get ('kersize', ""))
        gamma = str(kwargs.get ('gamma', ""))
        sig2lam = str(kwargs.get ('sig2lam', ""))
        f0 = str(kwargs.get ('f0', ""))
        thold = str(kwargs.get ('thold', ""))
        thetas = kwargs.get ('thetas', [])
        freqs = kwargs.get ('freqs', [])

        if len(kersize)==0 and len(gamma)==0 and len(sig2lam)==0 and len(f0)==0 and len(thold)==0 and len(freqs)==0 and len(thetas)==0 :
            raise IOError ("Illegal arguments passed to set_gabor_feature_params()")
        
        if (freqs == []):
            freqs = ""
        else:
            freqs = ",".join(str(i) for i in freqs)

        if (thetas == []):
            thetas = ""
        else:
            thetas = ",".join(str(i) for i in thetas)

        customize_gabor_feature_imp (id(self), kersize, gamma, sig2lam, f0, thetas, thold, freqs)

    
    def set_environment_params (self, **params):
        r"""Change environment parameters of this instance.

        Only the parameters passed are changed, except `verbose` (see Notes).

        Parameters
        ----------
        features : list of str, optional
            Stored and reported by :meth:`get_params`, but the enabled feature
            set does not change (see Notes).
        neighbor_distance : int, optional
            Neighbor cutoff distance, in pixels.
        pixels_per_micron : float, optional
            Stored and reported by :meth:`get_params`; no feature currently
            depends on it.
        coarse_gray_depth : int, optional
            Number of grey levels for texture features (see Notes).
        n_feature_calc_threads : int, optional
            Number of feature calculation threads.
        use_gpu_device : int, optional
            Accepted and ignored.
        verbose : int, optional
            Level of diagnostic output.
        dynamic_range : float, optional
            Number of grey levels a floating-point image is rescaled to.
        min_intensity : float, optional
            Intensity mapped to the lowest grey level. Negative values are
            ignored.
        max_intensity : float, optional
            Intensity mapped to `dynamic_range`. Negative values are ignored.
        ram_limit : int, optional
            Memory budget in megabytes, with the same effect as the
            constructor's `ram_limit`. A
            value larger than the available memory is rejected with a message on
            standard error and the previous limit is kept.

        Raises
        ------
        ValueError
            If a keyword is not one of the above.

        Notes
        -----
        `verbose` is applied on every call and falls back to 0 when omitted.

        The enabled feature set and the per-feature settings derived from
        `coarse_gray_depth` are fixed when the instance is constructed. Changing
        `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
        :meth:`get_params` reports but not the features computed by this
        instance; construct a new instance to change them.

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> nyx.set_environment_params(neighbor_distance=3, pixels_per_micron=2.0)
        >>> nyx.get_params("neighbor_distance", "pixels_per_micron")
        {'neighbor_distance': 3, 'pixels_per_micron': 2.0}
        """
        
        valid_params = [
            'features',
            'neighbor_distance',
            'pixels_per_micron',
            'coarse_gray_depth',
            'n_feature_calc_threads',
            'use_gpu_device',
            'verbose',
            'dynamic_range',
            'min_intensity',
            'max_intensity',
            'ram_limit',
        ]
        
        for key in params:
            if key not in valid_params:
                raise ValueError(f'Invalid environment parameter {key}. Value parameters are {params}')
        
        features = params.get('features', [])
        neighbor_distance = params.get ('neighbor_distance', -1)
        pixels_per_micron = params.get ('pixels_per_micron', -1)
        coarse_gray_depth = params.get ('coarse_gray_depth', 0)
        n_reduce_threads = params.get ('n_feature_calc_threads', 0)
        use_gpu_device = params.get ('use_gpu_device', -1)
        verb_lvl = params.get ('verbose', 0)
        dynamic_range = params.get('dynamic_range', -1)
        min_intensity = params.get('min_intensity', -1)
        max_intensity = params.get('max_intensity', -1)
        ram_limit = params.get('ram_limit', -1)
        
        set_environment_params_imp (id(self),
                                   features, 
                                   neighbor_distance, 
                                   pixels_per_micron,
                                   coarse_gray_depth,
                                   n_reduce_threads,
                                   use_gpu_device,
                                   dynamic_range,
                                   min_intensity,
                                   max_intensity,
                                   ram_limit,
                                   verb_lvl)
        
    def set_params(self, **params):
        r"""Change parameters of this instance.

        Accepts the keywords of :meth:`set_environment_params`, plus `ibsi` and
        the ``gabor_*`` keywords of the constructor, which are forwarded to
        :meth:`set_gabor_feature_params` without the ``gabor_`` prefix.

        Parameters
        ----------
        **params
            Any of `features`, `neighbor_distance`, `pixels_per_micron`,
            `coarse_gray_depth`, `n_feature_calc_threads`, `use_gpu_device`,
            `ibsi` (bool), `dynamic_range`, `min_intensity`, `max_intensity`,
            `ram_limit`, `verbose`, `gabor_kersize`, `gabor_gamma`,
            `gabor_sig2lam`, `gabor_f0`, `gabor_thold`, `gabor_thetas` (list) and
            `gabor_freqs` (list).

        Raises
        ------
        ValueError
            If a keyword is not recognized or a Gabor value is invalid.

        Notes
        -----
        The enabled feature set and the per-feature settings derived from
        `coarse_gray_depth` are fixed when the instance is constructed. Changing
        `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
        :meth:`get_params` reports but not the features computed by this
        instance; construct a new instance to change them. Changing `ibsi` afterwards does not enable the
        Intensity Histogram (``IH_*``) family.

        Gabor parameters are process-wide, not per instance: every constructor
        call that accepts ``gabor_*`` arguments sets them (to the defaults when
        omitted), and :meth:`set_gabor_feature_params` changes them for all
        instances.

        See Also
        --------
        set_environment_params, set_gabor_feature_params, get_params

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> nyx.set_params(neighbor_distance=3, gabor_thetas=[0, 90], gabor_freqs=[4, 16])
        >>> nyx.get_params("neighbor_distance", "gabor_thetas")
        {'gabor_thetas': [0.0, 90.0], 'neighbor_distance': 3}
        """
        
        available_environment_params = [
            "features",
            "neighbor_distance",
            "pixels_per_micron",
            "coarse_gray_depth",
            "n_feature_calc_threads",
            "use_gpu_device",
            "ibsi",
            "dynamic_range",
            "min_intensity",
            "max_intensity",
            "ram_limit",
            "verbose"
        ]
        
        environment_params = {}
        
        gabor_params = {}
        
        for key, value in params.items():
            if key.startswith("gabor_"):
                gabor_params[key[len("gabor_"):]] = value
            
            elif (key == "ibsi"):
                set_if_ibsi_imp (id(self), value)
            
            else:
                if (key not in available_environment_params):
                    raise ValueError ("Invalid parameter: ", key)
                else:
                    environment_params[key] = value
                
        if (len(gabor_params) > 0):
            self.set_gabor_feature_params(**gabor_params)
        
        if (len(environment_params) > 0):
            self.set_environment_params(**environment_params)


    def get_params(self, *args):
        r"""Return parameters of this instance.

        Parameters
        ----------
        *args : str
            Names of the parameters to return. With no names, all are returned.
            Unknown names are ignored. Available names: ``features``,
            ``neighbor_distance``, ``pixels_per_micron``, ``coarse_gray_depth``,
            ``n_feature_calc_threads``, ``ibsi``, ``dynamic_range``,
            ``min_intensity``, ``max_intensity``, ``ram_limit`` (megabytes),
            ``gabor_kersize``, ``gabor_gamma``, ``gabor_sig2lam``, ``gabor_f0``,
            ``gabor_thold``, ``gabor_thetas`` and ``gabor_freqs``.

        Returns
        -------
        dict
            Parameter name to value. ``ibsi`` is reported as 0 or 1. The GPU
            settings and `verbose` are not reported.

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> nyx.get_params("coarse_gray_depth", "neighbor_distance", "ibsi")
        {'coarse_gray_depth': 64, 'ibsi': 0, 'neighbor_distance': 5}
        """
        vars = list(args)
        
        return get_params_imp (id(self), vars)
        
    
    def get_arrow_ipc_file(self):
        r"""Return the path of the last Arrow IPC file written by this instance.

        Returns
        -------
        str
            Path of the most recent Arrow IPC or Parquet output file. The same
            path is returned by :meth:`get_arrow_ipc_file` and
            :meth:`get_parquet_file`, whichever format was written last.

        Raises
        ------
        RuntimeError
            If this build of Nyxus has no Arrow support.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> path = nyx.featurize_directory("int", "seg", output_type="arrowipc", output_path="out")
        >>> nyx.get_arrow_ipc_file() == path
        True
        """
        
        if self.arrow_is_enabled():
            return get_arrow_file_imp (id(self))
        else:
            raise RuntimeError("Nyxus was not built with Arrow. To use this functionality, rebuild Nyxus with Arrow support on.")
        
    def get_parquet_file(self):
        r"""Return the path of the last Parquet file written by this instance.

        Returns
        -------
        str
            Path of the most recent Arrow IPC or Parquet output file. The same
            path is returned by :meth:`get_arrow_ipc_file` and
            :meth:`get_parquet_file`, whichever format was written last.

        Raises
        ------
        RuntimeError
            If this build of Nyxus has no Arrow support.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import Nyxus
        >>> nyx = Nyxus(["MEAN"])
        >>> path = nyx.featurize_directory("int", "seg", output_type="parquet", output_path="out")
        >>> nyx.get_parquet_file() == path
        True
        """
        if self.arrow_is_enabled():
            return get_parquet_file_imp (id(self))
        else:
            raise RuntimeError("Nyxus was not built with Arrow. To use this functionality, rebuild Nyxus with Arrow support on.")
    

    
    def arrow_is_enabled(self):
        r"""Report whether this build of Nyxus supports Arrow IPC and Parquet output.

        Returns
        -------
        bool
            True when the "arrowipc" and "parquet" output types are available.

        Examples
        --------
        >>> from nyxus import Nyxus
        >>> enabled = Nyxus(["MEAN"]).arrow_is_enabled()
        """
        

        return arrow_is_enabled_imp (id(self))
       

class Nyxus3D:
    r"""Extract 3D features from volumes and their ROI masks.

    Volumes are read from files (:meth:`featurize_directory`,
    :meth:`featurize_files`); there is no numpy-array entry point.

    Parameters
    ----------
    features : list of str
        3D feature names and feature groups to compute. 3D groups:
        ``*3D_ALL*``, ``*3D_ALL_INTENSITY*``, ``*3D_ALL_MORPHOLOGY*``,
        ``*3D_ALL_TEXTURE*``, ``*3D_GLCM*``, ``*3D_GLDM*``, ``*3D_GLDZM*``,
        ``*3D_GLSZM*``, ``*3D_GLRLM*``, ``*3D_NGLDM*`` and ``*3D_NGTDM*``.
        Names are case-sensitive and must be upper case: ``"3MEAN"`` is accepted,
        ``"3mean"`` raises ValueError. (The command-line tool upper-cases its
        ``--features`` list; the Python API does not.) Entries are applied left to
        right; a leading ``-`` removes a feature or group enabled by an earlier
        entry and a leading ``+`` adds it, e.g. ``["*3D_ALL_INTENSITY*", "-3MEAN"]``.
    neighbor_distance : int, optional
        Default 5. Must be positive. There are no 3D neighbor features; the
        value is stored and reported by :meth:`get_params` only.
    pixels_per_micron : float, optional
        Default 1.0. Must be positive. Stored and reported by
        :meth:`get_params`; no 3D feature depends on it.
    coarse_gray_depth : int, optional
        Default 64. Must be positive. Grey depth of the MATLAB-style binning
        of 3D GLCM and 3D GLDZM, of the 3D NGLDM conversion, and the bin count
        of ``3ENTROPY`` and ``3UNIFORMITY``. 3D GLRLM, GLSZM, GLDM and NGTDM
        ignore it: they bin to their own ``greydepth`` metaparameter (see
        :meth:`set_metaparam`), 0 (no binning) by default. With ``ibsi=True``
        no 3D texture family bins.
    n_feature_calc_threads : int, optional
        Default 4. Number of feature calculation threads. Must be at least 1.
    use_gpu_device : int, optional
        Default -1 (CPU). No 3D feature uses the GPU, and the value is not
        passed to CUDA. A non-negative id forces `n_feature_calc_threads` to
        1, and falls back to -1 with a printed message when no GPU is
        available. A value below -1 raises RuntimeError on a build without
        GPU support.
    ibsi : bool, optional
        Default False. Compute the IBSI-compliant variant of the features that
        have one. The texture features then skip grey-level binning, whatever
        ``coarse_gray_depth`` and the ``greydepth`` metaparameters are.
    dynamic_range : float, optional
        Default 10000. Number of grey levels a floating-point image is
        rescaled to.
    min_intensity : float, optional
        Default 0.0. Intensity of a floating-point image mapped to the lowest
        grey level.
    max_intensity : float, optional
        Default 1.0. Intensity of a floating-point image mapped to
        `dynamic_range`.
    preserve_hu : bool, optional
        Default False. Not needed for CT: Hounsfield units are preserved and
        reported either way, despite the option's name. What it does: carries
        a floating-point slide on the offset-preserving map (1 grey level = 1
        intensity unit, offset by the slide's floored minimum) instead of the
        min-max rescaling controlled by `dynamic_range`, `min_intensity` and
        `max_intensity`. A DICOM or integer NIfTI slide takes it instead of
        the stored map, which keeps the stored integers as grey levels and
        carries RescaleSlope/RescaleIntercept (scl_slope/scl_inter) in the
        recorded inverse, so a fractional slope loses no stored level. Other
        integer slides take the offset map regardless of this flag. The DICOM
        and NIfTI rescale reaches the reported values either way, so CT
        Hounsfield units survive the load and negative values never wrap.

        Whichever map a slide takes is recorded and undone on the way out:

        * Every intensity feature -- location (mean, median, mode,
          percentiles, min, max), dispersion and the sum/energy family -- is
          reported in the slide's own intensity domain. A CT read in Hounsfield
          units is reported in Hounsfield units, negatives included.
        * Shift-invariant intensity features (variance, standard deviation,
          skewness, kurtosis, range, interquartile range) and all
          shape/texture features are unaffected.
        * Sub-unit precision survives on the stored map, which reports a
          fractional rescaled value exactly. Elsewhere grey levels are
          integers, so a value that is not integer-valued is reported as the
          grey level it was stored as: truncated, or rounded to nearest under
          `preserve_hu`. Hounsfield units are integer-valued, so CT is
          unaffected.
    use_physical_spacing : bool, optional
        Default False. Take the voxel spacing from the OME ``PhysicalSize*``
        metadata and treat non-cubic voxels as anisotropic. An anisotropy
        factor other than 1.0 in `anisotropy_x`, `anisotropy_y` or
        `anisotropy_z` takes precedence, and then all three come from those
        arguments. A factor of 1.0 cannot be told from the default and does
        not override the metadata.
    ram_limit : int, optional
        Memory budget in megabytes. ROIs whose estimated footprint reaches it
        are processed out of core, except when one read of the file returns
        the whole volume (NIfTI, or OME-Zarr chunked over all of Z): then such
        a ROI stops the run with RuntimeError. Default -1 keeps the built-in limit, half
        of the physical memory available when the instance is created. A
        value larger than the available memory is rejected with a message on
        standard error and the previous limit is kept.
    verbose : int, optional
        Default 0 (silent). Level of diagnostic output. Must be non-negative.
    anisotropy_x : float, optional
        Default 1.0. Size of a voxel along X relative to the other axes. Must
        be positive. The volume is featurized on its voxels as acquired and
        the shape features scale their coordinates by the anisotropy, so it
        changes only shape values.
    anisotropy_y : float, optional
        Default 1.0. Y-dimension counterpart of `anisotropy_x`. Must be
        positive.
    anisotropy_z : float, optional
        Default 1.0. Z-dimension counterpart of `anisotropy_x`. Must be
        positive.

    Other Parameters
    ----------------
    channel_signature, parent_channel, child_channel, aggregate
        Accepted without a warning and ignored.

    Raises
    ------
    ValueError
        If a feature name or group is not recognized or a numeric parameter
        is out of range.
    RuntimeError
        If `use_gpu_device` is below -1 and this build has no GPU support.

    Notes
    -----
    Unknown keyword arguments print a warning and are ignored. The ``gabor_*``
    keywords are not accepted: there is no 3D Gabor feature.

    An OME-TIFF takes its Z, C and T sizes from the OME metadata; a plain
    multi-page TIFF is read as one volume with one Z plane per page.

    The enabled feature set and the per-feature settings derived from
    `coarse_gray_depth` are fixed when the instance is constructed. Changing
    `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
    :meth:`get_params` reports but not the features computed by this
    instance; construct a new instance to change them.

    Examples
    --------
    >>> import numpy as np, os, tifffile
    >>> from nyxus import Nyxus3D
    >>> vol = np.zeros((6, 12, 12), dtype=np.uint16)
    >>> vol[1:5, 2:9, 2:9] = 7
    >>> os.makedirs("vol_int", exist_ok=True)
    >>> os.makedirs("vol_seg", exist_ok=True)
    >>> tifffile.imwrite("vol_int/v.ome.tif", vol, ome=True, metadata={"axes": "ZYX"})
    >>> tifffile.imwrite("vol_seg/v.ome.tif", (vol > 0).astype(np.uint16),
    ...                  ome=True, metadata={"axes": "ZYX"})
    >>> nyx = Nyxus3D(["3MEAN", "3VOXEL_VOLUME"])
    >>> df = nyx.featurize_directory("vol_int", "vol_seg", ".*")
    >>> df[["ROI_label", "3MEAN"]].values.tolist()
    [[1.0, 7.0]]
    """

    def __del__(self):
        # The backend keys this instance's settings on id(self), which is the object's
        # address -- and CPython hands that same address to the next object it allocates.
        # An entry left behind after this object dies is inherited wholesale by that next
        # instance, which would then silently start with our ram_limit, feature list, gpu
        # flag or ibsi mode. Drop it here rather than leave the address dirty.
        try:
            release_environment_imp(id(self))
        except Exception:
            pass    # interpreter shutdown may already have torn the backend module down

    def __init__(
        self,
        features: List[str],
        **kwargs
        ):
        valid_keys = {
            'neighbor_distance', 'pixels_per_micron', 'coarse_gray_depth',
            'n_feature_calc_threads', 'use_gpu_device', 'ibsi', 'channel_signature', 
            'parent_channel', 'child_channel', 'aggregate', 
            'dynamic_range', 'min_intensity', 'max_intensity', 'ram_limit',
            'verbose',
            'anisotropy_x',
            'anisotropy_y',
            'anisotropy_z', 'preserve_hu', 'use_physical_spacing'
        }

        # Check for unexpected keyword arguments
        invalid_keys = set(kwargs.keys()) - valid_keys
        if invalid_keys:
            print(f"Warning: unexpected keyword argument(s): {', '.join(invalid_keys)}")
        
        # parse kwargs
        features = features
        neighbor_distance = kwargs.get('neighbor_distance', 5)
        pixels_per_micron = kwargs.get('pixels_per_micron', 1.0)
        coarse_gray_depth = kwargs.get('coarse_gray_depth', 64)
        n_feature_calc_threads = kwargs.get('n_feature_calc_threads', 4)
        use_gpu_device = kwargs.get('use_gpu_device', -1)
        ibsi = kwargs.get('ibsi', False)
        dynamic_range = kwargs.get('dynamic_range', 10000)
        min_intensity = kwargs.get('min_intensity', 0.0)
        max_intensity = kwargs.get('max_intensity', 1.0)
        preserve_hu = kwargs.get('preserve_hu', False)   # CT/HU: preserve absolute Hounsfield values
        use_physical_spacing = kwargs.get('use_physical_spacing', False)   # opt-in OME PhysicalSize* voxel spacing (3D)
        verb_lvl = kwargs.get ('verbose', 0)
        aniso_x = kwargs.get('anisotropy_x', 1.0)
        aniso_y = kwargs.get('anisotropy_y', 1.0)
        aniso_z = kwargs.get('anisotropy_z', 1.0)
        
        if neighbor_distance <= 0:
            raise ValueError("Neighbor distance must be greater than zero.")

        if pixels_per_micron <= 0:
            raise ValueError("Pixels per micron must be greater than zero.")

        if coarse_gray_depth <= 0:
            raise ValueError("Custom number of grayscale levels (parameter coarse_gray_depth, default=64) must be positive.")

        if n_feature_calc_threads < 1:
            raise ValueError("There must be at least one feature calculation thread.")
        
        if(use_gpu_device > -1 and n_feature_calc_threads != 1):
            print("Gpu features only support a single thread. Defaulting to one thread.")
            n_feature_calc_threads = 1
            
        if(use_gpu_device > -1 and not gpu_available_imp(id(self))):
            print("No gpu available.")
            use_gpu_device = -1

        ram_limit = kwargs.get('ram_limit', -1)

        if verb_lvl < 0:
            raise ValueError ("verbosity must be non-negative")

        if aniso_x <= 0:
            raise ValueError ("anisotropy_x must be positive")

        if aniso_y <= 0:
            raise ValueError ("anisotropy_y must be positive")

        if aniso_z <= 0:
            raise ValueError ("anisotropy_z must be positive")

        initialize_environment(
            id(self),
            3, # 3D
            features,
            neighbor_distance,
            pixels_per_micron,
            coarse_gray_depth,
            n_feature_calc_threads,
            use_gpu_device,
            ibsi,
            dynamic_range,
            min_intensity,
            max_intensity,
            False,
            ram_limit,
            verb_lvl,
            aniso_x,
            aniso_y,
            aniso_z,
            False,  # merge_labels: 2D-segmented only
            preserve_hu,
            use_physical_spacing)
        
        # list of valid outputs that are used throughout featurize functions
        self._valid_output_types = ['pandas', 'arrowipc', 'parquet']

    def set_metaparam(
        self, 
        paramval: str):
        r"""Set a feature-specific parameter.

        Parameters
        ----------
        paramval : str
            Assignment of the form ``"<feature>/<parameter>=<value>"``, for
            example ``"3ngtdm/radius=2"``.
            Recognized names: ``glcm/greydepth``, ``glcm/offset``,
            ``3glcm/greydepth``, ``3glcm/offset``, ``3glcm/numang``,
            ``3glcm/sparseintensities``, ``3gldm/greydepth``, ``3ngtdm/greydepth``,
            ``3ngtdm/radius``, ``3glrlm/greydepth`` and ``3glszm/greydepth``. The
            value is an integer, except ``3glcm/sparseintensities`` which takes
            ``true`` or ``false``.

        Raises
        ------
        ValueError
            If the string is malformed, the feature or parameter is not
            recognized, or the value cannot be parsed.

        Notes
        -----
        A name without a ``/`` (e.g. ``"greydepth=8"``) is accepted and has no
        effect.

        See Also
        --------
        get_metaparam

        Examples
        --------
        >>> from nyxus import Nyxus3D
        >>> nyx = Nyxus3D(["*3D_NGTDM*"])
        >>> nyx.set_metaparam("3ngtdm/radius=2")
        >>> nyx.get_metaparam("3ngtdm/radius")
        2.0
        """
        ok,errordetails = set_metaparam_imp (id(self), paramval)
        if ok==False:
            raise  ValueError (f'Invalid metaparameter value {paramval}: {errordetails}')

    def get_metaparam (
        self, 
        paramname: str):
        r"""Get a feature-specific parameter.

        Parameters
        ----------
        paramname : str
            Name of the form ``"<feature>/<parameter>"``, for example
            ``"3ngtdm/radius"``. See :meth:`set_metaparam` for the names.

        Returns
        -------
        float
            Current value.

        Raises
        ------
        NameError
            If the feature or parameter is not recognized.

        Notes
        -----
        The ``3glcm/*`` names read the 2D ``glcm`` values, not the ones set
        through ``3glcm/*``, a known defect
        (`#513 <https://github.com/PolusAI/nyxus/issues/513>`__). ``3glszm/greydepth`` is read under the name
        ``3glsz/greydepth``. A name without a ``/`` returns an undefined value.

        See Also
        --------
        set_metaparam

        Examples
        --------
        >>> from nyxus import Nyxus3D
        >>> nyx = Nyxus3D(["*3D_GLRLM*"])
        >>> nyx.set_metaparam("3glrlm/greydepth=32")
        >>> nyx.get_metaparam("3glrlm/greydepth")
        32.0
        """
        val,errordetails = get_metaparam_imp (id(self), paramname)
        if len(errordetails) > 0:
            raise  NameError (f'Invalid metaparameter name {paramname}: {errordetails}')
        return val

    def featurize_directory(
        self,
        intensity_dir: str,
        label_dir: str,
        file_pattern: str,
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
    ):
        r"""Extract 3D features from the volumes in a directory.

        Pairs each intensity volume in `intensity_dir` with the mask volume of
        the same file name in `label_dir` and computes the features of every ROI.

        Parameters
        ----------
        intensity_dir : str
            Directory of intensity volumes.
        label_dir : str or None
            Directory of mask (label) volumes. None processes each intensity
            volume as one ROI.
        file_pattern : str
            Regular expression selecting the files. It must match the whole file
            name, case-sensitively, in both directories. Required; pass ".*" for
            all files.
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        OSError
            If `intensity_dir` or `label_dir` does not exist.
        ValueError
            If `output_type` is not recognized or `file_pattern` is not a valid
            pattern.
        RuntimeError
            If the dataset cannot be read.

        Examples
        --------
        See :class:`Nyxus3D`.
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

        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}.')
            
        if (output_type == 'pandas'):
            header, string_data, numeric_data = featurize_directory_3D_imp (id(self), intensity_dir, label_dir, file_pattern, output_type, "")
            df = pd.concat(
                [
                    pd.DataFrame(string_data, columns=header[: string_data.shape[1]]),
                    pd.DataFrame(numeric_data, columns=header[string_data.shape[1] :]),
                ],
                axis=1,
            )
            # Labels should always be uint
            if "ROI_label" in df.columns:
                df.ROI_label = df.ROI_label.astype(np.uint32)
            return df
        else:
            featurize_directory_3D_imp (id(self), intensity_dir, label_dir, file_pattern, output_type, output_path)
            return get_arrow_file_imp (id(self)) # return path to file


    def featurize_files (
        self,
        intensity_files: list,
        mask_files: list,
        single_roi: bool,
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
    ):
        r"""Extract 3D features from lists of volume files.

        Parameters
        ----------
        intensity_files : list of str
            Paths of intensity images.
        mask_files : list of str
            Paths of mask (label) volumes, paired with `intensity_files` by
            position. Required, one mask per intensity file, whatever
            `single_roi` is.
        single_roi : bool
            True merges every nonzero label of each mask into one ROI (label 1)
            and computes the 3D shape features from that ROI's bounding box;
            voxels where the mask is 0 stay out. Use
            :meth:`featurize_directory` with `label_dir` None for whole-volume
            ROIs.
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        OSError
            If `intensity_files` or `mask_files` is None.
        ValueError
            If `output_type` is not recognized.
        RuntimeError
            If a list is empty, the lists differ in length, a file does not
            exist, or featurization fails.

        Examples
        --------
        >>> import numpy as np, os, tifffile
        >>> from nyxus import Nyxus3D
        >>> vol = np.zeros((6, 12, 12), dtype=np.uint16)
        >>> vol[1:5, 2:9, 2:9] = 7
        >>> os.makedirs("vol_int", exist_ok=True)
        >>> os.makedirs("vol_seg", exist_ok=True)
        >>> tifffile.imwrite("vol_int/v.ome.tif", vol, ome=True, metadata={"axes": "ZYX"})
        >>> tifffile.imwrite("vol_seg/v.ome.tif", (vol > 0).astype(np.uint16),
        ...                  ome=True, metadata={"axes": "ZYX"})
        >>> nyx = Nyxus3D(["3MEAN"])
        >>> df = nyx.featurize_files(["vol_int/v.ome.tif"], ["vol_seg/v.ome.tif"], False)
        >>> df["3MEAN"].tolist()
        [7.0]
        """

        if intensity_files is None:
            raise IOError ("The list of intensity file paths is empty")

        if mask_files is None:
            raise IOError ("The list of segment file paths is empty")
        
        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}')

        if (output_type == 'pandas'):
            
            header, string_data, numeric_data = featurize_fname_lists_3D_imp (id(self), intensity_files, mask_files, single_roi, output_type, "")

            df = pd.concat(
                [
                    pd.DataFrame(string_data, columns=header[: string_data.shape[1]]),
                    pd.DataFrame(numeric_data, columns=header[string_data.shape[1] :]),
                ],
                axis=1,
            )

            # Labels should always be uint
            if "ROI_label" in df.columns:
                df.ROI_label = df.ROI_label.astype(np.uint32)

            return df
        
        else:
            
            featurize_fname_lists_3D_imp (id(self), intensity_files, mask_files, single_roi, output_type, output_path)
            return get_arrow_file_imp (id(self))


    def set_environment_params (self, **params):
        r"""Change environment parameters of this instance.

        Only the parameters passed are changed, except `verbose` (see Notes).

        Parameters
        ----------
        features : list of str, optional
            Stored and reported by :meth:`get_params`, but the enabled feature
            set does not change (see Notes).
        neighbor_distance : int, optional
            Stored and reported by :meth:`get_params`; no feature currently
            depends on it.
        pixels_per_micron : float, optional
            Stored and reported by :meth:`get_params`; no feature currently
            depends on it.
        coarse_gray_depth : int, optional
            Number of grey levels for texture features (see Notes).
        n_feature_calc_threads : int, optional
            Number of feature calculation threads.
        use_gpu_device : int, optional
            Accepted and ignored.
        verbose : int, optional
            Level of diagnostic output.
        dynamic_range : float, optional
            Number of grey levels a floating-point image is rescaled to.
        min_intensity : float, optional
            Intensity mapped to the lowest grey level. Negative values are
            ignored.
        max_intensity : float, optional
            Intensity mapped to `dynamic_range`. Negative values are ignored.

        Raises
        ------
        ValueError
            If a keyword is not one of the above.

        Notes
        -----
        `verbose` is applied on every call and falls back to 0 when omitted.

        The enabled feature set and the per-feature settings derived from
        `coarse_gray_depth` are fixed when the instance is constructed. Changing
        `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
        :meth:`get_params` reports but not the features computed by this
        instance; construct a new instance to change them.

        Examples
        --------
        >>> from nyxus import Nyxus3D
        >>> nyx = Nyxus3D(["3MEAN"])
        >>> nyx.set_environment_params(neighbor_distance=3, pixels_per_micron=2.0)
        >>> nyx.get_params("neighbor_distance", "pixels_per_micron")
        {'neighbor_distance': 3, 'pixels_per_micron': 2.0}
        """
        
        valid_params = [
            'features',
            'neighbor_distance',
            'pixels_per_micron',
            'coarse_gray_depth',
            'n_feature_calc_threads',
            'use_gpu_device',
            'verbose',
            'dynamic_range',
            'min_intensity',
            'max_intensity'
        ]
        
        for key in params:
            if key not in valid_params:
                raise ValueError(f'Invalid environment parameter {key}. Value parameters are {params}')
        
        features = params.get('features', [])
        neighbor_distance = params.get ('neighbor_distance', -1)
        pixels_per_micron = params.get ('pixels_per_micron', -1)
        coarse_gray_depth = params.get ('coarse_gray_depth', 0)
        n_reduce_threads = params.get ('n_feature_calc_threads', 0)
        use_gpu_device = params.get ('use_gpu_device', -1)
        verb_lvl = params.get ('verbose', 0)
        dynamic_range = params.get('dynamic_range', -1)
        min_intensity = params.get('min_intensity', -1)
        max_intensity = params.get('max_intensity', -1)
        ram_limit = -1 # no limit
        
        set_environment_params_imp (id(self),
                                   features, 
                                   neighbor_distance, 
                                   pixels_per_micron,
                                   coarse_gray_depth,
                                   n_reduce_threads,
                                   use_gpu_device,
                                   dynamic_range,
                                   min_intensity,
                                   max_intensity,
                                   ram_limit,
                                   verb_lvl)
        
    def set_params(self, **params):
        r"""Change parameters of this instance.

        Accepts the keywords of :meth:`set_environment_params` except
        `use_gpu_device` and `verbose`, plus `ibsi`.

        Parameters
        ----------
        **params
            Any of `features`, `neighbor_distance`, `pixels_per_micron`,
            `coarse_gray_depth`, `n_feature_calc_threads`, `ibsi` (bool),
            `dynamic_range`, `min_intensity` and `max_intensity`.

        Raises
        ------
        ValueError
            If a keyword is not recognized. ``using_gpu`` passes the first check
            but is rejected by :meth:`set_environment_params`.

        Notes
        -----
        The enabled feature set and the per-feature settings derived from
        `coarse_gray_depth` are fixed when the instance is constructed. Changing
        `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
        :meth:`get_params` reports but not the features computed by this
        instance; construct a new instance to change them.

        Examples
        --------
        >>> from nyxus import Nyxus3D
        >>> nyx = Nyxus3D(["3MEAN"])
        >>> nyx.set_params(neighbor_distance=3, ibsi=True)
        >>> nyx.get_params("neighbor_distance", "ibsi")
        {'ibsi': 1, 'neighbor_distance': 3}
        """
        
        available_environment_params = [
            "features",
            "neighbor_distance",
            "pixels_per_micron",
            "coarse_gray_depth",
            "n_feature_calc_threads",
            "using_gpu",
            "ibsi",
            "dynamic_range",
            "min_intensity",
            "max_intensity"
        ]
        
        environment_params = {}
        
        gabor_params = {}
        
        
        for key, value in params.items():
           
            if (key == "ibsi"):
                set_if_ibsi_imp (id(self), value)
            
            else:
                if (key not in available_environment_params):
                    raise ValueError(f"Invalid parameter {key}.")
                else:
                    environment_params[key] = value
                
                        
        if (len(environment_params) > 0):
            self.set_environment_params(**environment_params)
    
    def get_params(self, *args):
        r"""Return parameters of this instance.

        Parameters
        ----------
        *args : str
            Names of the parameters to return. With no names, all are returned.
            Unknown names are ignored. Available names: ``features``,
            ``neighbor_distance``, ``pixels_per_micron``, ``coarse_gray_depth``,
            ``n_feature_calc_threads``, ``ibsi``, ``dynamic_range``,
            ``min_intensity``, ``max_intensity``, ``ram_limit`` (megabytes),
            ``gabor_kersize``, ``gabor_gamma``, ``gabor_sig2lam``, ``gabor_f0``,
            ``gabor_thold``, ``gabor_thetas`` and ``gabor_freqs``.

        Returns
        -------
        dict
            Parameter name to value. ``ibsi`` is reported as 0 or 1. The GPU
            settings and `verbose` are not reported.

        Examples
        --------
        >>> from nyxus import Nyxus3D
        >>> nyx = Nyxus3D(["3MEAN"])
        >>> nyx.get_params("coarse_gray_depth", "neighbor_distance", "ibsi")
        {'coarse_gray_depth': 64, 'ibsi': 0, 'neighbor_distance': 5}
        """
        vars = list(args)
        
        return get_params_imp (id(self), vars)
        
    
    def get_arrow_ipc_file(self):
        r"""Return the path of the last Arrow IPC file written by this instance.

        Returns
        -------
        str
            Path of the most recent Arrow IPC or Parquet output file. The same
            path is returned by :meth:`get_arrow_ipc_file` and
            :meth:`get_parquet_file`, whichever format was written last.

        Raises
        ------
        RuntimeError
            If this build of Nyxus has no Arrow support.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import Nyxus3D
        >>> nyx = Nyxus3D(["3MEAN"])
        >>> path = nyx.featurize_directory("int", "seg", ".*", output_type="arrowipc", output_path="out")
        >>> nyx.get_arrow_ipc_file() == path
        True
        """
        
        if self.arrow_is_enabled():
            return get_arrow_file_imp (id(self))
        else:
            raise RuntimeError("Nyxus was not built with Arrow. To use this functionality, rebuild Nyxus with Arrow support on.")
        
    def get_parquet_file(self):
        r"""Return the path of the last Parquet file written by this instance.

        Returns
        -------
        str
            Path of the most recent Arrow IPC or Parquet output file. The same
            path is returned by :meth:`get_arrow_ipc_file` and
            :meth:`get_parquet_file`, whichever format was written last.

        Raises
        ------
        RuntimeError
            If this build of Nyxus has no Arrow support.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import Nyxus3D
        >>> nyx = Nyxus3D(["3MEAN"])
        >>> path = nyx.featurize_directory("int", "seg", ".*", output_type="parquet", output_path="out")
        >>> nyx.get_parquet_file() == path
        True
        """
        if self.arrow_is_enabled():
            return get_parquet_file_imp (id(self))
        else:
            raise RuntimeError("Nyxus was not built with Arrow. To use this functionality, rebuild Nyxus with Arrow support on.")
    

    
    def arrow_is_enabled(self):
        r"""Report whether this build of Nyxus supports Arrow IPC and Parquet output.

        Returns
        -------
        bool
            True when the "arrowipc" and "parquet" output types are available.

        Examples
        --------
        >>> from nyxus import Nyxus3D
        >>> enabled = Nyxus3D(["3MEAN"]).arrow_is_enabled()
        """
        

        return arrow_is_enabled_imp (id(self))

class ImageQuality:
    r"""Extract image quality features.

    Image quality features (FOCUS_SCORE, LOCAL_FOCUS_SCORE,
    POWER_SPECTRUM_SLOPE, MAX_SATURATION, MIN_SATURATION, SHARPNESS) are
    computed per ROI, or per whole image when no mask is given.

    Parameters
    ----------
    features : list of str
        Image quality feature names, or the group ``*ALL_IMQ*``. The 2D
        groups of :class:`Nyxus` (``*ALL*`` and others) are not accepted.
        Names are case-sensitive and must be upper case.
    neighbor_distance, pixels_per_micron, coarse_gray_depth, ibsi : optional
        Accepted for parity with :class:`Nyxus` (defaults 5, 1.0, 256,
        False), validated as there, and reported by :meth:`get_params`. No
        image quality feature depends on them.
    n_feature_calc_threads : int, optional
        Default 4. Number of feature calculation threads. Must be at least 1.
    min_intensity : float, optional
        Default 0.0. Intensity of a floating-point image mapped to the lowest
        grey level.
    max_intensity : float, optional
        Default 1.0. Intensity of a floating-point image mapped to the
        highest grey level.
    ram_limit : int, optional
        Memory budget in megabytes. With a mask image, a ROI whose estimated
        footprint reaches it is processed out of core, where
        ``POWER_SPECTRUM_SLOPE`` and ``SHARPNESS`` are not computed. In
        whole-image mode, and for arrays passed to :meth:`featurize`, such an
        image or ROI raises RuntimeError instead. Default -1 keeps the built-in limit, half
        of the physical memory available when the instance is created. A
        value larger than the available memory is rejected with a message on
        standard error and the previous limit is kept.
    verbose : int, optional
        Default 0 (silent). Level of diagnostic output. Must be non-negative.
    anisotropy_x : float, optional
        Default 1.0. Size of a pixel along X relative to Y. Must be positive.
        With files, ``POWER_SPECTRUM_SLOPE`` is computed on the ROI resampled
        by the factors, a known defect
        (`#519 <https://github.com/PolusAI/nyxus/issues/519>`__). The other five features are computed
        on the pixels as acquired and do not change, and in-memory arrays are
        not affected.
    anisotropy_y : float, optional
        Default 1.0. Y-dimension counterpart of `anisotropy_x`. Must be
        positive.

    Other Parameters
    ----------------
    dynamic_range, preserve_hu, use_physical_spacing
        Applied as in :class:`Nyxus` (defaults 10000, False, False), but the
        constructor also prints an "unexpected keyword argument" warning for
        them.
    gabor_kersize, gabor_gamma, gabor_sig2lam, gabor_f0, channel_signature
        Accepted without a warning and ignored.

    Raises
    ------
    ValueError
        If a feature name is not recognized or a numeric parameter is out of
        range.

    Notes
    -----
    Unknown keyword arguments print a warning and are ignored, except
    ``using_gpu``: it also prints the warning, but a non-negative value sets
    `n_feature_calc_threads` to 1, and a value below -1 raises RuntimeError
    on a build without GPU support. GPU computation is not available for
    image quality features.

    The enabled feature set and the per-feature settings derived from
    `coarse_gray_depth` are fixed when the instance is constructed. Changing
    `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
    :meth:`get_params` reports but not the features computed by this
    instance; construct a new instance to change them.

    Examples
    --------
    >>> import numpy as np
    >>> from nyxus import ImageQuality
    >>> rng = np.random.default_rng(0)
    >>> image = rng.integers(0, 1000, (64, 64)).astype(np.uint16)
    >>> imq = ImageQuality(["*ALL_IMQ*"])
    >>> df = imq.featurize(image)
    >>> df.shape
    (1, 15)
    """

    def __del__(self):
        # The backend keys this instance's settings on id(self), which is the object's
        # address -- and CPython hands that same address to the next object it allocates.
        # An entry left behind after this object dies is inherited wholesale by that next
        # instance, which would then silently start with our ram_limit, feature list, gpu
        # flag or ibsi mode. Drop it here rather than leave the address dirty.
        try:
            release_environment_imp(id(self))
        except Exception:
            pass    # interpreter shutdown may already have torn the backend module down

    def __init__(
        self,
        features: List[str],
        **kwargs
        ):
        
        valid_keys = {
            'neighbor_distance', 'pixels_per_micron', 'coarse_gray_depth',
            'n_feature_calc_threads', 'ibsi',
            'gabor_kersize', 'gabor_gamma', 'gabor_sig2lam', 'gabor_f0',
            'channel_signature', 'min_intensity', 'max_intensity', 'ram_limit',
            'verbose',
            'anisotropy_x', 
            'anisotropy_y'
        }

        # Check for unexpected keyword arguments
        invalid_keys = set(kwargs.keys()) - valid_keys
        if invalid_keys:
            print(f"Warning: unexpected keyword argument(s): {', '.join(invalid_keys)}")
        
        # parse kwargs
        features = features
        neighbor_distance = kwargs.get('neighbor_distance', 5)
        pixels_per_micron = kwargs.get('pixels_per_micron', 1.0)
        coarse_gray_depth = kwargs.get('coarse_gray_depth', 256)
        n_feature_calc_threads = kwargs.get('n_feature_calc_threads', 4)
        using_gpu = kwargs.get('using_gpu', -1)
        ibsi = kwargs.get('ibsi', False)
        dynamic_range = kwargs.get('dynamic_range', 10000)
        min_intensity = kwargs.get('min_intensity', 0.0)
        max_intensity = kwargs.get('max_intensity', 1.0)
        preserve_hu = kwargs.get('preserve_hu', False)   # CT/HU: preserve absolute Hounsfield values
        use_physical_spacing = kwargs.get('use_physical_spacing', False)   # opt-in OME PhysicalSize* voxel spacing (3D)
        ram_limit = kwargs.get('ram_limit', -1)
        verb_lvl = kwargs.get ('verbose', 0)
        aniso_x = kwargs.get('anisotropy_x', 1.0)
        aniso_y = kwargs.get('anisotropy_y', 1.0)
        
        if neighbor_distance <= 0:
            raise ValueError("Neighbor distance must be greater than zero.")

        if pixels_per_micron <= 0:
            raise ValueError("Pixels per micron must be greater than zero.")

        if coarse_gray_depth <= 0:
            raise ValueError("Custom number of grayscale levels (parameter coarse_gray_depth, default=256) must be positive.")

        if n_feature_calc_threads < 1:
            raise ValueError("There must be at least one feature calculation thread.")

        if(using_gpu > -1 and n_feature_calc_threads != 1):
            print("Gpu features only support a single thread. Defaulting to one thread.")
            n_feature_calc_threads = 1
            
        if(using_gpu > -1 and not gpu_available_imp(id(self))):
            print("No gpu available.")
            using_gpu = -1

        if verb_lvl < 0:
            raise ValueError ("verbosity must be non-negative")

        if aniso_x <= 0:
            raise ValueError ("anisotropy_x must be positive")

        if aniso_y <= 0:
            raise ValueError ("anisotropy_y must be positive")

        aniso_z = 1.0   # not used in 2D image quality assessment

        initialize_environment(
            id(self),
            2, # 2D
            features,
            neighbor_distance,
            pixels_per_micron,
            coarse_gray_depth,
            n_feature_calc_threads,
            using_gpu,
            ibsi,
            dynamic_range,
            min_intensity,
            max_intensity,
            True,
            ram_limit,
            verb_lvl,
            aniso_x,
            aniso_y,
            aniso_z,
            False,  # merge_labels: 2D-segmented only
            preserve_hu,
            use_physical_spacing)
        
        # list of valid outputs that are used throughout featurize functions
        self._valid_output_types = ['pandas', 'arrowipc', 'parquet']

    def featurize_directory(
        self,
        intensity_dir: str,
        label_dir: Optional[str] = None,
        file_pattern: Optional[str] = ".*",
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
    ):
        r"""Extract image quality features from the images in a directory.

        Pairs each intensity image in `intensity_dir` with the mask image of the
        same file name in `label_dir`. Without `label_dir` each image is one ROI.

        Parameters
        ----------
        intensity_dir : str
            Directory of intensity images.
        label_dir : str, optional
            Directory of mask (label) images. Default None processes each
            intensity image as one ROI covering the whole image.
        file_pattern : str, optional
            Default ".*". Regular expression selecting the files. It must match
            the whole file name, case-sensitively, in both directories.
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        OSError
            If `intensity_dir` or `label_dir` does not exist.
        ValueError
            If `output_type` is not recognized or `file_pattern` is not a valid
            pattern.
        RuntimeError
            If the dataset cannot be read.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import ImageQuality
        >>> imq = ImageQuality(["*ALL_IMQ*"])
        >>> imq.featurize_directory("int").shape
        (2, 15)
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

        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}.')
            
        if (output_type == 'pandas'):
            
            header, string_data, numeric_data = featurize_directory_imp (id(self), intensity_dir, label_dir, file_pattern, output_type, "")

            df = pd.concat(
                [
                    pd.DataFrame(string_data, columns=header[: string_data.shape[1]]),
                    pd.DataFrame(numeric_data, columns=header[string_data.shape[1] :]),
                ],
                axis=1,
            )

            # Labels should always be uint.
            if "ROI_label" in df.columns:
                df["ROI_label"] = df.ROI_label.astype(np.uint32)

            return df
        
        else:
            
            featurize_directory_imp (id(self), intensity_dir, label_dir, file_pattern, output_type, output_path)
            return get_arrow_file_imp (id(self)) # return path to file

    def featurize(
        self,
        intensity_images: np.ndarray,
        label_images: np.ndarray = np.array([[]]),
        intensity_names: list = [],
        label_names: list = [],
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
        
    ):
        r"""Extract image quality features from images held in numpy arrays.

        Parameters
        ----------
        intensity_images : numpy.ndarray
            One 2D image, or a 3D stack of them (first axis = image).
        label_images : numpy.ndarray, optional
            Mask (label) images matching `intensity_images` in shape. When
            omitted, each image is processed as one ROI covering the whole image.
        intensity_names : list of str, optional
            Names reported in the ``intensity_image`` column, one per image.
            Default ``Intensity0``, ``Intensity1``, ...
        label_names : list of str, optional
            Names reported in the ``mask_image`` column, one per image. Default
            ``Segmentation0``, ``Segmentation1``, ...
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        ValueError
            If the arrays are not both 2D or both 3D, their shapes differ, the
            number of names does not match the number of images, or
            `output_type` is not recognized.
        RuntimeError
            If a ROI's estimated footprint reaches `ram_limit`.

        Notes
        -----
        The arrays are cast to ``uint32`` by the backend: fractional values
        are truncated, and unlike :meth:`Nyxus.featurize` negative values are
        not shifted first, so they wrap around to large positive values.
        Shift negative images to a non-negative range before calling.

        Examples
        --------
        >>> import numpy as np
        >>> from nyxus import ImageQuality
        >>> rng = np.random.default_rng(0)
        >>> images = rng.integers(0, 1000, (2, 64, 64)).astype(np.uint16)
        >>> imq = ImageQuality(["FOCUS_SCORE", "SHARPNESS"])
        >>> df = imq.featurize(images, intensity_names=["a", "b"])
        >>> df["intensity_image"].tolist()
        ['a', 'b']
        """
        if (output_type != "" and output_type not in self._valid_output_types):
            raise ValueError(f'Invalid output type: {output_type}. Valid options are: {self._valid_output_types}')
        
        # Handle single_roi        
        if (label_images.size == 0):
            label_images = np.ones_like(intensity_images)

        
        # verify argument types
        if not isinstance(intensity_images, np.ndarray):
            raise ValueError("intensity_images parameter must be numpy.ndarray")

        if not isinstance(label_images, np.ndarray):
            raise ValueError("label_images parameter must be numpy.ndarray")
        
        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}.')
        
        # verify dimensions of images are the same
        if(intensity_images.ndim == 2):
            if(label_images.ndim != 2):
                raise ValueError("Both intensity and label arrays must be the same dimension")
            
            intensity_images = np.array([intensity_images])
            label_images = np.array([label_images])
            
        elif(intensity_images.ndim == 3):
            if(label_images.ndim != 3):
                raise ValueError("Both intensity and label arrays must be the same dimension")
            
        else:
            raise ValueError("Intensity and label arrays must be 2D or 3D")
            
        
        if(intensity_images.shape != label_images.shape):
            raise ValueError("Intensity and label image arrays must have the same number of images with matching dimensions")
        
        if (intensity_names == []):
            int_name = "Intensity"
            
            # Rebind to a fresh list (do NOT mutate the shared default arg)
            intensity_names = [int_name + str(i) for i in range(intensity_images.shape[0])]
        
        
        if (label_names == []):
            seg_name = "Segmentation"
            
            # Rebind to a fresh list (do NOT mutate the shared default arg)
            label_names = [seg_name + str(i) for i in range(label_images.shape[0])]
                
        if (intensity_images.shape[0] != len(intensity_names)):
            raise ValueError("Number of image names must be the same as the number of images.")
        
        if (label_images.shape[0] != len(label_names)):
            raise ValueError("Number of segmentation names must be the same as the number of images.")
        
        if (output_type == 'pandas'):
                
            header, string_data, numeric_data, error_message = featurize_montage_imp (id(self), intensity_images, label_images, intensity_names, label_names, output_type, "")
            
            self.error_message = error_message
            if(error_message != ''):
                print(error_message)

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
            
        else:
            
            ret = featurize_montage_imp (id(self), intensity_images, label_images, intensity_names, label_names, output_type, output_path)
            
            self.error_message = ret[0]
            
            if(self.error_message != ''):
                raise RuntimeError('Error calculating features: ' + error_message[0])
            
            return get_arrow_file_imp (id(self)) # return path to file
                
    def featurize_files (
        self,
        intensity_files: list,
        mask_files: Optional[list] = [],
        single_roi: Optional[bool] = True,
        output_type: Optional[str] = "pandas",
        output_path: Optional[str] = "",
    ):
        r"""Extract image quality features from lists of image files.

        Parameters
        ----------
        intensity_files : list of str
            Paths of intensity images.
        mask_files : list of str, optional
            Paths of mask (label) images, paired with `intensity_files` by
            position. Default []. Must be a list; pass [] rather than None.
        single_roi : bool, optional
            Default True: each intensity image is one ROI covering the whole
            image and `mask_files` is ignored. False pairs `intensity_files`
            with `mask_files`.
        output_type : {"pandas", "arrowipc", "parquet"}, optional
            Output format. Default "pandas" returns a DataFrame; "arrowipc" and
            "parquet" write a file and return its path.
        output_path : str, optional
            Destination of the Arrow IPC or Parquet file; ignored for "pandas".
            Default "" writes ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet``
            in the current working directory. A path with a file name
            (``out/features.arrow``) writes that file; a directory path (``out``)
            writes the default file name inside it. A missing last directory is
            created; if its parent is missing too, the call raises RuntimeError, a
            known defect (`#518 <https://github.com/PolusAI/nyxus/issues/518>`__).

        Returns
        -------
        pandas.DataFrame or str
            For "pandas", a DataFrame with one row per ROI per image: the columns
            ``intensity_image``, ``mask_image``, ``phys_unit``, ``ROI_label``,
            ``t_index``, ``c_index``, ``phys_x``, ``phys_y``, ``phys_z``, then one
            column per feature. Otherwise the path of the written file.

        Raises
        ------
        OSError
            If `intensity_files` is None, or `mask_files` is None while
            `single_roi` is False.
        TypeError
            If `mask_files` is None while `single_roi` is True.
        ValueError
            If `output_type` is not recognized.
        RuntimeError
            If `intensity_files` is empty, a file does not exist, the two
            lists differ in length while `single_roi` is False, or
            featurization fails.

        Examples
        --------
        The file-based examples assume the working directory holds the ``int`` and
        ``seg`` folders of the two-image sample under ``tests/python/data`` in the
        Nyxus repository.

        >>> from nyxus import ImageQuality
        >>> imq = ImageQuality(["*ALL_IMQ*"])
        >>> imq.featurize_files(["int/p0_y1_r1_c0.ome.tif"]).shape
        (1, 15)
        """

        if intensity_files is None:
            raise IOError ("The list of intensity file paths is empty")

        if mask_files is None and not single_roi:
            raise IOError ("The list of segment file paths is empty. Supply mask images or set single_roi to True")
        
        if (output_type not in self._valid_output_types):
            raise  ValueError(f'Invalid output type {output_type}. Valid output types are {self._valid_output_types}')

        if (output_type == 'pandas'):
            
            header, string_data, numeric_data = featurize_fname_lists_imp (id(self), intensity_files, mask_files, single_roi, output_type, "")

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
        
        else:
            
            featurize_fname_lists_imp (id(self), intensity_files, mask_files, single_roi, output_type, output_path)
            return get_arrow_file_imp (id(self))


    def blacklist_roi(self, blacklist:str):
        r"""Set the ROI blacklist.

        Blacklisted ROI labels are skipped during featurization of files
        (:meth:`featurize_directory`, :meth:`featurize_files`) and do not appear in
        the output. Each call replaces the previous blacklist.

        Parameters
        ----------
        blacklist : str
            Either a global list of labels, ``"27,28,30"``, or per-file lists
            separated by semicolons, ``"file1.ome.tif:5,6,7;file2.ome.tif:11,12"``.
            A per-file list names the mask file without its directory; a name
            with a directory part is accepted but matches no file.

        Raises
        ------
        OSError
            If `blacklist` is None or blank.
        RuntimeError
            If `blacklist` cannot be parsed.

        See Also
        --------
        clear_roi_blacklist, roi_blacklist_get_summary

        Examples
        --------
        >>> from nyxus import ImageQuality
        >>> nyx = ImageQuality(["*ALL_IMQ*"])
        >>> nyx.blacklist_roi("27,28,30")
        >>> nyx.roi_blacklist_get_summary()
        'global 27 28 30'
        """
        if blacklist is None:
            raise IOError ("ROI blacklist argument is empty")

        if len(blacklist.strip()) == 0:
            raise IOError ("ROI blacklist argument is non-informative")

        blacklist_roi_imp (id(self), blacklist)

    def clear_roi_blacklist(self):
        r"""Clear the ROI blacklist.

        See Also
        --------
        blacklist_roi

        Examples
        --------
        >>> from nyxus import ImageQuality
        >>> nyx = ImageQuality(["*ALL_IMQ*"])
        >>> nyx.blacklist_roi("5")
        >>> nyx.clear_roi_blacklist()
        >>> nyx.roi_blacklist_get_summary() is None
        True
        """
        clear_roi_blacklist_imp (id(self))

    def roi_blacklist_get_summary (self):
        r"""Describe the ROI blacklist.

        Returns
        -------
        str or None
            Human-readable summary of the blacklist, or None when it is empty.

        See Also
        --------
        blacklist_roi

        Examples
        --------
        >>> from nyxus import ImageQuality
        >>> nyx = ImageQuality(["*ALL_IMQ*"])
        >>> nyx.blacklist_roi("file1.ome.tif:5,6;file2.ome.tif:7")
        >>> nyx.roi_blacklist_get_summary()
        'file1.ome.tif : 5 6 file2.ome.tif : 7'
        """
        s = roi_blacklist_get_summary_imp (id(self))
        s = s.strip()
        if len(s) == 0:
            return None
        return s
    
    def set_gabor_feature_params (self, **kwargs):
        r"""Set the parameters of the GABOR feature.

        Only the parameters passed are changed.

        Parameters
        ----------
        kersize : int, optional
            Side of the filter kernel, in pixels.
        gamma : float, optional
            Aspect ratio of the Gaussian factor.
        sig2lam : float, optional
            Spatial frequency bandwidth (sigma to lambda ratio).
        f0 : float, optional
            Angular frequency of the baseline low-pass filter, in radians per
            pixel; its wavelength is :math:`2\pi/f_0` pixels.
        thold : float, optional
            Lower threshold of the filtered-image to baseline ratio.
        thetas : list of float, optional
            Orientations of the Gaussian, in degrees, conventionally 0-180
            (not enforced), one per entry of `freqs`. Must be a list; a
            comma-separated string is rejected.
        freqs : list of float, optional
            Angular frequencies of the filters' harmonic factor, in radians per
            pixel (wavelength :math:`2\pi/f` pixels), one per entry of `thetas`.
            Must be a list.

        Raises
        ------
        ValueError
            If a keyword is not one of the above, or a value is invalid (for
            example `thetas` and `freqs` of different lengths).
        OSError
            If no keyword is passed.

        Notes
        -----
        Gabor parameters are process-wide: the change applies to every instance,
        and the next :class:`Nyxus` constructor call resets them.

        Examples
        --------
        >>> from nyxus import ImageQuality
        >>> nyx = ImageQuality(["*ALL_IMQ*"])
        >>> nyx.set_gabor_feature_params(kersize=16, gamma=0.1, sig2lam=0.8,
        ...                              f0=0.1, thold=0.025,
        ...                              thetas=[0, 90], freqs=[4, 16])
        >>> nyx.get_params("gabor_thetas", "gabor_freqs")
        {'gabor_freqs': [4.0, 16.0], 'gabor_thetas': [0.0, 90.0]}
        """

        params = [
            'kersize',
            'gamma',
            'sig2lam',
            'f0',
            'thold',
            'thetas',
            'freqs'
        ]
        
        for key in kwargs:
            if key not in params:
                raise ValueError(f"Invalid Gabor parameter {key}. The valid parameters are: {params}")
    
        kersize = str(kwargs.get ('kersize', ""))
        gamma = str(kwargs.get ('gamma', ""))
        sig2lam = str(kwargs.get ('sig2lam', ""))
        f0 = str(kwargs.get ('f0', ""))
        thold = str(kwargs.get ('thold', ""))
        thetas = kwargs.get ('thetas', [])
        freqs = kwargs.get ('freqs', [])

        if len(kersize)==0 and len(gamma)==0 and len(sig2lam)==0 and len(f0)==0 and len(thold)==0 and len(freqs)==0 and len(thetas)==0 :
            raise IOError ("Illegal arguments passed to set_gabor_feature_params()")
        
        if (freqs == []):
            freqs = ""
        else:
            freqs = ",".join(str(i) for i in freqs)

        if (thetas == []):
            thetas = ""
        else:
            thetas = ",".join(str(i) for i in thetas)

        customize_gabor_feature_imp (id(self), kersize, gamma, sig2lam, f0, thetas, thold, freqs)

    
    def set_environment_params (self, **params):
        r"""Change environment parameters of this instance.

        Only the parameters passed are changed, except `verbose` (see Notes).

        Parameters
        ----------
        features : list of str, optional
            Stored and reported by :meth:`get_params`, but the enabled feature
            set does not change (see Notes).
        neighbor_distance : int, optional
            Stored and reported by :meth:`get_params`; no feature currently
            depends on it.
        pixels_per_micron : float, optional
            Stored and reported by :meth:`get_params`; no feature currently
            depends on it.
        coarse_gray_depth : int, optional
            Number of grey levels for texture features (see Notes).
        n_feature_calc_threads : int, optional
            Number of feature calculation threads.
        use_gpu_device : int, optional
            Accepted and ignored.
        verbose : int, optional
            Level of diagnostic output.
        dynamic_range : float, optional
            Number of grey levels a floating-point image is rescaled to.
        min_intensity : float, optional
            Intensity mapped to the lowest grey level. Negative values are
            ignored.
        max_intensity : float, optional
            Intensity mapped to `dynamic_range`. Negative values are ignored.

        Raises
        ------
        ValueError
            If a keyword is not one of the above.

        Notes
        -----
        `verbose` is applied on every call and falls back to 0 when omitted.

        The enabled feature set and the per-feature settings derived from
        `coarse_gray_depth` are fixed when the instance is constructed. Changing
        `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
        :meth:`get_params` reports but not the features computed by this
        instance; construct a new instance to change them.

        Examples
        --------
        >>> from nyxus import ImageQuality
        >>> nyx = ImageQuality(["*ALL_IMQ*"])
        >>> nyx.set_environment_params(neighbor_distance=3, pixels_per_micron=2.0)
        >>> nyx.get_params("neighbor_distance", "pixels_per_micron")
        {'neighbor_distance': 3, 'pixels_per_micron': 2.0}
        """
        
        valid_params = [
            'features',
            'neighbor_distance',
            'pixels_per_micron',
            'coarse_gray_depth',
            'n_feature_calc_threads',
            'use_gpu_device',
            'verbose',
            'dynamic_range',
            'min_intensity',
            'max_intensity'
        ]
        
        for key in params:
            if key not in valid_params:
                raise ValueError(f'Invalid environment parameter {key}. Value parameters are {params}')
        
        features = params.get('features', [])
        neighbor_distance = params.get ('neighbor_distance', -1)
        pixels_per_micron = params.get ('pixels_per_micron', -1)
        coarse_gray_depth = params.get ('coarse_gray_depth', 0)
        n_reduce_threads = params.get ('n_feature_calc_threads', 0)
        use_gpu_device = params.get ('use_gpu_device', -1)
        verb_lvl = params.get ('verbose', 0)
        dynamic_range = params.get('dynamic_range', -1)
        min_intensity = params.get('min_intensity', -1)
        max_intensity = params.get('max_intensity', -1)
        ram_limit = -1 # no limit
        
        set_environment_params_imp (id(self),
                                   features,
                                   neighbor_distance, 
                                   pixels_per_micron,
                                   coarse_gray_depth,
                                   n_reduce_threads,
                                   use_gpu_device,
                                   dynamic_range,
                                   min_intensity,
                                   max_intensity,
                                   ram_limit,
                                   verb_lvl)
        
    def set_params(self, **params):
        r"""Change parameters of this instance.

        Accepts the keywords of :meth:`set_environment_params` except
        `use_gpu_device`, plus `ibsi` and the ``gabor_*`` keywords, which are
        forwarded to :meth:`set_gabor_feature_params` without the prefix.

        Parameters
        ----------
        **params
            Any of `features`, `neighbor_distance`, `pixels_per_micron`,
            `coarse_gray_depth`, `n_feature_calc_threads`, `ibsi` (bool),
            `dynamic_range`, `min_intensity`, `max_intensity`, `verbose`,
            `gabor_kersize`, `gabor_gamma`, `gabor_sig2lam`, `gabor_f0`,
            `gabor_thold`, `gabor_thetas` (list) and `gabor_freqs` (list).

        Raises
        ------
        ValueError
            If a keyword is not recognized. ``using_gpu`` passes the first check
            but is rejected by :meth:`set_environment_params`.

        Notes
        -----
        The enabled feature set and the per-feature settings derived from
        `coarse_gray_depth` are fixed when the instance is constructed. Changing
        `features`, `coarse_gray_depth` or `ibsi` afterwards updates what
        :meth:`get_params` reports but not the features computed by this
        instance; construct a new instance to change them.

        Examples
        --------
        >>> from nyxus import ImageQuality
        >>> imq = ImageQuality(["*ALL_IMQ*"])
        >>> imq.set_params(neighbor_distance=3)
        >>> imq.get_params("neighbor_distance")
        {'neighbor_distance': 3}
        """
        
        available_environment_params = [
            "features",
            "neighbor_distance",
            "pixels_per_micron",
            "coarse_gray_depth",
            "n_feature_calc_threads",
            "using_gpu",
            "ibsi",
            "dynamic_range",
            "min_intensity",
            "max_intensity",
            "verbose"
        ]
        
        environment_params = {}
        
        gabor_params = {}
        
        
        for key, value in params.items():
            if key.startswith("gabor_"):
                gabor_params[key[len("gabor_"):]] = value
            
            elif (key == "ibsi"):
                set_if_ibsi_imp (id(self), value)
            
            else:
                if (key not in available_environment_params):
                    raise ValueError(f"Invalid parameter {key}")
                else:
                    environment_params[key] = value
                
                
        if (len(gabor_params) > 0):
            self.set_gabor_feature_params(**gabor_params)
        
        if (len(environment_params) > 0):
            self.set_environment_params(**environment_params)
    
    def get_params(self, *args):
        r"""Return parameters of this instance.

        Parameters
        ----------
        *args : str
            Names of the parameters to return. With no names, all are returned.
            Unknown names are ignored. Available names: ``features``,
            ``neighbor_distance``, ``pixels_per_micron``, ``coarse_gray_depth``,
            ``n_feature_calc_threads``, ``ibsi``, ``dynamic_range``,
            ``min_intensity``, ``max_intensity``, ``ram_limit`` (megabytes),
            ``gabor_kersize``, ``gabor_gamma``, ``gabor_sig2lam``, ``gabor_f0``,
            ``gabor_thold``, ``gabor_thetas`` and ``gabor_freqs``.

        Returns
        -------
        dict
            Parameter name to value. ``ibsi`` is reported as 0 or 1. The GPU
            settings and `verbose` are not reported.

        Examples
        --------
        >>> from nyxus import ImageQuality
        >>> nyx = ImageQuality(["*ALL_IMQ*"])
        >>> nyx.get_params("coarse_gray_depth", "neighbor_distance", "ibsi")
        {'coarse_gray_depth': 256, 'ibsi': 0, 'neighbor_distance': 5}
        """
        vars = list(args)
        
        return get_params_imp (id(self), vars)


class Nested:
    r"""Relate ROIs of two mask channels as parents and children.

    :meth:`find_relations` relates each child ROI to every parent ROI whose
    bounding box contains the child's bounding box; :meth:`featurize` joins
    child features to those relations and optionally aggregates them.

    Parameters
    ----------
    aggregate : list, optional
        Aggregations applied by :meth:`featurize`, in any form accepted by
        :meth:`pandas.DataFrame.aggregate`: names such as ``"mean"``,
        ``"min"``, ``"max"``, ``"std"``, ``"count"``, functions, or
        ``(name, function)`` tuples such as
        ``("nanmean", lambda x: np.nanmean(x))``. Default [] (no aggregation).

    Examples
    --------
    The file-based examples assume the working directory holds the ``int`` and
    ``seg`` folders of the two-image sample under ``tests/python/data`` in the
    Nyxus repository. In it, ``*_c1`` masks hold the parents and ``*_c0`` masks
    the children.

    >>> from nyxus import Nested, Nyxus
    >>> nyx = Nyxus(["MEAN"])
    >>> child_features = nyx.featurize_directory("int", "seg",
    ...                                          file_pattern=r".*_c0\.ome\.tif")
    >>> child_features = child_features.rename(columns={"ROI_label": "label"})
    >>> nn = Nested(aggregate=["mean", "max"])
    >>> rels = nn.find_relations("seg", r".*_c1\.ome\.tif", r".*_c0\.ome\.tif")
    >>> list(rels.columns)
    ['Image', 'Parent_Label', 'Child_Label']
    >>> df = nn.featurize(rels, child_features)
    >>> bool(df[("MEAN", "mean")].notna().all())
    True
    """

    def __del__(self):
        # The backend keys this instance's settings on id(self), which is the object's
        # address -- and CPython hands that same address to the next object it allocates.
        # An entry left behind after this object dies is inherited wholesale by that next
        # instance, which would then silently start with our ram_limit, feature list, gpu
        # flag or ibsi mode. Drop it here rather than leave the address dirty.
        try:
            release_environment_imp(id(self))
        except Exception:
            pass    # interpreter shutdown may already have torn the backend module down

    def __init__(self, aggregate: Optional[list] = []):
        
        self.aggregate = aggregate
    
    
    def find_relations(
        self,
        label_dir: str,
        parent_file_pattern: str, 
        child_file_pattern: str):
    
        r"""Find parent-child relations between two sets of mask images.

        Parent and child masks are read from the same directory and paired in
        file order; each child ROI is related to every parent ROI whose bounding
        box contains the child's bounding box.

        Parameters
        ----------
        label_dir : str
            Directory holding both the parent and the child mask images.
        parent_file_pattern : str
            Regular expression selecting the parent masks, e.g.
            ``r".*_c1\.ome\.tif"``.
        child_file_pattern : str
            Regular expression selecting the child masks, e.g.
            ``r".*_c0\.ome\.tif"``.

        Returns
        -------
        pandas.DataFrame
            One row per relation, with the columns ``Image`` (path of the parent
            mask), ``Parent_Label`` and ``Child_Label``.

        Raises
        ------
        OSError
            If `label_dir` does not exist.
        ValueError
            If a pattern is not valid.
        RuntimeError
            If no parent or no child files match, their counts differ, or the
            hierarchy cannot be computed.

        Notes
        -----
        Progress information is printed to standard output.

        Examples
        --------
        See :class:`Nested`.
        """

        if not os.path.exists(label_dir):
            raise IOError (f"Provided label image directory '{label_dir}' does not exist.")

        header, string_data, numeric_data = findrelations_imp (id(self), label_dir, parent_file_pattern, child_file_pattern)

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
	
    def featurize(self, parent_child_map: pd.DataFrame, child_features: pd.DataFrame):
        r"""Join child ROI features to parent-child relations.

        Merges `parent_child_map` with `child_features` on ``Child_Label`` ==
        ``label``. With aggregations set in the constructor, the joined feature
        columns are grouped by ``label`` and aggregated. Without them, the result
        is pivoted with ``label`` as the index and ``Child_Label`` as the columns.

        Parameters
        ----------
        parent_child_map : pandas.DataFrame
            Relations returned by :meth:`find_relations`.
        child_features : pandas.DataFrame
            Features of the child ROIs, e.g. from :meth:`Nyxus.featurize_directory`.
            Must have a ``label`` column holding the child ROI label; Nyxus names
            that column ``ROI_label``, so rename it first.

        Returns
        -------
        pandas.DataFrame
            Aggregated features, one row per ``label``, with a
            ``(feature, aggregation)`` column for each pair when aggregations are
            set; otherwise the pivoted table.

        Raises
        ------
        KeyError
            If `child_features` has no ``label`` column.

        Notes
        -----
        The aggregated and pivoted columns are every joined column after the
        sixth, which on Nyxus output includes ``label`` itself. Without
        aggregations the pivot therefore raises ValueError on Nyxus output.
        ``label`` holds the child label, so the grouping is by child, not by
        parent.

        Examples
        --------
        See :class:`Nested`.
        """
        
        joined_df = parent_child_map.merge(child_features, left_on=['Child_Label'], right_on=['label'])
        
        feature_columns = list(joined_df.columns)[6:]

        if(self.aggregate == []):
            joined_df.apply(lambda x: x) # convert group_by object to dataframe
        
            return joined_df.pivot_table(index='label', columns='Child_Label', values=feature_columns)
        
        agg_features = {}
        for col in feature_columns:
            agg_features[col] = self.aggregate
            
        return joined_df.groupby(by='label').agg(agg_features)
    
