Neighbourhood grey-tone difference matrix, 3D (3NGTDM)
======================================================

The 3D NGTDM compares each voxel with the average of the voxels around it. For every grey level
the matrix holds how often the level occurs and how far, in total, its voxels differ from their
neighbourhood averages. The five features and their formulas are those of the 2D family, defined on
the :doc:`2D NGTDM page <f_ngtdm>`; this page covers what is specific to volumes.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3NGTDM_*``: 5 features, one value per ROI
   * - **Group**
     - ``*3D_NGTDM*``; also part of ``*3D_ALL_TEXTURE*`` and ``*3D_ALL*``
   * - **Dimensions**
     - 3D. The 2D family, ``NGTDM_*``, is on the :doc:`2D NGTDM page <f_ngtdm>`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes. The streamed path builds the same matrix as the in-memory one.
       When one read of the file returns the whole volume (NIfTI, or OME-Zarr chunked over all of
       Z), an oversized ROI stops the run with an error (see :doc:`/large_images`).
   * - **IBSI**
     - All 5 features have an IBSI identifier. IBSI calls a matrix built from the whole volume the
       3D aggregation (KOBO), which is what Nyxus computes.

.. tip::

   The grey levels of this family are set by the ``3ngtdm/greydepth`` metaparameter, not by
   ``--coarseGrayDepth``. Its default, 0, uses the raw intensities. With that default, a ROI that
   fills its bounding box and whose lowest intensity is 0 or 1 reproduces PyRadiomics with
   ``binWidth=1``; with a higher lowest intensity, all features but busyness do. Voxels of the
   bounding box that lie outside the mask still take part in the matrix (see
   :ref:`ngtdm-3d-possible-issues`).


How the matrix is built
-----------------------

Nyxus copies the ROI's bounding box into a cube, with the voxels outside the mask set to 0, and bins
the cube's intensities into grey levels. When the lowest grey level in the cube is 0, every voxel is
raised by one level, so the voxels outside the mask become grey level 1.

Nyxus then visits every voxel of the cube whose grey level is not the background level: 1 with
MATLAB-style binning (a positive ``3ngtdm/greydepth``), 0 otherwise. Its neighbourhood is every
voxel of the cube within a Chebyshev distance :math:`\delta` of it, the voxel itself excluded. With
the default :math:`\delta = 1` that is the 26 voxels of the surrounding 3 × 3 × 3 block. All of
these neighbours are averaged, whatever their grey level and whether or not they are in the mask;
only the neighbours outside the cube are left out. The voxel adds 1 to :math:`n_i` for its grey
level :math:`g_i`, and adds :math:`|g_i - \bar{A}|` to :math:`s_i`, where :math:`\bar{A}` is that
average.

- **Neighbourhood:** the :math:`(2\delta+1)^3 - 1` voxels within Chebyshev distance :math:`\delta`,
  set by ``3ngtdm/radius`` (1 by default). Neighbours outside the bounding box are left out, so a
  corner voxel of the box has 7 neighbours at :math:`\delta = 1`, an edge voxel 11, a face voxel 17
  and an inner voxel 26.
- **Directions:** none. One matrix is built from the whole volume, and each feature has one value.
- **Which voxels count:** the grey level decides, not the ROI mask. Voxels of the bounding box
  outside the mask are visited as voxels of the lowest grey level, except with MATLAB-style binning,
  and are always averaged into their neighbours' :math:`\bar{A}`. A ROI voxel of intensity 0 is
  treated the same way as one outside the mask.
- **Grey levels:** set by ``3ngtdm/greydepth``, described under `Settings`_. In IBSI mode the
  intensities are not binned, the matrix has a row for every integer from 0 to the ROI's largest
  intensity, and the levels are always raised by one.
- **Probabilities:** :math:`p_i = n_i / \sum_k n_k`. Every visited voxel has at least one neighbour
  in a cube of more than one voxel, so :math:`\sum_k n_k` is the number of visited voxels.


Worked example
^^^^^^^^^^^^^^

Take a 3 × 3 × 3 ROI that fills its bounding box and already has four grey levels. Its planes
:math:`z = 0, 1, 2` are

.. math::

   \begin{bmatrix} 1 & 2 & 2\\ 3 & 1 & 4\\ 2 & 2 & 3 \end{bmatrix},\qquad
   \begin{bmatrix} 4 & 3 & 1\\ 2 & 4 & 3\\ 1 & 3 & 2 \end{bmatrix},\qquad
   \begin{bmatrix} 2 & 1 & 3\\ 4 & 2 & 1\\ 3 & 4 & 4 \end{bmatrix}

With the default settings the grey levels are the intensities. All 27 voxels are visited. Grey
level 1 has six voxels: one corner, with 7 neighbours averaging :math:`19/7`; one face centre, with
17 neighbours averaging :math:`42/17`; and four edge voxels, with 11 neighbours averaging
:math:`26/11`, :math:`30/11`, :math:`29/11` and :math:`30/11`. So

.. math::

   s_1 = \left(\tfrac{19}{7} - 1\right) + \left(\tfrac{42}{17} - 1\right)
         + \tfrac{26 + 30 + 29 + 30}{11} - 4 = \tfrac{12618}{1309} = 9.6394

The matrix is below. Its rows are grey levels 1 to 4 and its columns are :math:`n_i`, :math:`p_i`
and :math:`s_i`.

.. math::

   \begin{bmatrix}
   6 & 0.2222 & 9.6394\\
   8 & 0.2963 & 4.7410\\
   7 & 0.2593 & 3.5424\\
   6 & 0.2222 & 8.9535 \end{bmatrix},
   \qquad \sum_i n_i = 27, \qquad N_{g,p} = 4, \qquad \sum_i s_i = 26.876

The exact values are :math:`s_2 = 6206/1309`, :math:`s_3 = 4637/1309` and
:math:`s_4 = 17925/2002`. The *Example* value of each feature in the :ref:`index <ngtdm-3d-index>` is
what Nyxus returns for this ROI with the default settings; PyRadiomics returns the same five values.
The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/ngtdm_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_NGTDM*" \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

The symbols are those of the :doc:`2D page <f_ngtdm>`, with voxels in place of pixels. The
differences are:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`g_i`
     - Grey level of row :math:`i`, after the raise by one when the cube's lowest level is 0. In IBSI
       mode this is always the intensity plus one.
   * - :math:`\bar{A}`
     - Mean grey level of all the voxels of the cube within distance :math:`\delta`, in the mask or
       not.
   * - :math:`n_i`
     - Number of visited voxels of grey level :math:`g_i`, including voxels of the bounding box
       outside the mask.
   * - :math:`N_{v,p}`
     - Number of visited voxels whose :math:`\bar{A}` is positive, which is every visited voxel.
       Complexity divides by it.
   * - :math:`N_{g,p}`
     - Number of distinct grey levels in the cube, the level of the voxels outside the mask
       included.

Sums over :math:`i` and :math:`j` run over the rows of the matrix. Rows with :math:`p_i = 0` add
nothing. Every feature uses the grey level :math:`g_i`, never the row position.


Settings
--------

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Grey levels
     - not available
     - ``set_metaparam("3ngtdm/greydepth=N")``
     - 0 (default): no binning, the grey levels are the intensities. Positive: :math:`N` MATLAB-style
       bins from 0 to the ROI maximum, with intensity 0 as level 1, the background level. Negative:
       :math:`|N|` PyRadiomics-style bins over the ROI's range, with intensity 0 as level 0.
   * - Distance
     - not available
     - ``set_metaparam("3ngtdm/radius=N")``
     - :math:`\delta`, the Chebyshev radius of the neighbourhood, a positive integer; 1 by default.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - Overrides ``3ngtdm/greydepth`` with 0, and gives the matrix a row for every integer from 0 to
       the ROI maximum.

``--coarseGrayDepth`` (``coarse_gray_depth``) has no effect on this family: on the worked example
``coarse_gray_depth=8`` returns the default values. ``--pixelDistance`` does not apply either. An
unrecognised ``3ngtdm/...`` parameter is rejected with an error. All options are listed in the
:ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``3NGTDM_COARSENESS``,
``3NGTDM_CONTRAST`` and so on. There are no per-direction columns and no ``_AVE`` columns.


.. _ngtdm-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example with the default settings.

.. include:: ngtdm_3d_index.inc


Features
--------

Each feature is the 2D formula evaluated on the 3D matrix. The links lead to the definitions.

.. _feature-3ngtdm-busyness:

Busyness
^^^^^^^^

:Code: ``3NGTDM_BUSYNESS``
:Range: :math:`[0, \infty)`, dimensionless

Defined as :ref:`NGTDM_BUSYNESS <feature-ngtdm-busyness>`. It is 0 when :math:`N_{g,p} = 1` or when
only one grey level has :math:`p_i > 0`.

**Possible issues.** The only feature that depends on the grey levels themselves rather than on
their differences, so it changes when all levels shift by the same amount. On the worked example it
is 1.477 with the default settings, which PyRadiomics also returns, and 1.502 in IBSI mode, where
the levels are raised by one. The other four features are the same in both modes. PyRadiomics
numbers its bins from 1 at the ROI's lowest intensity, while Nyxus keeps the intensities: adding 9
to every voxel of the worked example leaves PyRadiomics at 1.477 but gives 0.9682 in Nyxus, and
0.8984 in IBSI mode.


.. _feature-3ngtdm-coarseness:

Coarseness
^^^^^^^^^^

:Code: ``3NGTDM_COARSENESS``
:Range: :math:`(0, \infty)`, inverse grey levels

Defined as :ref:`NGTDM_COARSENESS <feature-ngtdm-coarseness>`. When every visited voxel equals its
neighbourhood average the division gives infinity and the output is the ``--noval`` substitute,
0 by default (``-0.0`` in Arrow and Parquet files; see :ref:`conventions-missing`), where IBSI
defines :math:`10^6`.


.. _feature-3ngtdm-complexity:

Complexity
^^^^^^^^^^

:Code: ``3NGTDM_COMPLEXITY``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`NGTDM_COMPLEXITY <feature-ngtdm-complexity>`, with :math:`N_{v,p}` the number of
visited voxels.


.. _feature-3ngtdm-contrast:

Contrast
^^^^^^^^

:Code: ``3NGTDM_CONTRAST``
:Range: :math:`[0, \infty)`, cubed grey levels

Defined as :ref:`NGTDM_CONTRAST <feature-ngtdm-contrast>`, with :math:`N_{g,p}` the number of
distinct grey levels in the bounding box. When the box holds voxels outside the mask, their level
is one of them.


.. _feature-3ngtdm-strength:

Strength
^^^^^^^^

:Code: ``3NGTDM_STRENGTH``
:Range: :math:`[0, \infty)`, grey levels

Defined as :ref:`NGTDM_STRENGTH <feature-ngtdm-strength>`. When :math:`\sum_i s_i = 0` the division
is 0/0 and the output is the ``--noval`` substitute, 0 by default, as in IBSI (``-0.0`` in Arrow
and Parquet files).


.. _ngtdm-3d-possible-issues:

Possible issues
---------------

Voxels outside the mask take part
   IBSI and PyRadiomics use only ROI voxels, as centres and as neighbours. Nyxus uses the whole
   bounding box. With the default settings and in IBSI mode the voxels outside the mask become grey
   level 1: they are visited as voxels of that level, are counted in :math:`N_{g,p}`, and are
   averaged into their neighbours' :math:`\bar{A}`. Leaving the eight corners of the worked example
   out of the mask gives coarseness 0.1015, contrast 0.3376, busyness 1.663, complexity 15.68 and
   strength 1.024 in both modes. PyRadiomics gives 0.1843, 0.2407, 1.097, 6.190 and 0.9691. With
   MATLAB-style binning those voxels are not visited, since level 1 is the background level, but
   they are still averaged into the neighbourhoods. ROIs that fill their bounding box are not
   affected.

ROI voxels of intensity 0
   A ROI voxel of intensity 0 cannot be told apart from a voxel outside the mask. In IBSI mode and
   with the default settings it becomes grey level 1 and is counted. With MATLAB-style binning it
   is level 1, the background level, and is not visited; so are ROI voxels below
   :math:`x_{\max} / N`.

Grey-level values depend on the binning
   Every feature except busyness scales with the grey-level spacing. On the worked example
   ``3ngtdm/greydepth=64`` maps the levels 1 to 4 to 17, 33, 49 and 64, and gives coarseness
   0.009831, contrast 734.1, busyness 1.485, complexity 1294 and strength 11.13. Compare values
   only between runs with the same ``3ngtdm/greydepth``.

The neighbourhood radius
   A larger :math:`\delta` averages over more voxels and smooths :math:`\bar{A}`. On the worked
   example ``3ngtdm/radius=2`` gives coarseness 0.1583, contrast 0.1860, busyness 1.445,
   complexity 5.173 and strength 0.7222. In PyRadiomics the neighbourhood of radius 2 is
   ``distances=[1, 2]``, not ``[2]``, which names one shell only.

Constant ROI
   When all of a ROI's voxels have one intensity and the ROI fills its bounding box, the binned
   volume has a single grey level. With the default settings, MATLAB-style and PyRadiomics-style
   binning, every feature returns the ``--noval`` substitute (0 by default). In IBSI mode the
   matrix is built, since its rows run from 0 to the intensity; coarseness and strength return the
   ``--noval`` substitute (``-0.0`` in Arrow and Parquet files) and the other three return 0. IBSI
   gives :math:`10^6` for coarseness. A constant ROI that does not fill its bounding box gains the
   voxels outside the mask as a second level, and its features are computed: a constant
   3 × 3 × 3 block of intensity 7 without its eight corners gives coarseness 0.02793, contrast 31.52, busyness
   3.357, complexity 18.57 and strength 1.176.

A single-voxel ROI
   The voxel has no neighbour and the matrix has one level, so every feature returns the
   ``--noval`` substitute.

Run time with large intensities in IBSI mode
   In IBSI mode the matrix has one row per integer from 0 to the ROI's largest intensity, and
   contrast, busyness, complexity and strength each loop over every pair of rows, empty or not;
   all five are computed whichever ones are selected. A 6 × 6 × 6 ROI takes 0.04 s with
   intensities up to 4095, 0.31 s up to 16383 and 7.1 s up to 65535. With the default settings the
   matrix has a row only for each intensity present, and the same ROI takes 0.02 s.


Validation
----------

The test suite checks every 3D NGTDM feature against PyRadiomics 3.0.1 on a 4 × 4 × 3 phantom whose
ROI fills its bounding box, with ``3ngtdm/greydepth=0`` and PyRadiomics' ``binWidth=1``, at
``3ngtdm/radius`` 1 and 2. All five agree to a relative 10\ :sup:`-9`, with no difference measured,
and the matrix itself (:math:`n_i`, :math:`p_i`, :math:`s_i` for each grey level) is checked
against PyRadiomics' too. Neither configuration has voxels outside the mask or uses binning, so the
behaviour described under `Possible issues`_ is not compared with a reference tool. MATLAB-style
binning is covered by regression tests only. Every Nyxus feature value quoted on this page is
pinned by the worked-example script; the PyRadiomics values and the run times are not.


References
----------

1. Amadasun M, King R. *Textural features corresponding to textural properties.* IEEE Transactions
   on Systems, Man, and Cybernetics. 1989;19(5):1264–1274. doi:10.1109/21.44046
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107. (PyRadiomics)
