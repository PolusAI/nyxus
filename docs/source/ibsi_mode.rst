.. _ibsi-mode:

IBSI mode
=========

IBSI mode makes the texture families follow the definitions of the Image Biomarker
Standardisation Initiative (IBSI). It also makes the intensity histogram family (``IH_*``)
available. The mode is off by default.

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         nyx = Nyxus(["*ALL*"], ibsi=True)

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --ibsi=true --intDir=/data/int --segDir=/data/seg --outDir=/data/out \
                 --outputType=singlecsv

On the command line ``true``, ``1`` and ``on`` turn the mode on, in any case. Any other value
turns it off without an error.


What changes
------------

IBSI mode changes three things: grey-level binning, the layout of some texture matrices, and the
set of available features.

**No grey-level binning.** By default the texture families first bin the ROI's intensities into
``--coarseGrayDepth`` grey levels (64 by default). In IBSI mode they use the intensities as they
are. The 2D families affected are:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Family
     - Default
     - IBSI mode
   * - :doc:`GLCM <Math/f_glcm>`
     - 64 MATLAB-style levels. Each neighbour pair is counted in one order only.
     - No binning. The matrix has one row and column per grey value from 1 to the largest
       intensity in the ROI's bounding box. Each pair is counted in both orders.
   * - :doc:`GLRLM <Math/f_glrlm>`, :doc:`GLSZM <Math/f_glszm>`, :doc:`GLDM <Math/f_gldm>`
     - 64 levels. One matrix row per grey level that occurs in the ROI.
     - No binning. One matrix row per grey value from 1 to the largest one, so absent grey
       values give empty rows.
   * - :doc:`NGTDM <Math/f_ngtdm>`
     - 64 levels.
     - No binning; the rows are indexed by grey value.
   * - :doc:`GLDZM <Math/f_gldzm>`
     - 64 levels.
     - No binning.
   * - :doc:`NGLDM <Math/f_ngldm>`
     - Its own conversion to ``--coarseGrayDepth`` levels.
     - No conversion.

One binning step remains in IBSI mode: before building the 2D GLCM, Nyxus bins the ROI's minimum
and maximum to ``--coarseGrayDepth`` levels and when they fall into one level every GLCM feature returns 0, whatever
``--noval`` is (a known defect, `#511 <https://github.com/PolusAI/nyxus/issues/511>`__), so a ROI whose minimum is at least 63/64 of its maximum gets 0
in every GLCM column (see :ref:`glcm-possible-issues`).

The 3D twins of these families skip binning in the same way; by default they bin into 64 levels
like the 2D families (see :ref:`conventions-grey-levels`).

**The intensity histogram family becomes available.** The 46
:doc:`IH_* <Math/f_intensity_histogram>` features are computed only in IBSI mode, from a
histogram with ``|--coarseGrayDepth|`` bins (24 at 0). In IBSI mode ``*ALL*`` includes
them. Without IBSI mode, a request for ``IH_*`` features or ``*ALL_IH*`` gives no ``IH_*``
columns. The command line then prints a warning; Python prints it only with ``verbose`` above 0.

**Unchanged.** IBSI mode does not change the intensity, morphology, moment, Gabor, Zernike,
erosion, fractal, neighbour and image quality features. ``ENTROPY``, ``UNIFORMITY`` and
``HISTOGRAM`` keep binning into ``--coarseGrayDepth`` bins.

The family pages compare the IBSI-mode values with IBSI, PyRadiomics and MIRP. Each texture page
has a tip on reproducing those tools, such as the one in :doc:`Math/f_glcm`.


Example
-------

The script computes every 2D feature of two images, once with the default settings and once in
IBSI mode, and lists the families whose values differ:

.. literalinclude:: ../examples/ibsi_mode.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

   756 802 ['IH_MEAN_VAL', 'IH_VARIANCE_VAL', 'IH_SKEWNESS_VAL']
   ['GLCM', 'GLDM', 'GLDZM', 'GLRLM', 'GLSZM', 'NGLDM', 'NGTDM']
   [20.789930555555557, 680.5772479062924, 0.0010357191235226917]
   [20.789930555555557, 266.70620661687036, 0.0016546934699797648]

IBSI mode adds 46 columns. ``MEAN`` is the same in both runs. ``GLCM_CONTRAST_AVE`` drops
from 680.6 to 266.7, because the default run first stretches grey levels 1 to 40 over 64 levels.
On a 16-bit microscopy image with 66 ROIs, IBSI mode changed 300 of the 747 shared feature
columns. All of them belonged to the seven texture families; only ``NGLDM_DCP`` stayed the same.


Edge cases
----------

.. warning::

   **Images with large intensities.** Without binning, the 2D GLCM matrix has one row and one
   column per grey value up to the ROI's largest intensity. A ROI whose largest value is 50 000
   needs a 50 000 × 50 000 matrix of doubles, about 20 GB. When that allocation fails, Nyxus
   prints ``cannot allocate a ... matrix`` and reports 0 for every GLCM feature of the image. On a
   16-bit image whose ROI maxima ran from 19 552 to 65 519, every GLCM value of all 66 ROIs was 0.
   Rescale such images to a small integer range before using IBSI mode, or leave it off.

.. warning::

   **Set IBSI mode when you create the object.** ``set_params(ibsi=True)`` on an existing
   ``Nyxus`` object changes what ``get_params`` reports but not the results: the texture values
   stay binned, and ``IH_*`` columns do not appear. Pass ``ibsi=True`` to the constructor.
