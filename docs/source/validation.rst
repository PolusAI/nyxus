.. _validation:

How features are validated
==========================

Nyxus checks its feature values against independent implementations wherever one exists. Each
feature carries a validation status, which the :doc:`featurelist` shows in its *Validation*
column. This page explains what the statuses mean and where they come from.


Four kinds of test
------------------

The test suite separates its tests into four kinds, each in its own files:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Kind
     - What it checks
     - Claims correctness?
   * - Oracle
     - Nyxus's value equals the value of an external tool, or of a closed-form calculation, on
       the same input and matched settings.
     - Yes. Only oracle tests can make a feature *vetted*.
   * - Regression
     - Nyxus's value equals a value pinned earlier, so any change is noticed.
     - No. It guards against drift.
   * - Invariant
     - The value obeys a required relation, such as two Nyxus code paths giving the same
       result.
     - Partly.
   * - Mechanics
     - The plumbing works: option handling, feature gating, defaults, input and output.
     - No.

An oracle test runs at the settings where the reference tool computes the same quantity. Many
texture features are vetted in IBSI mode, for example, and pinned by regression tests at the
default settings. The *Validation* section at the end of each family page names the settings,
the reference tools and the tolerance.


The coverage registry
---------------------

The file ``tests/vetting/oracle_coverage.csv`` in the source repository records every assertion.
It has one row per feature and settings recipe, and usually one per oracle; for the 2D GLCM and
GLRLM families it records one oracle per feature even where the test suite checks two, so the
feature list names only one of them. The main columns are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Content
   * - ``dim``, ``family``, ``feature``
     - Which feature the row covers: ``2D``, ``3D`` or ``IMQ``, and its family and code.
   * - ``status``
     - ``vetted``, ``invariant``, ``regression`` or ``untested``.
   * - ``oracle``
     - The reference tool of a ``vetted`` row; empty for the other statuses.
   * - ``config_recipe``
     - The settings at which the assertion runs.
   * - ``tolerance``
     - The allowed difference, for example ``rel=1e-9`` or ``abs=1e-9``.
   * - ``test_name``, ``benchmark``
     - The test case and the input image the assertion uses.

The feature list rolls the rows up per feature and shows the best status:

* **vetted** -- at least one oracle test checks the value. The column names the tools.
* **invariant** -- a test checks a relation the value must satisfy, but no oracle checks it.
* **regression** -- a test pins the current value, but no independent implementation confirms
  it.
* **untested** -- no test checks the value.

Continuous integration keeps the registry honest. It checks that the test cases and benchmark
images the rows name exist, and that a ``vetted`` row names an oracle. It also checks that the
test files assert what the registry claims, and that the feature list matches the registry.


Reference tools
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Oracle
     - Used for
   * - IBSI
     - Published reference values on the IBSI digital phantom: 2D intensity, intensity histogram,
       GLCM, GLRLM, GLSZM, GLDZM, GLDM (as IBSI's NGLDM), NGLDM and NGTDM. The registry records
       IBSI rows for intensity, GLCM and GLRLM only.
   * - PyRadiomics
     - Intensity, GLCM, GLRLM and GLDM in 2D and 3D; GLSZM, NGTDM and shape in 3D.
   * - MIRP
     - GLCM, GLSZM, GLDZM, NGLDM, NGTDM and the intensity histogram in 2D; GLDZM, NGLDM and shape
       in 3D.
   * - MATLAB
     - Intensity statistics and shape measurements in 2D and 3D.
   * - scikit-image
     - Moments, Gabor filters and shape measurements.
   * - CellProfiler
     - Shape, neighbour and image quality measurements.
   * - IMEA, FracLac, OpenCV
     - Feret and other diameters, fractal dimension, and image quality measurements.
   * - analytic
     - Values computed in closed form on purpose-built images.

Tolerances range from exact agreement (``exact`` or ``abs=0``), or a relative 10\ :sup:`-9` to
10\ :sup:`-12` where Nyxus and the tool compute the same arithmetic, to a few percent, where the
tool uses a different discretization.


Known deviations
----------------

Where Nyxus deliberately or knowingly computes something other than a reference tool, the
family page says so in its *Possible issues* section, with the size of the difference. The main
ones are:

* Morphology: the minimum enclosing circle truncates chord midpoints, a known defect
  (`#498 <https://github.com/PolusAI/nyxus/issues/498>`__), and the Feret statistics count one
  direction twice, a known defect (`#500 <https://github.com/PolusAI/nyxus/issues/500>`__). See
  :ref:`morphology-possible-issues`.
* Moments: the weighted moments use an approximate distance search. See
  :ref:`moments-possible-issues`.
* GLCM: the matrix is asymmetric by default, so several features differ from PyRadiomics and
  MIRP unless IBSI mode is on. See :ref:`glcm-possible-issues`.
* Texture entropies: GLCM, GLRLM, GLSZM and GLDM use a fast logarithm approximation. An entropy
  is off by at most 0.009 bits: a relative 3.1 × 10\ :sup:`-3` at most on the worked examples,
  about 8 × 10\ :sup:`-3` at most on random distributions, and more on entropies near 0.
  ``GLCM_INFOMEAS1``, derived from entropies, differs by up to 7 × 10\ :sup:`-3` on the worked
  example (see :ref:`conventions-entropy`). See :doc:`Math/f_glcm` and
  :ref:`gldm-possible-issues`.
* Background pixels: with the default binning some families count background as grey level 1.
  See :ref:`glrlm-possible-issues`, :ref:`glszm-possible-issues`, :ref:`gldzm-possible-issues` and
  :ref:`ngtdm-possible-issues`.
* GLSZM: 2D zones are not fully 8-connected. See :ref:`glszm-possible-issues`.
* GLDZM: the distance next to background is one too large. See :ref:`gldzm-possible-issues`.
* NGLDM: grey levels are converted differently from the other texture families. See
  :ref:`ngldm-possible-issues`.
* Gabor: the command line and Python use different default filter pairs. See
  :ref:`gabor-possible-issues`.
* Erosion: ``EROSIONS_2_VANISH_COMPLEMENT`` is always 0. See :ref:`erosion-possible-issues`.
* Zernike: the disk differs from CellProfiler's, and the features are intensity-weighted. See
  :ref:`zernike-possible-issues`.
* Intensity: percentiles come from a histogram, and in-memory arrays are not reported in the
  input's units. See :ref:`intensity-possible-issues`.
* Fractal dimension and image quality: see :ref:`fractal-possible-issues` and
  :ref:`imq-possible-issues`.
* 3D families: see :ref:`intensity-3d-possible-issues`, :ref:`glcm-3d-possible-issues`,
  :ref:`glrlm-3d-possible-issues`, :ref:`glszm-3d-possible-issues`,
  :ref:`gldm-3d-possible-issues`, :ref:`ngldm-3d-possible-issues` and
  :ref:`ngtdm-3d-possible-issues`.
