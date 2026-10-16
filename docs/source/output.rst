.. _output:

Output formats and columns
==========================

Nyxus returns one table row per ROI, or one row per image in whole-slide mode. This page
describes where that table goes, how its columns are laid out, and how missing values appear.


Output types
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Command line ``--outputType``
     - Python ``output_type``
     - Result
   * - ``separatecsv``
     - --
     - One CSV file per intensity/label image pair.
   * - ``singlecsv``
     - --
     - One CSV file for the whole run.
   * - ``arrowipc``
     - ``"arrowipc"``
     - One Arrow IPC file (``.arrow``). Needs a build with Arrow support.
   * - ``parquet``
     - ``"parquet"``
     - One Parquet file (``.parquet``). Needs a build with Arrow support.
   * - --
     - ``"pandas"`` (default)
     - A pandas DataFrame returned by the call.

``--outputType`` has no default on the command line; the values are not case-sensitive. The PyPI
and conda packages are built with Arrow support. An example that writes and reads an Arrow IPC
file is in the section *Arrow IPC and Parquet output* of :doc:`cmdline_and_examples`.


File names and locations
------------------------

**Command line.** Every file goes to ``--outDir``, which must exist.

* ``separatecsv`` names each file after its image pair: ``_INT_<intensity file>_SEG_<label
  file>.csv``. For example, ``_INT_p0_c0.ome.tif_SEG_p0_c0.ome.tif.csv``.
* ``singlecsv``, ``arrowipc`` and ``parquet`` write ``<name>.csv``, ``<name>.arrow`` or
  ``<name>.parquet``. ``<name>`` is ``--resultFname``, by default ``NyxusFeatures``.

A rerun overwrites the files. Within one run, later images and planes are appended.

**Python.** With ``output_type="arrowipc"`` or ``"parquet"``, the call writes a file and returns
its path. ``output_path`` sets where:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - ``output_path``
     - File written
   * - ``""`` (default)
     - ``NyxusFeatures.arrow`` or ``NyxusFeatures.parquet`` in the current directory.
   * - An existing directory, or a path ending in ``/``
     - ``NyxusFeatures.<ext>`` inside it.
   * - A path with an extension, such as ``results/run1.arrow``
     - That file, with the extension replaced to match the output type. With
       ``output_type="parquet"``, ``results/run1.arrow`` becomes ``results/run1.parquet``.
   * - A path without an extension whose parent directory exists
     - That path with the extension added.
   * - A path without an extension whose parent does not exist, or a bare name such as
       ``results`` (``./results`` counts as having a parent)
     - ``NyxusFeatures.<ext>`` inside a directory of that name.

Nyxus creates the file's directory if it is missing, but not a missing parent of that directory.


Column layout
-------------

Every table starts with nine description columns, in this order (CSV output with
``--annot=true`` inserts the annotation columns after ``phys_unit``):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Content
   * - ``intensity_image``
     - Intensity file name, without the directory. For in-memory arrays, the name you passed or
       ``Intensity0``, ``Intensity1``, ...
   * - ``mask_image``
     - Label file name, or ``Segmentation0``, ... for arrays.
   * - ``phys_unit``
     - Physical unit of the pixel size, read from OME metadata. Empty when the file has none.
   * - ``ROI_label``
     - The ROI's label value. ``-1`` on an aggregated row (see below).
   * - ``t_index``
     - Time point of the plane, from 0. Always 0 in 2D runs.
   * - ``c_index``
     - Channel of the plane, from 0. Always 0 in 2D runs.
   * - ``phys_x``, ``phys_y``, ``phys_z``
     - Pixel or voxel size along each axis, from OME metadata; 1 when the file has none.

Feature columns follow, in Nyxus's own feature order rather than the order of your request. Some
features produce several columns, named with a suffix: ``GLCM_*`` and ``GLRLM_*`` get one column
per angle (``_0``, ``_45``, ``_90``, ``_135``) except the ``*_AVE`` features, ``GABOR`` one per filter (``_0``, ``_1``, ...),
``ZERNIKE2D`` ``_Z0`` to ``_Z29``, and ``HISTOGRAM`` ``_BIN_0`` to ``_BIN_<n-1>``. The
:doc:`featurelist` gives the columns of each feature.

In pandas, ``ROI_label`` is an unsigned 32-bit integer and the numeric columns are 64-bit floats.
In Arrow and Parquet files, ``ROI_label`` is a signed 32-bit integer.

CSV files quote the column names. Feature values are written with 6 significant digits, so a
``MEAN`` of 39857.84117647059 in pandas reads 39857.8 in the CSV file. Use Arrow or Parquet output
when you need full precision.

In whole-slide mode (a label directory equal to the intensity directory, or ``single_roi=True``
in Python) ``intensity_image`` holds the intensity path as given: the directory joined with the
file name, or the path passed to ``featurize_files``. ``mask_image`` is empty.

.. literalinclude:: ../examples/output_columns.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

   ['intensity_image', 'mask_image', 'phys_unit', 'ROI_label', 't_index', 'c_index', 'phys_x', 'phys_y', 'phys_z', 'MEAN', 'AREA_PIXELS_COUNT']
             intensity_image  ROI_label         MEAN  AREA_PIXELS_COUNT
   0  plate1_well_A1.ome.tif          1  2112.805556              576.0
   1  plate1_well_A1.ome.tif          2  2079.703125              768.0
   2  plate1_well_B2.ome.tif          1  2072.798611              576.0
   3  plate1_well_B2.ome.tif          2  1978.470052              768.0
                                  MEAN  AREA_PIXELS_COUNT
   intensity_image
   plate1_well_A1.ome.tif  2096.254340              672.0
   plate1_well_B2.ome.tif  2025.634332              672.0


Annotations
-----------

``--annot=true`` adds columns ``anno0``, ``anno1``, ... after ``phys_unit``. Nyxus takes the label
file name up to its first dot and splits it at underscores; empty pieces are dropped. The label
file ``p0_y1_r1_c0.ome.tif`` gives ``p0``, ``y1``, ``r1`` and ``c0``.

.. code-block:: console

   $ nyxus --features=MEAN --annot=true --intDir=/data/int --segDir=/data/seg \
           --outDir=/data/out --outputType=singlecsv

Annotations appear only in CSV output for images with label images. With Arrow IPC or Parquet
output, ``--annot=true`` currently stops the run with a crash and leaves an unreadable file, so
leave it off there. This is a known defect (`#517 <https://github.com/PolusAI/nyxus/issues/517>`__). Annotations always split at ``_``: ``--annotsep`` is
accepted but not used. The Python interface has no annotation option.


Aggregation
-----------

``--aggr=true`` replaces each image's ROI rows with one row: the mean of every feature over the
image's ROIs, with ``ROI_label`` set to ``-1``.

.. code-block:: console

   $ nyxus --features="*ALL_INTENSITY*" --aggr=true --intDir=/data/int --segDir=/data/seg \
           --outDir=/data/out --outputType=singlecsv

Aggregation applies to CSV output for images with label images. Arrow, Parquet and whole-slide
output ignore it. In Python, group the DataFrame instead, as the example above does.

.. note::

   The mean includes ROIs whose feature could not be computed, at their ``--noval`` value. With
   ``--noval=-999``, 11 of 73 ROIs of an image had no valid ``NGTDM_COARSENESS``, and the
   aggregated row reported -150.5 for it. Values that are still NaN or infinite at output time
   are not averaged this way: see the ``--aggr`` entry in the :ref:`cli-reference`.


Missing values
--------------

A feature that cannot be computed for a ROI, such as a texture feature of a ROI with one
intensity, is reported as the ``--noval`` value, 0 by default. Values that are still NaN or
infinite at output time become ``--noval`` in CSV files and in pandas, and ``-0.0`` in Arrow and
Parquet files. The tables therefore contain no NaN.

.. code-block:: console

   $ nyxus --features="*ALL_NGTDM*" --noval=-999 --intDir=/data/int --segDir=/data/seg \
           --outDir=/data/out --outputType=singlecsv

Choose a ``--noval`` that no real value can take, so you can tell such cells from real zeros in
CSV output. In Arrow and Parquet files, values that are still NaN or infinite at output time are
written as ``-0.0`` whatever ``--noval`` is, and some real values are also ``-0.0``, so these
cells cannot be told apart from zeros there. The Python interface has no ``noval`` option and
always uses 0.

``--tinyval`` is accepted and validated, but no feature reads it in the current code.
