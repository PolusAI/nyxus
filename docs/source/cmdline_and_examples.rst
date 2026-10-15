Command line and Python examples
================================

Running Nyxus from the command line
-----------------------------------

A command-line run reads pairs of intensity and label images from two directories and writes
the features of every ROI to an output directory. Four options are required on every run:

.. code-block:: bash

   nyxus --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

``--outputType`` has no default: choose ``separatecsv`` (one CSV per image pair), ``singlecsv``,
``arrowipc`` or ``parquet``. Without ``--features``, Nyxus computes every 2D feature except the
IBSI intensity histogram (``IH_*``) features, which need ``--ibsi=true``. Every option is listed
with its default in the :ref:`cli-reference`.

Quote option values that contain ``*``, ``?``, ``[`` or ``\``, such as
``--features="*ALL_INTENSITY*"`` and the regular expressions of ``--filePattern``. Unquoted, the
shell treats them as file-name patterns or escapes: zsh, the default shell on macOS, stops on a
pattern that matches no file, and bash and zsh remove the backslashes from a regular expression.

Command-line examples
---------------------

The examples use ``/data/int``, ``/data/seg`` and ``/data/out`` for the intensity, label and output
directories.

Requesting specific features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Zernike features and the first three Hu moments:

.. code-block:: bash

   nyxus --features=ZERNIKE2D,HU_M1,HU_M2,HU_M3 --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

.. _requesting-feature-groups:

Requesting feature groups
^^^^^^^^^^^^^^^^^^^^^^^^^

All intensity features and the basic morphology features. On the command line, feature and
group names are not case-sensitive:

.. code-block:: bash

   nyxus --features="*all_intensity*,*basic_morphology*" --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

Groups and individual features can be mixed:

.. code-block:: bash

   nyxus --features="*all_intensity*,*basic_morphology*,zernike2d" --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

A name prefixed with ``-`` removes that feature or group, and one prefixed with ``+`` adds it, which
is the same as no prefix. Names are applied from left to right, so a removal must come after the
group it removes from. Prefixes work in ``--features`` and in Python, not in a feature list
file (see :ref:`below <feature-list-file>`). All intensity features except ``MEAN``:

.. code-block:: bash

   nyxus --features="*all_intensity*,-mean" --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

The same works in Python, for example ``Nyxus(["*ALL_INTENSITY*", "-MEAN"])``. ``-COV`` and
``-*ALL*`` add instead of removing: ``["MEAN", "-COV"]`` gives ``COV`` and ``MEAN``, and
``["*ALL_INTENSITY*", "-*ALL*"]`` gives every feature that ``*ALL*`` selects. On an image quality
name a ``-`` or ``+`` prefix is ignored: ``ImageQuality(["FOCUS_SCORE", "-SHARPNESS"])`` computes
both features.

Requesting the IBSI intensity histogram features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The intensity histogram (``IH_*``) family requires IBSI mode (``--ibsi=true``) and uses
``--coarseGrayDepth`` as the histogram bin count. Add ``--mergerois=true`` to compute one
whole-foreground ROI instead of one ROI per label:

.. code-block:: bash

   nyxus --features="*ALL_IH*" --ibsi=true --coarseGrayDepth=32 --mergerois=true --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

.. _feature-list-file:

Reading the feature list from a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A long feature list can be kept in a text file, with names separated by commas or new lines.
``#`` starts a comment. Only individual feature names work in a file: Nyxus drops every
character other than letters, digits, ``_`` and ``,``, so spaces join neighbouring names, the
``*`` of a group name is removed, and a ``-`` or ``+`` prefix is lost.
Suppose ``feature_list.txt`` contains:

.. code-block:: text

   # first-order statistics
   mean,min,kurtosis
   skewness

Pass the file name to ``--features``:

.. code-block:: bash

   nyxus --features=feature_list.txt --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

Selecting files
^^^^^^^^^^^^^^^

``--filePattern`` is a regular expression that must match the whole file name. It defaults to
``.*``, which selects every file. To process only OME-TIFF files:

.. code-block:: bash

   nyxus --filePattern=".*\.ome\.tif" --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

Nyxus prints ``Warning: --filePattern= already has value '.*'`` whenever ``--filePattern`` is
given. The warning is harmless: the pattern you give is the one used.

Whole-image features
^^^^^^^^^^^^^^^^^^^^

Normally each nonzero label in a label image is one ROI. To treat each whole intensity image as a
single ROI, give the intensity directory as the label directory too:

.. code-block:: bash

   nyxus --features="*basic_morphology*" --intDir=/data/int --segDir=/data/int --outDir=/data/out --outputType=singlecsv

Pairing intensity and label images explicitly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default an intensity image is paired with the label image of the same file name:

.. code-block:: text

   /data/int/image_1.ome.tif    /data/seg/image_1.ome.tif
   /data/int/image_2.ome.tif    /data/seg/image_2.ome.tif

When several intensity images share one label image, list the pairs in a text file, one pair per
line: the intensity file name first, then the label file name. Names are relative to ``--intDir``
and ``--segDir``. For a file ``mapping.txt``:

.. code-block:: text

   image_1.ome.tif       image_A.ome.tif
   image_2.ome.tif       image_A.ome.tif
   image_3.ome.tif       image_B.ome.tif

pass the directory that holds it with ``--intSegMapDir`` and its name with ``--intSegMapFile``.
``--filePattern`` is not applied to the listed pairs. This option is available for 2D data.

.. code-block:: bash

   nyxus --intSegMapDir=/data --intSegMapFile=mapping.txt --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

Python examples
---------------

The examples below are run by the test suite exactly as shown. Each prints part of the feature
table; the full table has one column per feature after nine description columns
(``intensity_image, mask_image, phys_unit, ROI_label, t_index, c_index, phys_x, phys_y, phys_z``).

Image files listed explicitly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``featurize_files`` takes a list of intensity files and the matching list of label files.
``single_roi=False`` treats each nonzero label as a ROI; ``True`` treats each whole image as one.

.. literalinclude:: ../examples/featurize_files.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

     intensity_image  mask_image  ROI_label         MEAN
   0      i1.ome.tif  m1.ome.tif          1  2081.718750
   1      i1.ome.tif  m1.ome.tif          2  2034.362500
   2      i1.ome.tif  m1.ome.tif          3  2123.508242
   3      i2.ome.tif  m2.ome.tif          1  2120.781250
   4      i2.ome.tif  m2.ome.tif          2  1980.590625
   5      i2.ome.tif  m2.ome.tif          3  2088.107143

Images in memory
^^^^^^^^^^^^^^^^

``featurize`` takes numpy arrays: one 2D image, or a stack of 2D images along the first axis, with
a label array of the same shape. Images are named ``Intensity0``, ``Intensity1``, ... unless names
are given.

.. literalinclude:: ../examples/featurize_arrays.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

     intensity_image     mask_image  ROI_label      MEAN  MEDIAN
   0      Intensity0  Segmentation0          1  2.950000     4.0
   1      Intensity1  Segmentation1          1  2.263158     1.0
     intensity_image    mask_image  ROI_label      MEAN  MEDIAN
   0         cells_a  cells_a_mask          1  2.950000     4.0
   1         cells_b  cells_b_mask          1  2.263158     1.0

A directory of images, and reading and changing parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``featurize_directory`` pairs each intensity image with the label image of the same name.
``get_params`` returns the current parameters, all of them when called with no arguments, and
``set_params`` records new values, which ``get_params`` then reports.

.. note::

   ``features``, ``coarse_gray_depth`` and ``ibsi`` changed with ``set_params`` after the object is
   created do not reach the computation. After ``set_params(coarse_gray_depth=8)``,
   ``GLCM_ASM_0`` keeps the value of the 64-level matrix (0.000882 on a random 40 x 40 image,
   where an object created with ``coarse_gray_depth=8`` gives 0.0162), and after
   ``set_params(features=["MAX"])`` the object still computes its original features. Pass these
   three to the constructor, or create a new object.

.. literalinclude:: ../examples/directory_and_params.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

   (4, 194)
   {'coarse_gray_depth': 64, 'n_feature_calc_threads': 4}
   {'coarse_gray_depth': 256, 'n_feature_calc_threads': 4}

Arrow IPC and Parquet output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass ``output_type="arrowipc"`` or ``output_type="parquet"`` to any ``featurize*`` call to write the
results to a file instead of returning a DataFrame; the call returns the file's path.
``output_path`` sets the directory or file name. This needs a Nyxus build with Arrow support,
which the PyPI and conda packages have.

.. literalinclude:: ../examples/arrow_output.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output (the path uses the platform's separator):

.. code-block:: text

   results/NyxusFeatures.arrow
   1 ['intensity_image', 'mask_image', 'phys_unit', 'ROI_label']

Nested ROIs
^^^^^^^^^^^

The ``Nested`` class relates ROIs of two label channels: a ROI of the child channel is a child of a
ROI of the parent channel when its bounding box lies inside the parent's bounding box. For
example, consider the intensity and label images of a parent channel,

.. list-table::

    * - .. figure:: img/parent_int.png

           Fig 1. Parent channel intensity

      - .. figure:: img/parent_seg.png

           Fig 2. Parent channel segmentation

and of a child channel:

.. list-table::

    * - .. figure:: img/child_int.png

           Fig 3. Child channel intensity

      - .. figure:: img/child_seg.png

           Fig 4. Child channel segmentation

Some child ROIs lie inside parent ROIs. In this data set the channel is part of the file name:
``c0`` marks the child channel and ``c1`` the parent channel, so a regular expression selects
each one. First compute the features of the child ROIs, then find the relations:

.. literalinclude:: ../examples/nested_rois.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

``find_relations`` returns one row per parent/child pair, with the columns
``Image, Parent_Label, Child_Label``.

.. warning::

   ``find_relations`` reports every child ROI as contained in every parent ROI: on the images
   above it returns 4818 pairs for 73 parents and 66 children, while 57 of the children lie
   inside a parent's bounding box. ``Nested.featurize`` expects the child label column to be
   named ``label``, while ``featurize_directory`` names it ``ROI_label``, and it groups the
   joined child features by that child label, not by parent. Check the relations before relying
   on aggregated results.
