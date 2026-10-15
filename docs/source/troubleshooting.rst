.. _troubleshooting:

Troubleshooting and performance
===============================

This page lists the errors Nyxus reports most often, with their exact messages, and shows how to
control its threads and memory and how to time a run. Every option it names is described in the
:ref:`cli-reference`.

Common errors
-------------

Feature names
^^^^^^^^^^^^^

An unknown feature or group name stops the run before any image is read. In Python the
``Nyxus`` constructor raises a ``ValueError``:

.. code-block:: text

   Error: 'mean' is not a valid 2D feature name

In 3D the message says ``3D feature name``, and for the ``ImageQuality`` class ``Image Quality
feature name``. The command line reports each unknown name, in upper case, prints the valid
names, and stops:

.. code-block:: text

   Error: expecting MEANX to be a proper 2D feature name or feature file path
   Stopping due to errors while parsing user requested features

* **Case.** The command line converts ``--features`` to upper case, so ``mean`` and
  ``*all_intensity*`` work there. Python passes the names unchanged, so write them in upper case:
  ``"MEAN"``, ``"*ALL_INTENSITY*"``.
* **Image quality.** Image quality features cannot be mixed with other features. Request them
  through the ``ImageQuality`` class (see :doc:`imagequality`). It refuses other feature names,
  except after ``*ALL_IMQ*``: names that follow it are dropped without a message.
* **IBSI histogram features.** The ``IH_*`` features need IBSI mode (``--ibsi=true``, Python
  ``ibsi=True``). Without it they are removed from the request. The command line prints
  ``Warning: Intensity Histogram (IH_*) features require IBSI mode; they are disabled because IBSI
  compliance is off.``; Python, whose default verbosity is 0, prints nothing and returns no
  ``IH_*`` columns.

.. warning::

   The groups ``*ALL_BUT_GABOR*`` and ``*ALL_BUT_GLCM*`` are listed as group names but define no
   features, so a request that uses them stops with ``Error: '*ALL_BUT_GABOR*' is not a valid 2D
   feature name``. List the groups you need instead.

The example below triggers the feature-name and file-pattern errors and prints their messages:

.. literalinclude:: ../examples/troubleshooting_errors.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

   Error: 'mean' is not a valid 2D feature name
   Invalid filepattern *.tif
   Error reading the dataset: no intensity and/or label files to process, probably due to file pattern .*\.tiff
     intensity_image  ROI_label        MEAN
   0        img1.tif          1  2081.71875
   1        img1.tif          2  2034.36250

The file-pattern check also prints ``Exception while checking file pattern *.tif`` on the error
stream.

Missing or misspelled command-line options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--intDir``, ``--segDir``, ``--outDir`` and ``--outputType`` are required. A missing one stops
the run with, for example:

.. code-block:: text

   Error: Missing argument --outputType

An option Nyxus does not know is listed after ``Error - unrecognized arguments:`` and stops the
run. This includes options of features that the build lacks: ``--useGpu`` and ``--gpuDeviceID``
exist only in GPU builds, and ``--exclusivetiming`` only in timing builds.

In Python, a misspelled keyword argument does not stop the run. The constructor prints
``Warning: unexpected keyword argument(s): n_threads`` and ignores the argument, so check
for this line. ``set_params`` with an unknown name raises ``ValueError``.

Files not found or not matched
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--filePattern`` and ``file_pattern`` are regular expressions that must match the whole file
name. A shell wildcard such as ``*.tif`` is not a valid regular expression. The command line
reports ``Error: invalid file pattern '*.tif'`` and Python raises
``ValueError: Invalid filepattern *.tif``. Write ``.*\.tif`` instead.

A valid pattern that matches no file, or matches only in one of the two directories, stops the
run with ``no intensity and/or label files to process, probably due to file pattern ...``.
Python prefixes it with ``Error reading the dataset:`` and the command line with ``Errors while
reading the dataset:``. Other dataset messages are:

* ``mismatch: 3 intensity images vs 2 mask images``: the pattern selects different numbers of files
  in the two directories;
* ``cannot find the mask counterpart for img1.tif;``: no label file has the same name;
* ``cannot access directory ...``: a directory does not exist. Python checks the input
  directories first and raises ``OSError: Provided intensity image directory '...' does not exist.``;
* ``File does not exist: ...``: a path passed to ``featurize_files`` does not exist.

.. warning::

   In whole-slide mode (no label directory in Python, or ``--segDir`` equal to ``--intDir``), a
   pattern that matches nothing is not reported. ``featurize_directory("int",
   file_pattern=r".*\.tiff")`` returns one row of placeholder columns ``column1`` to
   ``column4`` holding ``blank``, ``blank``, ``0.0`` and ``0.0``. Check the row count of the result.

Unsupported or compiled-out formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nyxus picks the reader from the file extension (see :doc:`supported_formats`). A file it cannot
read is reported on the console:

.. code-block:: text

   Error while initializing the image loader for intensity image file int\b.png: unsupported file
   type: int\b.png -- Nyxus reads .tif, .tiff, .ome.tif, .ome.tiff, .zarr, .ome.zarr, .dcm, .dicom,
   .nii and .nii.gz

Python then raises ``RuntimeError: Error: 1``, so read the console line above it for the cause.
Restrict ``file_pattern`` to the image files to avoid it.

.. note::

   Nyxus writes these messages from compiled code straight to the process's standard output and
   error, not through Python's ``sys.stdout``, and the exception carries only ``Error: 1``. In a
   Jupyter notebook on Windows (tested with ipykernel 7.3), the messages go to the terminal that
   runs the kernel, and the cell shows only the exception. Look there, or run the same call in a
   plain ``python`` session in a terminal. A fix that puts the cause into the exception is
   coming.

A build without an optional part refuses its files or outputs:

* OME-Zarr: ``This version of Nyxus was not build with OmeZarr support``;
* compressed NIfTI (``.nii.gz``) in a build without OME-Zarr: ``failed to find header file for
  '<file>.nii.gz'``;
* DICOM: ``This version of Nyxus was not build with DICOM support``;
* Arrow IPC and Parquet on the command line: ``Error: Nyxus must be built with Apache Arrow enabled
  to use Arrow output types. Please rebuild with the flag USEARROW=ON.`` The CMake option is
  ``USE_ARROW`` (or ``ALLEXTRAS``);
* Arrow IPC and Parquet in Python: the console shows ``Apache Arrow functionality is not
  available. Please install Nyxus with Arrow enabled to use this functionality.`` and the call
  raises ``RuntimeError``.

The PyPI and conda packages include all of them, except that the Windows wheels on PyPI have no
OME-Zarr support since 0.9.3 (see :doc:`installation`). The first conda-forge Windows builds of
0.11.0 (build number 0) also lack OME-Zarr, ``.nii.gz`` and DICOM; conda installs one of them into
an environment that holds ``libarrow`` 19 or 20.
:ref:`installation-check` shows how to check a build.

GPU
^^^

* Python ``Nyxus(..., use_gpu_device=0)`` on a machine or package without a usable CUDA GPU raises
  ``ValueError: No need to set GPU device ID because GPU is unavailable``. ``Nyxus3D``
  (``use_gpu_device``) and ``ImageQuality`` (``using_gpu``) instead print ``No gpu available.``
  and compute on the CPU.
* On the command line, a device that cannot be initialized stops the run with ``Error: cannot use
  GPU device ID 0. You can disable GPU usage via command line option --useGpu=false``.

Memory limit
^^^^^^^^^^^^

A ``--ramLimit`` above the available memory is reported as:

.. code-block:: text

   Error: RAM limit 1000000000 megabytes (=1048576000000000 bytes) exceeds the actual amount of available RAM 92068978688 bytes

The command line stops. Python prints the same line, keeps its previous limit and continues.
Check the limit in effect with ``get_params("ram_limit")``.

Performance
-----------

Threads
^^^^^^^

Nyxus divides the ROIs of each image among its feature calculation threads:
``--reduceThreads`` on the command line and ``n_feature_calc_threads`` in Python, 4 by default.
Python also accepts it in ``set_params``.

* More threads help when an image holds many ROIs, or, in whole-slide mode, when the run has
  many images: whole-slide mode gives each thread one image. On a single image a whole-slide
  run uses one thread: ``*ALL_INTENSITY*`` and ``*ALL_GLCM*`` on one 1024 x 1024 image took
  0.21 s with 1 thread and 0.23 s with 4.
* When a GPU device is requested (``Nyxus3D`` ``use_gpu_device``, ``ImageQuality``
  ``using_gpu``) with more than one thread, they reduce the thread count to 1 and print
  ``Gpu features only support a single thread. Defaulting to one thread.``

Memory
^^^^^^

``--ramLimit`` (Python ``ram_limit``) is a memory budget in megabytes. Its default is half of the
physical memory available when Nyxus starts. A ROI whose estimated footprint reaches the budget
is processed out of core, and the other ROIs are processed in batches that fit within it. In
Python, ``get_params("ram_limit")`` reports the budget in megabytes. ``Nyxus3D`` accepts
``ram_limit`` in its constructor only, not in ``set_params``. :doc:`large_images` says which
features differ when a ROI is processed out of core. On a 2D image that Nyxus reads as several
tiles (a tiled TIFF with more than one tile, or a striped TIFF larger than 1024 x 1024 pixels),
out-of-core processing raises an error for most 2D families or gives wrong values, depending on
the family and the ROI. Only the intensity features of a
tiled image stay correct; see :ref:`large-images-tiles`.

Cost of the feature families
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most families take about the same time as the intensity features. Erosion, geometric moments
and Gabor cost the most, and their cost grows with ROI size. The times below were measured with
the Python package on a 1024 x 1024 16-bit image, with 4 threads. Absolute times depend on the
machine; compare the rows with each other.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Features
     - 64 ROIs of 5,013 pixels
     - 4 ROIs of 180,905 pixels
   * - ``*ALL_INTENSITY*``
     - 0.07 s
     - 0.11 s
   * - ``*ALL_GLCM*``
     - 0.09 s
     - 0.10 s
   * - ``GABOR``
     - 0.23 s
     - 0.43 s
   * - ``*GEOMOMS*``
     - 0.34 s
     - 0.69 s
   * - ``EROSIONS_2_VANISH``
     - 0.17 s
     - 1.38 s
   * - ``*ALL_MORPHOLOGY*`` (includes the erosion features)
     - 0.16 s
     - 1.44 s
   * - ``*ALL*``
     - 0.95 s
     - 3.36 s

To speed up a run:

* Request only the features you need. ``*ALL*`` is the default when ``--features`` is omitted.
* Leave out ``GABOR``, the erosion features (``EROSIONS_2_VANISH``,
  ``EROSIONS_2_VANISH_COMPLEMENT``) and the moment groups (``*GEOMOMS*``, ``*IGEOMOMS*``,
  ``*SGEOMOMS*``) when you do not use them. Gabor computes one filtered image per frequency and
  orientation pair, so fewer ``--gaborfreqs`` / ``--gabortheta`` pairs cost less.
* A GPU build runs the geometric moments on the GPU (``--useGpu=true``, Python
  ``use_gpu_device``), and erosion too, which returns 0 unless a moment feature is in the same
  run; Gabor and GLSZM run there too on the command line; leave the GPU off for
  ``GABOR`` in Python. See :ref:`gpu`.

Timing a run
^^^^^^^^^^^^

In Python, time each call with ``time.perf_counter``. This script compares two groups and two
thread counts:

.. literalinclude:: ../examples/timing_a_run.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output (times vary by machine):

.. code-block:: text

   *ALL_INTENSITY*    1 threads: 0.04 s for 16 ROIs
   *ALL_INTENSITY*    4 threads: 0.02 s for 16 ROIs
   *ALL_MORPHOLOGY*   1 threads: 0.12 s for 16 ROIs
   *ALL_MORPHOLOGY*   4 threads: 0.04 s for 16 ROIs

On the command line, time the whole run with your shell, for example ``time nyxus ...`` on Linux
and macOS, or ``Measure-Command { nyxus ... }`` in PowerShell.

For a per-step breakdown, build the command-line program with the ``CHECKTIMING`` macro. It is not
a CMake option, so pass it as a compiler flag:

.. code-block:: bash

   cmake -S . -B build-cli -DBUILD_CLI=ON -DCMAKE_CXX_FLAGS=-DCHECKTIMING

A timing build prints the start, finish and gross elapsed time of every run. For a 2D run with
label images it also prints a table of time per step, and writes the table to
``inclusive_nyxustiming.csv`` in the output directory, for the whole collection. With
``--exclusivetiming=true`` it writes one ``<mask file>_nyxustiming.csv`` per image pair instead.
Both need ``--verbose`` at 1 or more, which is the default.
