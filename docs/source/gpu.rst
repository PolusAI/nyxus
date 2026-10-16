.. _gpu:

GPU acceleration
================

A Nyxus build with CUDA support can compute a few expensive 2D feature families on an NVIDIA
GPU. Every other family runs on the CPU, so a GPU run returns the same columns as a CPU run.

.. note::

   The test suite runs without a GPU, and the GPU values are not compared with the CPU values in
   continuous integration. Some GPU features differ from the CPU features by much more than
   rounding: on the same ROIs about half of the shape and intensity moment columns differ, some
   by a fraction of a percent and some by many times the CPU value (see :doc:`Math/f_2dmoments`).


What runs on the GPU
--------------------

Only 2D ROIs use the GPU. Erosion, moments and Gabor use it only for ROIs that Nyxus processes in
memory; out-of-core ROIs (see ``--ramLimit`` in the :ref:`cli-reference`) compute them on the CPU.
GLSZM uses the GPU in memory and out of core. 3D volumes are computed on the CPU. Erosion, moments
and Gabor also need a label image: in whole-slide mode only GLSZM uses the GPU.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Family
     - Features
     - GPU work
   * - Erosion
     - ``EROSIONS_2_VANISH``, ``EROSIONS_2_VANISH_COMPLEMENT``
     - Erodes each ROI on the GPU, but only when a moment feature is in the same run: without one,
       the buffer the erosion kernel needs is not allocated, every ROI gets 0 and the run prints
       ``ErosionFeature: error calculating on GPU``. This is a known defect (`#520 <https://github.com/PolusAI/nyxus/issues/520>`__).
   * - Shape moments
     - ``*SGEOMOMS*``: spatial, central, normalized, Hu and weighted moments
     - Computes the moments and their weights on the GPU.
   * - Intensity moments
     - ``*IGEOMOMS*``: the ``IMOM_*`` codes
     - Same as the shape moments, weighted by intensity.
   * - Gabor
     - ``GABOR``
     - Applies the filter bank on the GPU through an FFT. Command line only (see below).
   * - GLSZM
     - ``*ALL_GLSZM*``
     - Finds the zones on the CPU and sums the 16 features on the GPU. Command line only.

The family pages describe how the GPU results relate to the CPU results:
:ref:`erosion-possible-issues`, :doc:`Math/f_2dmoments`, :ref:`gabor-possible-issues` and
:doc:`Math/f_glszm`.


Get a build with GPU support
----------------------------

The ``nyxus`` package on PyPI is CPU-only. You need a GPU build:

* The CUDA wheels are published as separate packages, named ``nyxus-cuda12x`` for CUDA 12 and
  ``nyxus-cuda11x`` for CUDA 11, for Linux x86_64 (the newest ``nyxus-cuda12x`` with Windows
  wheels is 0.9.3, a known issue, `#526 <https://github.com/PolusAI/nyxus/issues/526>`__). They
  do not bundle cuFFT, and the Linux wheels do not bundle the CUDA runtime either, so install them
  with the CUDA toolkit.
* The conda-forge ``nyxus`` package has CUDA 12.9 builds for Linux x86_64 and Windows x86_64,
  which bring the runtime and cuFFT with them (see :doc:`installation`).
* To build from source, pass ``-DUSEGPU=ON`` to CMake. The CUDA toolkit must be installed, or
  CMake stops with an error. See :doc:`installation` for the full build steps.

A source build compiles kernels for a fixed list of architectures chosen by the CUDA major
version: compute capability 5.0 to 9.0 with CUDA 12, and 7.5 to 12.1 with CUDA 13.


Check for a GPU
---------------

``nyxus.gpu_is_available`` tells whether a usable CUDA device is present.
``nyxus.get_gpu_properties`` lists the devices. Each entry is a dictionary with the keys
``Device number``, ``Device name``, ``Memory`` and ``Capability``. Both functions take the
``id()`` of a ``Nyxus`` object. In a CPU-only build they return ``False`` and an empty list.

The command-line program has no device listing. Use ``nvidia-smi`` to see the device numbers.


Turn the GPU on
---------------

.. tab-set::

   .. tab-item:: Python

      Pass ``use_gpu_device`` with a device number when you create the object. The default,
      ``-1``, computes everything on the CPU.

      .. literalinclude:: ../examples/gpu_check.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

      Output without a GPU:

      .. code-block:: text

         No CUDA GPU available: the features are computed on the CPU
         (2, 190)

   .. tab-item:: Command line

      Add ``--useGpu=true``. ``--gpuDeviceID`` lists the devices Nyxus may use; it defaults
      to ``0``.

      .. code-block:: console

         $ nyxus --features="*GEOMOMS*,GABOR" --useGpu=true --gpuDeviceID=0,1 \
                 --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

      With several IDs, Nyxus picks the listed device with the most free memory at startup.
      Both options exist only in GPU builds.

Use ``featurize_directory`` or ``featurize_files`` with the GPU. ``featurize`` on arrays never
allocates the GPU buffers that the erosion and moment code uses, so do not combine it with
``use_gpu_device`` when the features include erosion or the moments.


When the GPU is not available
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situation
     - What happens
   * - ``--useGpu`` given to a CPU-only build
     - The run stops: the option is not recognized.
   * - ``--useGpu`` with a value other than ``true`` or ``false``
     - The run stops with an error that lists the valid values.
   * - ``--useGpu=true`` and none of the ``--gpuDeviceID`` devices can be used
     - The run stops with ``cannot use any of GPU devices in <ids> due to <reason>``, where the
       reason is the last device's failure, for example ``invalid device ID 7``. A listed device
       that cannot be used is skipped when another one can.
   * - The chosen device fails to initialize
     - The run stops and suggests ``--useGpu=false``.
   * - ``Nyxus(..., use_gpu_device=n)`` without a usable GPU
     - ``ValueError: No need to set GPU device ID because GPU is unavailable``.
   * - ``Nyxus3D(..., use_gpu_device=n)`` or ``ImageQuality(..., using_gpu=n)`` without a usable
       GPU
     - Prints ``No gpu available.`` and computes on the CPU, with one feature thread.
       ``ImageQuality`` also prints ``Warning: unexpected keyword argument(s): using_gpu``, but
       reads the value.
   * - ``nyx.use_gpu_device(n)`` in a CPU-only build
     - ``RuntimeError: this Nyxus backend was built without the GPU support``.

Nyxus never moves to the CPU on its own once the GPU is in use: a failed allocation of the GPU
cache stops the run with ``allocating GPU cache failed``, and a failed GLSZM allocation gives that
ROI the ``--noval`` value.


Deviations in the Python interface
----------------------------------

.. warning::

   * The device number given to ``use_gpu_device`` is not passed to CUDA, so the computation
     runs on the CUDA default device, device 0. A number past the last device prints ``Warning:
     GPU choice (<n>) is out of range. Defaulting to device 0``.
   * The method ``nyx.use_gpu_device(n)`` turns the GPU on or off rather than choosing a device:
     ``0`` turns it off and any other number turns it on.
   * ``set_params(use_gpu_device=n)`` is accepted and has no effect.
   * The GLSZM family stays on the CPU in Python, because its GPU switch is read before the GPU
     is turned on.
   * Gabor must not be computed on the GPU from Python: the GPU filter bank is built only by the
     command-line program, so the Python package runs the GPU convolution with an empty filter
     bank: it prints CUDA errors and then, depending on the input, raises ``RuntimeError: Error:
     1`` or returns 0 in every ``GABOR`` column. Leave the GPU off for ``GABOR`` in Python.
   * ``Nyxus3D`` and ``ImageQuality`` drop to one feature thread whenever a device number is
     given, even when they then compute on the CPU. No 3D or image-quality feature has a GPU
     path.
