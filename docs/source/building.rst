.. _building:

Building from source
====================

This page builds the Python package and the command-line program from the repository. To install a
published package instead, see :doc:`installation`.

Build from a git clone of the repository. The package version comes from the git tags, so a
GitHub source archive of the development head or a copy without the ``.git`` directory stops with
``setuptools-scm was unable to detect version``:

.. code-block:: bash

   git clone https://github.com/PolusAI/nyxus.git
   cd nyxus

The ``ci-utils`` and ``tests`` paths on this page are relative to this directory.

Requirements
------------

* CMake 3.20 or newer.
* A C++20 compiler. On Windows, use MSVC: the conda-forge Windows libraries are built with MSVC
  and do not link with GCC.
* libtiff, which is always required.
* For each optional part, its libraries:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Part
     - Libraries
   * - OME-Zarr input
     - z5 (headers), Blosc, zlib, nlohmann_json; zstd for Zstandard-compressed Zarr version 3
       arrays, used when both its library and header are found
   * - DICOM input
     - DCMTK and zlib; fmjpeg2k for JPEG 2000 pixel data
   * - Arrow IPC and Parquet output
     - Apache Arrow and Parquet
   * - Python package
     - pybind11
   * - GPU
     - a CUDA toolkit whose ``nvcc`` supports your host compiler

Getting the dependencies
------------------------

.. tab-set::

   .. tab-item:: conda (recommended)

      The files in ``ci-utils/envs/`` list the conda-forge packages:

      * ``conda_cpp.txt``: the C++ libraries, CMake and pybind11;
      * ``conda_cpp_mac.txt``: the same without DCMTK and fmjpeg2koj, for Apple Silicon Macs and
        Linux aarch64;
      * ``conda_py.txt``: numpy, pandas, pyarrow and pip;
      * ``conda_linux_compiler.txt`` and ``conda_linux_gpu.txt``: a Linux compiler and the CUDA 12.6
        compiler and libraries.

      .. code-block:: bash

         conda create -n nyxus_build -c conda-forge python=3.12 \
             --file ci-utils/envs/conda_cpp.txt --file ci-utils/envs/conda_py.txt
         conda activate nyxus_build

      On Apple Silicon Macs and on Linux aarch64, use ``conda_cpp_mac.txt`` in place of
      ``conda_cpp.txt``: the other file does not solve there, and this one has no DICOM libraries.
      On Linux, also add ``--file ci-utils/envs/conda_linux_compiler.txt`` so that the compiler
      matches the conda libraries, as the CI builds do.

      Point CMake at the environment with the ``NYXUS_DEP_DIR`` environment variable:
      ``$CONDA_PREFIX`` on Linux and macOS, ``%CONDA_PREFIX%\Library`` on Windows. When it is set,
      CMake uses it as ``CMAKE_PREFIX_PATH`` and as a library directory.

   .. tab-item:: Linux and macOS scripts

      ``ci-utils/install_prereq_linux.sh`` downloads and builds the dependencies into
      ``./local_install``:

      .. code-block:: bash

         bash ci-utils/install_prereq_linux.sh --build_arrow yes
         export NYXUS_DEP_DIR=$PWD/local_install

      Its options are ``--min_build yes`` (only pybind11, libtiff and their codecs, no OME-Zarr,
      DICOM or Boost), ``--build_arrow yes`` (also Arrow and Parquet; off by default) and
      ``--install_dir <path>``, a path relative to the current directory (the script prefixes it
      with ``../../`` for each dependency build). The script needs ``curl``, ``unzip``, ``bzip2``,
      ``make``, CMake and a C++ compiler. It does not stop when a step fails, so check its output
      for errors before building. On macOS, run it with bash 4 or newer (for example Homebrew's
      ``bash``): the system ``/bin/bash`` 3.2 reports ``bad substitution`` at the ``--min_build``
      check, ignores the option and builds every dependency.

   .. tab-item:: Windows script

      ``ci-utils\install_prereq_win.bat`` downloads and builds the dependencies, including Arrow,
      into ``.\local_install``. Run it from a Visual Studio developer command prompt:

      .. code-block:: bat

         set PATH=%PATH%;C:\Program Files\Git\usr\bin
         ci-utils\install_prereq_win.bat
         set NYXUS_DEP_DIR=%cd%\local_install
         set PATH=%NYXUS_DEP_DIR%\bin;%PATH%

      ``--min_build yes`` skips OME-Zarr and DICOM. The first line adds Git for Windows'
      ``usr\bin``, for the ``unzip`` the script needs, at the end of ``PATH``. Placed earlier, its
      ``link.exe`` and ``tar.exe`` hide the MSVC linker and the Windows ``tar`` that the script
      uses for ``.zip`` files. The script does not stop when a step fails, so check its output for
      errors before building. Its Boost download fails (the URL returns HTTP 404); the OME-Zarr
      build does not need Boost, so this error can be ignored.

      The dependency DLLs in ``local_install\bin`` must be on ``PATH`` when you run the
      command-line program. Python does not search ``PATH`` for an extension module's DLLs, so in
      Python call ``os.add_dll_directory(r"<source tree>\local_install\bin")`` before
      ``import nyxus``.

Building the Python package
---------------------------

``pip`` builds the package through ``setup.py``, which configures CMake with ``-DALLEXTRAS=ON``
(see `CMake options`_). With ``NYXUS_DEP_DIR`` set as above, run from the source tree:

.. code-block:: bash

   python -m pip install . -vv

Pass more CMake options in the ``CMAKE_ARGS`` environment variable, separated by spaces. For
example, a GPU build, which needs a CUDA toolkit: ``CMAKE_ARGS="-DUSEGPU=ON" python -m pip install
. -vv`` in bash, or ``set CMAKE_ARGS=-DUSEGPU=ON`` before the ``pip`` command in a Windows command
prompt.

The package then includes every optional part whose libraries CMake finds. A missing optional
dependency switches its part off and the build continues, except the GPU part: without a CUDA
toolkit, configuring stops. Not every such case prints a warning (a missing DCMTK drops DICOM with
only status lines), so check the dependency summary at the end of the configure output, which
lists which optional parts were requested and which were found:

.. code-block:: text

   -- Nyxus Dependencies Summary
   -- ================================
   -- Support | Requested | Found
   -- ================================
   -- Z5      | ON | ON
   -- Arrow   | ON | ON
   -- Dicom   | ON | ON
   -- CUDA    | OFF | OFF

libtiff and pybind11 are required: configuring stops without them. A build without OME-Zarr or
DICOM support refuses such files with ``This version of Nyxus was not build with OmeZarr support``
or ``This version of Nyxus was not build with DICOM support``.

Building the command-line program
---------------------------------

.. code-block:: bash

   cmake -S . -B build-cli -DBUILD_CLI=ON -DALLEXTRAS=ON -DCMAKE_PREFIX_PATH=$NYXUS_DEP_DIR
   cmake --build build-cli --config Release --parallel 4

The executable is ``build-cli/nyxus`` (``build-cli\Release\nyxus.exe`` with a Visual Studio
generator). Use a directory other than ``build``, which ``pip`` uses for the Python package.

CMake options
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Option
     - Default
     - Effect
   * - ``BUILD_CLI``
     - not set (off)
     - Builds the ``nyxus`` command-line executable.
   * - ``BUILD_LIB``
     - not set (off)
     - Builds the Python extension module ``backend``. ``setup.py`` turns it on.
   * - ``NOEXTRAS``
     - ``ON``
     - Turns ``USE_Z5``, ``USE_DCMTK`` and ``USE_ARROW`` off: the build reads TIFF, OME-TIFF and
       uncompressed NIfTI only. Because it is on by default, a ``USE_*`` option has no effect
       unless you also pass ``-DNOEXTRAS=OFF``.
   * - ``ALLEXTRAS``
     - ``OFF``
     - Turns ``USE_Z5``, ``USE_DCMTK`` and ``USE_ARROW`` on. It takes precedence over
       ``NOEXTRAS``. ``setup.py`` always passes it.
   * - ``USE_Z5``
     - ``OFF``
     - OME-Zarr input. Also enables compressed NIfTI (``.nii.gz``), which needs the zlib that this
       option finds. Dropped with a warning when zlib, Blosc, nlohmann_json or the z5 headers
       are missing.
   * - ``USE_DCMTK``
     - ``OFF``
     - DICOM input. Dropped when DCMTK or zlib is missing. JPEG 2000 pixel data is read when
       fmjpeg2k is also found.
   * - ``USE_ARROW``
     - ``OFF``
     - Arrow IPC and Parquet output (``--outputType=arrowipc`` or ``parquet``). Dropped with a
       warning when Arrow or Parquet is missing.
   * - ``USE_STATIC_ARROW``
     - ``OFF``
     - macOS only. Links Arrow and Parquet statically into the Python module and exports only its
       init symbol, so the module's Arrow stays private when pyarrow loads its own Arrow in the
       same process. Configuring stops when the Arrow installation has no static libraries. The
       macOS wheels are built with it on. Elsewhere it is ignored with a warning.
   * - ``USEGPU``
     - ``OFF``
     - Builds the CUDA kernels and the ``--useGpu`` and ``--gpuDeviceID`` options. Configuring
       stops with ``USEGPU flag was set to ON but no CUDA compiler was found`` when no CUDA
       compiler or toolkit is found. The GPU architectures compiled for follow the CUDA major
       version, from 10 to 13.
   * - ``RUN_GTEST``
     - ``OFF``
     - Downloads GoogleTest and builds the C++ test executable ``runAllTests``.
   * - ``RUN_BENCHMARKS``
     - ``OFF``
     - Builds the ``bench_convex_hull`` micro-benchmark (see ``benchmarks/README.md``).
   * - ``CMAKE_BUILD_TYPE``
     - ``Release`` (``Debug`` with MSVC)
     - Single-configuration generators build ``Release`` unless told otherwise. With MSVC, CMake
       itself presets ``Debug``, so pass ``-DCMAKE_BUILD_TYPE=Release`` when you use Ninja or
       NMake. This is a known defect (`#524 <https://github.com/PolusAI/nyxus/issues/524>`__).

The C++ standard is C++20 and is required (``CMAKE_CXX_STANDARD 20``). On MSVC the build adds
``/permissive- /EHsc /bigobj /Zc:__cplusplus``.

Running the tests
-----------------

After installing the Python package, run the Python test suite from the source tree. Use a
checkout of the same version as the installed package (``git checkout v0.12.1`` for 0.12.1): the
tests on the development head cover features that a released package may not have.

.. code-block:: bash

   python -m pip install pytest bfio imagecodecs pyarrow
   python -m pytest tests/python/

With ``-DRUN_GTEST=ON``, the build also produces ``runAllTests``, the C++ test executable. It
does not need ``BUILD_CLI`` or ``BUILD_LIB``.
