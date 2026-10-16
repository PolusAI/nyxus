.. _installation:

Installation
============

Nyxus comes as a Python package and as a command-line program. The Python packages on PyPI and
conda-forge hold the Python module only; run the command-line program from a Docker image, or
build either one from source.

Python packages
---------------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install nyxus

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge nyxus

   .. tab-item:: pip, CUDA 12

      .. code-block:: bash

         pip install nyxus-cuda12x

The PyPI wheels are built for CPython 3.10, 3.11, 3.12 and 3.13 on these platforms:

* Linux x86_64 and aarch64, glibc 2.28 or newer (``manylinux_2_28``);
* Windows x86_64;
* macOS x86_64 (10.15 or newer) and arm64 (11.0 or newer).

PyPI has no source distribution, so where no wheel fits, ``pip`` falls back to an older release
without a warning, or stops with ``No matching distribution found for nyxus``:

* Release 0.11.0 has no wheel for CPython 3.13 on Linux x86_64, so ``pip install nyxus`` installs
  0.10.0 there. Use Python 3.10 to 3.12, conda-forge or a source build to get 0.11.0. This is a known
  issue (`#525 <https://github.com/PolusAI/nyxus/issues/525>`__).
* On Linux with a glibc older than 2.28 (for example CentOS 7 or Ubuntu 18.04), ``pip`` installs
  0.8.3, the last release built for ``manylinux2014``, on x86_64 with Python 3.8 to 3.12, and an
  even older release with Python 3.6 or 3.7. With Python 3.13 or on aarch64 it stops with ``No
  matching distribution found``. The conda-forge Linux x86_64 packages need glibc 2.17 or newer,
  so use conda on these systems; on aarch64, build from source.
* On Python 3.9, ``pip`` installs 0.9.3, the last release built for it. On Python 3.8 it installs
  0.8.3 on Linux x86_64, Windows and Intel Macs, and stops with ``No matching distribution found``
  on Apple Silicon Macs and Linux aarch64.
* Python 3.14 has no wheel of any release, so ``pip`` stops with ``No matching distribution found
  for nyxus``; the conda-forge package has Python 3.14 builds.
* There is no wheel for other platforms, such as Alpine and other musl-based Linux, Windows on ARM
  or 32-bit systems. Build from source there.

Check the installed version with ``python -c "import nyxus; print(nyxus.__version__)"``.

The conda-forge package is built for Linux x86_64, Windows x86_64 and macOS x86_64, for Python 3.10
to 3.14. It has no build for Apple Silicon Macs or Linux aarch64, where ``conda install`` reports
that the package is not available from the current channels (``PackagesNotFoundError``,
``PackagesNotFoundInChannelsError`` in recent conda); install the PyPI wheel into the conda
environment there:

.. code-block:: bash

   conda create -n nyxus -c conda-forge python=3.12 numpy pandas
   conda activate nyxus
   pip install nyxus

The Python package needs NumPy and pandas, which ``pip`` and ``conda`` install with it; NumPy 1
and 2 both work. To read Arrow IPC or Parquet output back into Python, also install ``pyarrow``.

What the packages read
^^^^^^^^^^^^^^^^^^^^^^

Every package build goes through ``setup.py``, which configures CMake with ``-DALLEXTRAS=ON``.
The PyPI and conda-forge packages therefore read DICOM files and write Arrow IPC and Parquet
output, in addition to TIFF, OME-TIFF and NIfTI. They also read OME-Zarr and compressed NIfTI
(``.nii.gz``), with these exceptions:

* The Windows wheels on PyPI (``nyxus`` 0.9.3 and 0.11.0, and ``nyxus-cuda12x`` 0.9.3) were built
  without OME-Zarr support, and so without ``.nii.gz``: they refuse OME-Zarr and fail on
  ``.nii.gz`` files. This is a known issue (`#527 <https://github.com/PolusAI/nyxus/issues/527>`__).
* The first conda-forge Windows builds of 0.11.0 (build number 0, a build string ending in ``_0``
  in ``conda list nyxus``) read none of OME-Zarr, ``.nii.gz`` and DICOM; conda installs one of
  them into an environment that holds ``libarrow`` 19 or 20.

These exceptions concern the published packages. What a source build reads depends on the
libraries it finds (see `Building from source`_).

.. warning::

   The conda-forge Windows packages of 0.11.0 were built in Debug mode: their extension module
   imports the Visual Studio debug runtime (``MSVCP140D.dll``, ``VCRUNTIME140D.dll``,
   ``ucrtbased.dll``), which only an installation of Visual Studio provides, not the Visual C++
   redistributable. Without Visual Studio, ``import nyxus`` fails. This is a known issue
   (`#528 <https://github.com/PolusAI/nyxus/issues/528>`__).

GPU packages
^^^^^^^^^^^^

``nyxus-cuda12x`` is the same package built with ``-DUSEGPU=ON`` against CUDA 12, and
``nyxus-cuda11x`` the same against CUDA 11. Both are published as Linux x86_64 wheels. The newest
``nyxus-cuda12x`` release with Windows wheels is 0.9.3, for CPython 3.9 to 3.12, a known issue
(`#526 <https://github.com/PolusAI/nyxus/issues/526>`__). On Windows with Python 3.13 or newer,
``pip install nyxus-cuda12x`` therefore stops with ``No matching distribution found for
nyxus-cuda12x``; use Python 3.9 to 3.12 there, or conda-forge with Python 3.10 to 3.13, where conda
picks the CUDA build (it needs Visual Studio, see the warning above).

The Linux wheels do not bundle the CUDA runtime or cuFFT, and the Windows wheels bundle the
runtime but not cuFFT: install the CUDA runtime and cuFFT libraries of the matching CUDA version,
and an NVIDIA driver that supports it. Without them, ``import nyxus`` fails: on Linux with
``ImportError: libcufft.so.11: cannot open shared object file`` (``libcufft.so.10`` for
``nyxus-cuda11x``), on Windows with ``ImportError: DLL load failed while importing backend`` (the
missing file is ``cufft64_11.dll``). On Linux, a CUDA toolkit of the package's major version
provides both libraries, with its ``lib64`` directory on ``LD_LIBRARY_PATH``.

``nyxus``, ``nyxus-cuda12x`` and ``nyxus-cuda11x`` all install the module ``nyxus``, so an
environment holds only one of them. To change from one to another, remove all of them and install
the one you want (check first that a wheel of it exists for your platform and Python, or the
environment is left without Nyxus):

.. code-block:: bash

   pip uninstall -y nyxus nyxus-cuda12x nyxus-cuda11x
   pip install nyxus-cuda12x

The conda-forge ``nyxus`` package has CUDA 12.9 builds for Linux x86_64 and Windows x86_64 next to
its CPU builds. A CUDA build depends on the conda packages ``cuda-cudart`` and ``libcufft`` (about
155 MB), so conda installs the runtime with it. On those two platforms conda currently picks a
CUDA build, both on a machine without a GPU and on one whose driver supports CUDA 12, and the CPU
build for an older driver. The exception is Python 3.14 on Windows, where it picks the CPU build.
To get the CPU build everywhere, ask for it:

.. code-block:: bash

   conda install -c conda-forge nyxus "cuda-version<12"

``conda list cuda-cudart`` in the environment shows whether a CUDA runtime was installed.
:doc:`gpu` says which features run on the GPU and how to turn it on.

Docker image
------------

The ``polusai/nyxus`` image on Docker Hub holds the command-line program, built with all
optional formats and with CUDA support. The images of releases 0.9.2 to 0.11.0 are built on
``nvidia/cuda:11.5.2-runtime-ubuntu20.04`` (CUDA 11.5), images 0.4.0 to 0.8.3 on CUDA 11.0, and
the tags up to 0.2.3 on one older image without CUDA. The ``Dockerfile`` in the repository uses
``nvidia/cuda:12.6.0-runtime-ubuntu22.04``, and images built from it get that base. The images are
built for ``linux/amd64`` only: on Apple Silicon, Docker runs them under emulation, and on other
ARM hosts the command-line program has to be built from source. Each release since 0.9.2 is
tagged with its version number, for example ``polusai/nyxus:0.11.0``, and the newest release is
also tagged ``latest``. The image's entry point is the ``nyxus`` executable, so the container
arguments are Nyxus options.

#. Pull a release:

   .. code-block:: bash

      docker pull polusai/nyxus:0.11.0

#. Mount the directory that holds your images, and pass paths inside the container. The output
   directory (``/path/to/experiment/out`` here) must already exist:

   .. code-block:: bash

      docker run --rm -v /path/to/experiment:/data polusai/nyxus:0.11.0 \
          --intDir=/data/int --segDir=/data/seg --outDir=/data/out \
          --outputType=singlecsv --features="*ALL_INTENSITY*"

#. To use a GPU, add ``--gpus all`` to ``docker run`` and ``--useGpu=true`` to the Nyxus options.
   This needs the NVIDIA Container Toolkit on the host.

#. Nyxus sets its default memory budget from the host's memory, not from a limit given with
   ``docker run --memory``. When you set such a limit, also pass ``--ramLimit`` with a value in
   megabytes below it, so that large ROIs are processed out of core rather than exceeding the
   container's memory (see :doc:`large_images`).

Every option is listed in the :ref:`cli-reference`.

.. _installation-check:

Checking the installation
-------------------------

The script below prints the package version, whether Arrow IPC and Parquet output are
available, and whether a CUDA GPU can be used, and then computes one feature.

.. literalinclude:: ../examples/check_install.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output from a CPU package:

.. code-block:: text

   Nyxus version: 0.11.0
   Arrow IPC and Parquet output: True
   CUDA GPU usable: False
   MEAN of 1, 2, 3, 4: 2.5

``gpu_is_available`` takes the ``id`` of a ``Nyxus`` object. It returns ``False`` for a package
built without CUDA, and otherwise ``True`` when at least one CUDA device is found.

The command-line program has no ``--version`` option. It prints its version in the first line of
every run, whatever the ``--verbose`` level, for example
``Nyxus /// 0.11.0 /// (c) 2021-2025 Axle Research Build of <date>``. ``nyxus --help`` prints the
usage screen.

No call reports whether OME-Zarr or DICOM support is compiled in. A build without it refuses
such a file with ``This version of Nyxus was not build with OmeZarr support`` or ``This version
of Nyxus was not build with DICOM support``. When you build from source, the CMake configure step
prints which optional parts were requested and which were found:

.. code-block:: text

   -- Nyxus Dependencies Summary
   -- ================================
   -- Support | Requested | Found
   -- ================================
   -- Z5      | ON | ON
   -- Arrow   | ON | ON
   -- Dicom   | ON | ON
   -- CUDA    | OFF | OFF

Building from source
--------------------

Build from a git clone of the repository. The package version comes from the git tags, so a
GitHub source archive of the development head or a copy without the ``.git`` directory stops with
``setuptools-scm was unable to detect version``:

.. code-block:: bash

   git clone https://github.com/PolusAI/nyxus.git
   cd nyxus

The ``ci-utils`` and ``tests`` paths on this page are relative to this directory. The build steps
below describe the development head; a release tag can differ (0.11.0, for example, is built as
C++17 and has no ``ci-utils/envs/conda_cpp_mac.txt``).

Requirements
^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^

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
      errors before building.

      The script's Boost download, in its OME-Zarr part, currently fails: the URL returns HTTP
      404. The Windows release wheels built with the script since 0.9.3 have no OME-Zarr support.
      This is a known issue (`#527 <https://github.com/PolusAI/nyxus/issues/527>`__).

      The dependency DLLs in ``local_install\bin`` must be on ``PATH`` when you run the
      command-line program. Python does not search ``PATH`` for an extension module's DLLs, so in
      Python call ``os.add_dll_directory(r"<source tree>\local_install\bin")`` before
      ``import nyxus``.

Building the Python package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pip`` builds the package through ``setup.py`` (see the CMake options below). With
``NYXUS_DEP_DIR`` set as above, run from the source tree:

.. code-block:: bash

   python -m pip install . -vv

Pass more CMake options in the ``CMAKE_ARGS`` environment variable, separated by spaces. For
example, a GPU build, which needs a CUDA toolkit: ``CMAKE_ARGS="-DUSEGPU=ON" python -m pip install
. -vv`` in bash, or ``set CMAKE_ARGS=-DUSEGPU=ON`` before the ``pip`` command in a Windows command
prompt.

The package then includes every optional part whose libraries CMake finds. A missing optional
dependency switches its part off and the build continues, except the GPU part: without a CUDA
toolkit, configuring stops. Not every such case prints a warning (a missing DCMTK drops DICOM with
only status lines), so check the dependency summary at the end of the configure output. libtiff
and pybind11 are required: configuring stops without them.

Building the command-line program
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cmake -S . -B build-cli -DBUILD_CLI=ON -DALLEXTRAS=ON -DCMAKE_PREFIX_PATH=$NYXUS_DEP_DIR
   cmake --build build-cli --config Release --parallel 4

The executable is ``build-cli/nyxus`` (``build-cli\Release\nyxus.exe`` with a Visual Studio
generator). Use a directory other than ``build``, which ``pip`` uses for the Python package.

CMake options
^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

After installing the Python package, run the Python test suite from the source tree. Use a
checkout of the same version as the installed package (``git checkout 0.11.0`` for 0.11.0): the
tests on the development head cover features that a released package may not have.

.. code-block:: bash

   python -m pip install pytest bfio imagecodecs pyarrow
   python -m pytest tests/python/

With ``-DRUN_GTEST=ON``, the build also produces ``runAllTests``, the C++ test executable. It
does not need ``BUILD_CLI`` or ``BUILD_LIB``.
