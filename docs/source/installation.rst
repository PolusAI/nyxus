.. _installation:

Installation
============

Nyxus comes as a Python package and as a command-line program. The Python packages hold the
Python module only; the command-line program runs from a Docker image or a source build.

Python package
--------------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install nyxus

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge nyxus

* **pip** has wheels for Python 3.10 to 3.13 on Linux (x86_64 and aarch64, glibc 2.28 or newer),
  Windows x86_64 and macOS (Intel and Apple Silicon).
* **conda-forge** has packages for Python 3.11 to 3.14 on Linux x86_64, Windows x86_64 and Intel
  Macs. On Apple Silicon and Linux aarch64, install the pip wheel into a conda environment.

On any other platform, :doc:`build from source <building>`. If ``pip`` or ``conda`` reports that
no package matches, see :ref:`troubleshooting-install`.

The packages read every supported format, including OME-Zarr, DICOM and ``.nii.gz``, and write
Arrow IPC and Parquet. To read Arrow or Parquet output back into Python, also install
``pyarrow``.

GPU package
-----------

.. tab-set::

   .. tab-item:: pip, CUDA 12

      .. code-block:: bash

         pip install nyxus-cuda12x

   .. tab-item:: pip, CUDA 11

      .. code-block:: bash

         pip install nyxus-cuda11x

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge nyxus

* ``nyxus-cuda12x`` has wheels for Linux x86_64 and Windows x86_64, ``nyxus-cuda11x`` for Linux
  x86_64. They need the CUDA runtime and cuFFT of the same CUDA version, and an NVIDIA driver that
  supports it; a CUDA toolkit provides both libraries.
* On Linux x86_64 and Windows x86_64, the conda-forge ``nyxus`` package also has CUDA 12 builds,
  which bring the CUDA libraries with them, and conda usually picks one of them. To get the CPU
  build, add ``"cuda-version<12"`` to the ``conda install`` command.

``nyxus``, ``nyxus-cuda12x`` and ``nyxus-cuda11x`` all install the same ``nyxus`` module, so keep
only one of them in an environment. To switch, uninstall all three first:

.. code-block:: bash

   pip uninstall -y nyxus nyxus-cuda12x nyxus-cuda11x
   pip install nyxus-cuda12x

:doc:`gpu` says which features run on the GPU and how to turn it on.

Docker image
------------

The ``polusai/nyxus`` image holds the command-line program, with every optional format and CUDA
support, for ``linux/amd64``. Each release is tagged with its version, and the newest is also
tagged ``latest``. The container arguments are Nyxus options:

.. code-block:: bash

   docker pull polusai/nyxus:latest
   docker run --rm -v /path/to/experiment:/data polusai/nyxus:latest \
       --intDir=/data/int --segDir=/data/seg --outDir=/data/out \
       --outputType=singlecsv --features="*ALL_INTENSITY*"

Paths are paths inside the container, and the output directory must already exist. To use a GPU,
add ``--gpus all`` to ``docker run`` and ``--useGpu=true`` to the Nyxus options; this needs the
NVIDIA Container Toolkit on the host. When you limit the container's memory with ``docker run
--memory``, also pass ``--ramLimit`` (in megabytes) below that limit, see :doc:`large_images`.

Every option is listed in the :ref:`cli-reference`.

.. _installation-check:

Checking the installation
-------------------------

This script prints the version, whether Arrow IPC and Parquet output are available and whether a
CUDA GPU can be used, and then computes one feature:

.. literalinclude:: ../examples/check_install.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output from a CPU package:

.. code-block:: text

   Nyxus version: 0.12.1
   Arrow IPC and Parquet output: True
   CUDA GPU usable: False
   MEAN of 1, 2, 3, 4: 2.5

For the command-line program, ``nyxus --help`` prints the usage screen.

Building from source
--------------------

To build the Python package or the command-line program yourself, for example on a platform
without a package, see :doc:`building`.
