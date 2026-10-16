.. _volumetric:

3D volumes
==========

In 3D mode Nyxus reads each file as a volume and each label value of the label volume as one 3D
ROI. It computes the 3D feature families: intensity, shape and the seven texture families. A file
with several channels or timepoints gives one row per ROI, channel and timepoint.

Featurize a volume
------------------

.. tab-set::

   .. tab-item:: Python

      Use the ``Nyxus3D`` class. ``featurize_files`` takes a list of intensity volumes and the
      matching list of label volumes:

      .. literalinclude:: ../examples/volume_files.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

      Output:

      .. code-block:: text

           intensity_image  ROI_label        3MEAN   3MIN    3MAX
         0   cells.ome.tif          1  2053.729492  100.0  3995.0
         1   cells.ome.tif          2  2033.109000  101.0  3999.0

      ``featurize_directory(intensity_dir, label_dir, file_pattern)`` pairs the volumes of two
      directories by file name. Unlike the 2D class, it has no default ``file_pattern``: pass
      ``".*"`` to read every file.

      ``Nyxus3D`` has no ``mergerois`` argument. Passed anyway, it prints ``Warning: unexpected
      keyword argument(s): mergerois`` and each label still gives its own row.

   .. tab-item:: Command line

      Add ``--dim=3`` to an ordinary run. Without ``--features``, every 3D feature
      (``*3D_ALL*``) is computed:

      .. code-block:: bash

         nyxus --dim=3 --features="*3D_ALL_INTENSITY*" --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

      In 3D the Gabor and nested-ROI options are refused, and ``--intSegMapFile`` is ignored.

      ``--mergerois=true`` is accepted in 3D but has no effect: each label still gives its own
      row. On a label volume with three labels, ``--dim=3 --mergerois=true`` wrote three rows,
      the same as without the option, while the same option on a 2D image merged its labels into
      one row.

``Nyxus3D`` reads files only. It has no ``featurize`` method for numpy arrays. To featurize an
array, write it to an OME-TIFF file as in the example, with ``metadata={"axes": "ZYX"}``.

.. note::

   Give ``Nyxus3D.featurize_files`` paths that have a directory part, such as
   ``volumes/cells.ome.tif`` or ``./cells.ome.tif``. A bare file name such as ``cells.ome.tif`` is
   refused with ``File does not exist: /cells.ome.tif``.

``featurize_files`` needs a list of label files even with ``single_roi=True``. Passing the
intensity list twice, ``featurize_files(volumes, volumes, single_roi=True)``, makes each volume
one ROI of its nonzero voxels: zero voxels are left out. The 3D shape features of such a ROI
describe its bounding box rather than its voxels, a known defect (`#494 <https://github.com/PolusAI/nyxus/issues/494>`__). For whole-volume mode, where the ROI is
every voxel, background included, use ``featurize_directory(volume_dir, None, ".*")`` or
``featurize_directory(volume_dir, volume_dir, ".*")``; see :ref:`large-images-whole`.

3D feature groups
-----------------

``Nyxus3D`` and ``--dim=3`` accept only 3D feature codes, which start with ``3``, and the 3D
groups. A 2D name such as ``MEAN`` or ``*ALL_GLCM*`` is refused with ``is not a valid 3D feature
name``. The 3D groups are:

* ``*3D_ALL*``: every 3D feature;
* ``*3D_ALL_INTENSITY*``: first-order intensity statistics;
* ``*3D_ALL_MORPHOLOGY*``: shape;
* ``*3D_ALL_TEXTURE*``: all seven texture families;
* ``*3D_GLCM*``, ``*3D_GLDM*``, ``*3D_GLDZM*``, ``*3D_GLSZM*``, ``*3D_GLRLM*``, ``*3D_NGLDM*`` and
  ``*3D_NGTDM*``: one texture family each.

:doc:`featuregroups` lists the members of each group. There are no 3D neighbor or 3D moment
features.

Voxel spacing
-------------

Nyxus featurizes a volume on its voxels as acquired; nothing is resampled. When the voxels of a
volume are not cubic, give a scale factor for each axis. The 3D shape features scale their voxel
coordinates by these factors: volumes scale by the product of the factors, and axis lengths are
those of the scaled coordinates. The 3D intensity and texture features do not change.

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         nyx = Nyxus3D(["*3D_ALL*"], anisotropy_x=1.0, anisotropy_y=1.0, anisotropy_z=2.0)

      A factor of zero or below raises ``ValueError``. The factors are set when the object is
      created; ``set_params`` refuses them with ``ValueError: Invalid parameter anisotropy_z.``

   .. tab-item:: Command line

      .. code-block:: bash

         nyxus --dim=3 --anisoz=2 --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

      The command line does not check that the factors are positive.

A factor of 1 counts as unset. With ``anisotropy_z=2.0``, a ROI of 2048 voxels on a cubic grid
reports twice the ``3VOXEL_VOLUME``, while ``3MEAN``, ``3INTEGRATED_INTENSITY`` and the texture
features are unchanged.

To take the spacing from the file, set ``use_physical_spacing=True`` (command line:
``--use-physical-spacing=true``). Nyxus then reads the OME ``PhysicalSizeX``, ``PhysicalSizeY`` and
``PhysicalSizeZ`` of each volume and divides them by the smallest of the three, so the finest axis
keeps a factor of 1. A factor other than 1 in any ``anisotropy_*`` or ``--aniso*`` replaces the
physical spacing on all three axes; the axes not given then use 1. For
example, a volume with a pixel size of 0.5 x 0.5 x 2 µm gets the factors 1, 1 and 4: a ROI of 500
voxels then reports a ``3VOXEL_VOLUME`` of about 2000. Shape features are measured in units of
the finest axis's voxel, not in physical units. The ``phys_x``, ``phys_y`` and ``phys_z`` columns report the
file's pixel size whether or not the option is on.

How a volume is read
--------------------

Nyxus picks the reader from the file extension. :doc:`supported_formats` lists the extensions,
pixel types and codecs of each format; this table shows where the Z axis comes from.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Format
     - Volume
   * - OME-TIFF
     - The Z, C and T sizes and the plane order come from the OME-XML. A plain multi-page TIFF is
       read as one channel and one timepoint, with one Z plane per page.
   * - OME-Zarr
     - The axis order comes from the ``axes`` metadata, so any order of T, C, Z, Y and X works.
   * - NIfTI
     - The third dimension is Z and the fourth is time. The voxel spacing in the header is not
       read, so ``use_physical_spacing`` has no effect.
   * - DICOM
     - Each file is a single 2D plane. A DICOM series cannot be read as a volume; convert it to
       one NIfTI or OME-TIFF file first.

.. warning::

   The command-line reference describes ``{set d+}`` file patterns that assemble a volume from a
   series of 2D slice files, such as ``vol_z{set d+}.ome.tif``. Such a pattern currently matches
   no files, and the run stops with ``No intensity and/or label file pairs to
   process``. Stack the slices into one OME-TIFF file instead.

Channels and timepoints
-----------------------

In 3D, Nyxus processes every channel and every timepoint of a file as a separate volume. The
``t_index`` and ``c_index`` columns tell the rows apart. The label volume is paired with the
frames of the intensity file as described in :doc:`supported_formats`, section "Dimensions,
channels and timepoints".

.. literalinclude:: ../examples/volume_channels.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

      ROI_label  t_index  c_index     3MEAN
   0          1      0.0      0.0   533.128
   1          1      1.0      0.0   556.734
   2          1      0.0      1.0  1533.892
   3          1      1.0      1.0  1547.404
   4          1      0.0      2.0  2535.930
   5          1      1.0      2.0  2540.690
      t_index     3MEAN
   2      0.0  1533.892
   3      1.0  1547.404

No option selects a channel or a timepoint: Nyxus computes all of them, and you keep the rows you
need. In pandas, ``t_index`` and ``c_index`` are floating-point columns like the features.

The 2D ``Nyxus`` class and ``--dim=2`` read one channel and one timepoint. They refuse an OME-TIFF
or OME-Zarr file with more, with the error ``carries 2 channels and 1 timepoints; 2D
featurization reads one channel and one timepoint``. To compute 2D features of one channel, save
that plane to its own file first.

Large volumes
-------------

A ROI whose estimated memory footprint reaches the RAM limit is computed out of core: Nyxus
streams its voxels from the file one read at a time. Every 3D family supports this path, and it
gives the same values as the in-memory path, the axis features up to the last digits (see
:doc:`Math/f_morphology_3d`). A whole volume larger than the limit is
streamed the same way. When a single read of the intensity or label file returns the whole
volume, as with a NIfTI file or an OME-Zarr array chunked over all of Z, the volume cannot be
streamed and the run stops with an error. See :doc:`large_images` for the RAM limit and its
options.
