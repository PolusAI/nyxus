.. _wipp:

WIPP plugin
===========

Nyxus is available as a plugin for `WIPP <https://github.com/usnistgov/WIPP>`__, the Web Image
Processing Pipeline. The plugin runs the Nyxus command-line program in the ``polusai/nyxus``
Docker image of the same version; ``plugin.json`` at the repository root describes it to WIPP.
Each plugin input becomes the command-line option of the same name, listed in the
:ref:`cli-reference`.

Inputs
------

.. list-table::
   :widths: 24 16 60
   :header-rows: 1

   * - Input
     - Required
     - Meaning
   * - ``intDir``
     - yes
     - Collection of intensity images.
   * - ``segDir``
     - yes
     - Collection of label images. Give the intensity collection again for whole-slide mode,
       where each image is one ROI.
   * - ``filePattern``
     - yes
     - Regular expression that selects files in both collections, for example ``.*`` for all
       files or ``.*c0\.ome\.tif`` for files ending in ``c0.ome.tif``.
   * - ``features``
     - yes
     - Feature groups and features to compute, chosen from the list the plugin offers (``*ALL*``,
       ``*ALL_INTENSITY*`` and so on). The groups are described in :doc:`featuregroups`.
   * - ``outputType``
     - yes
     - ``singlecsv`` (one CSV file for all images), ``separatecsv`` (one CSV file per image),
       ``arrowipc`` or ``parquet``. See :doc:`output`.
   * - ``pixelDistance``
     - no
     - Distance in pixels within which two ROIs count as neighbours; 5 by default.
   * - ``pixelsPerCentimeter``
     - no
     - Pixel size, in pixels per centimetre. Accepted, but no feature uses it (see
       :ref:`conventions-units`).
   * - ``embeddedpixelsize``
     - no
     - Accepted for compatibility; the program reads it but no feature uses it.

The plugin writes its results to the output collection ``outDir``, in the format chosen with
``outputType``.
