Developer guide
===============

This guide is for people who change Nyxus itself: add a feature, a feature group, a setting or a
command-line option, fix a computation, or write tests and documentation for one. The first part is
a step-by-step walkthrough of adding a feature. The rest is reference material that the walkthrough
links to.

All C++ paths are relative to ``src/nyx/`` unless they start with another top-level directory.

.. contents:: On this page
   :local:
   :depth: 1


How a run flows through the code
--------------------------------

A run, whether it is started from the command line or from Python, goes through the same stages.

#. **Parse.** ``Environment`` (``environment.h``) collects the options. The command line goes
   through ``Environment::parse_cmdline`` in ``environment.cpp``; Python goes through
   ``initialize_environment`` and ``set_environment_params_imp`` in
   ``python/new_bindings_py.cpp``.
#. **Select features.** ``Environment::expand_featuregroups`` in ``env_features.cpp`` turns the
   requested names and groups into a set of enabled feature codes in ``FeatureSet``
   (``featureset.h``).
#. **Compile.** ``FeatureManager::compile`` checks that every 2D and 3D feature code has exactly
   one provider and that there are no dependency cycles.
   ``FeatureManager::apply_user_selection`` keeps the feature methods that the enabled codes need.
   ``Environment::compile_feature_settings`` builds one settings vector per feature class.
#. **Scan.** The workflow for the run type (``workflow_2d_segmented.cpp``,
   ``workflow_2d_whole.cpp``, ``workflow_3d_segmented.cpp``, ``workflow_3d_whole.cpp`` or
   ``workflow_pythonapi.cpp``) scans each image pair, builds one ``LR`` record per ROI, and
   estimates each ROI's memory footprint.
#. **Compute.** ROIs whose footprint is below the RAM limit are computed in memory by the
   functions in ``reduce_trivial_rois.cpp``. For file input, larger ROIs are computed out of core
   by ``processNontrivialRois`` (``phase3.cpp``) or ``processNontrivialRois_3D``
   (``phase3_3d.cpp``); in-memory arrays (``workflow_pythonapi.cpp``) and whole slides refuse
   them. See :ref:`devguide-dispatch`.
#. **Output.** The values that each feature wrote into ``LR::fvals`` are written as CSV, Arrow IPC
   or Parquet, or returned to Python as a table (``output_2_csv.cpp``, ``output_2_buffer.cpp``,
   ``output_2_apache.cpp``).


.. _devguide-adding-a-feature:

Adding a feature
----------------

The example adds a 2D feature class ``MyFeature`` that provides two codes, ``MY_FEATURE_A`` and
``MY_FEATURE_B``. The steps for a 3D or image-quality feature are the same; the differences are
noted in each step.

Do the steps in order. Each one names the reference section that explains the code it touches.

Step 1. Add the feature codes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add one enumerator per output code to the matching enum in ``featureset.h``: ``Feature2D``,
``Feature3D`` or ``FeatureIMQ``. Add them immediately before ``_COUNT_``:

.. code-block:: c++

   enum class Feature2D
   {
       COV = 0,
       ...
       IH_BIN_SIZE,

       MY_FEATURE_A,
       MY_FEATURE_B,

       _COUNT_,
       _FIRST_ = COV
   };

Adding a 2D code changes the numeric value of every 3D and image-quality code, because those enums
start where the previous one ends. Never store or compare a feature code as a number. See
:ref:`devguide-feature-codes`.

Step 2. Give the codes user-facing names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add one entry per code to the name map in ``featureset.cpp``:

- 2D: ``Nyxus::UserFacingFeatureNames``;
- 3D: ``Nyxus::UserFacing_3D_featureNames`` (3D names start with ``3``, for example ``3COV``);
- image quality: ``Nyxus::UserFacingIMQFeatureNames``.

.. code-block:: c++

   { "MY_FEATURE_A", Feature2D::MY_FEATURE_A },
   { "MY_FEATURE_B", Feature2D::MY_FEATURE_B },

This name is what users type, what the output column is called, and what the documentation
generator lists.

Step 3. Declare the feature class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``features/my_feature.h``. The class derives from ``FeatureMethod`` and declares a static
``featureset`` list that names every code it provides:

.. code-block:: c++

   #pragma once

   #include <unordered_map>
   #include "../dataset.h"
   #include "../feature_method.h"
   #include "../feature_settings.h"
   #include "../roi_cache.h"

   class MyFeature : public FeatureMethod
   {
   public:
       const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
       {
           Nyxus::Feature2D::MY_FEATURE_A,
           Nyxus::Feature2D::MY_FEATURE_B
       };

       MyFeature();

       // in-RAM path
       void calculate (LR& r, const Fsettings& s);

       // out-of-core path
       void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {}
       void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);

       // results
       void save_value (std::vector<std::vector<double>>& feature_vals);
       void cleanup_instance();

       // hooks used by the dispatcher
       static bool required (const FeatureSet& fs) { return fs.anyEnabled (MyFeature::featureset); }
       static void extract (LR& r, const Fsettings& s);
       static void parallel_process_1_batch (size_t start, size_t end,
           std::vector<int>* ptrLabels, std::unordered_map<int, LR>* ptrLabelData,
           const Fsettings& s, const Dataset& ds);

   private:
       double val_a = 0, val_b = 0;
   };

Write the ``featureset`` list exactly in this form, with fully qualified ``Nyxus::Feature2D::``
names. The group expanders and the documentation generator both read it. See
:ref:`devguide-feature-method`.

Step 4. Declare provided features and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the constructor, pass a short class name to ``FeatureMethod``, list the provided codes, and list
the codes whose values or helper data this class reads:

.. code-block:: c++

   MyFeature::MyFeature() : FeatureMethod ("MyFeature")
   {
       provide_features ({ Nyxus::Feature2D::MY_FEATURE_A, Nyxus::Feature2D::MY_FEATURE_B });
       add_dependencies ({ Nyxus::Feature2D::PERIMETER });   // reads the contour
   }

Every code must have exactly one provider. ``FeatureManager::compile`` rejects a 2D or 3D code with
none or with several; image-quality codes are not checked. Dependencies are listed as feature
codes, not classes. They are used to order and check the feature methods; in the in-RAM path you
also have to place the dispatch block correctly (step 9).

Step 5. Implement the in-RAM calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``features/my_feature.cpp`` and implement ``calculate``, ``save_value``, ``extract`` and
the batch function:

.. code-block:: c++

   void MyFeature::calculate (LR& r, const Fsettings& s)
   {
       for (const Pixel2& px : r.raw_pixels)
       {
           // px.x, px.y are image coordinates, px.inten is the grey level
           ...
       }
       val_a = ...;
       val_b = ...;
   }

   void MyFeature::save_value (std::vector<std::vector<double>>& fvals)
   {
       fvals[(int)Nyxus::Feature2D::MY_FEATURE_A][0] = val_a;
       fvals[(int)Nyxus::Feature2D::MY_FEATURE_B][0] = val_b;
   }

   void MyFeature::extract (LR& r, const Fsettings& s)
   {
       MyFeature f;
       f.calculate (r, s);
       f.save_value (r.fvals);
   }

   void MyFeature::parallel_process_1_batch (size_t start, size_t end,
       std::vector<int>* ptrLabels, std::unordered_map<int, LR>* ptrLabelData,
       const Fsettings& s, const Dataset& ds)
   {
       for (auto i = start; i < end; i++)
       {
           int lab = (*ptrLabels)[i];
           LR& r = (*ptrLabelData)[lab];
           extract (r, s);
       }
   }

The batch function must have the signature that ``runParallel`` (``parallel.h``) expects. Some
classes call it ``reduce``; the name does not matter, the signature does.

Read only the ``LR`` fields listed in :ref:`devguide-lr`, and read settings only through the
``STNGS_*`` macros (:ref:`devguide-settings`). Decide what the feature reports for a
constant-intensity ROI and for an ROI too small for the definition, and report it the same way on
every path.

Step 6. Implement the out-of-core calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An ROI larger than the RAM limit has no ``raw_pixels``; its pixels are in the disk-backed cloud
``LR::raw_pixels_NT``. Implement ``osized_calculate`` so that it gives exactly the same values as
``calculate``. The simplest way is to rebuild the in-RAM data and delegate:

.. code-block:: c++

   void MyFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
   {
       val_a = val_b = 0;                 // this instance is reused across ROIs
       r.rebuild_raw_pixels_from_cloud(); // or rebuild_aux_image_matrix_from_cloud()
       calculate (r, s);
   }

   void MyFeature::cleanup_instance()
   {
       val_a = val_b = 0;
   }

Two things differ from the in-RAM path:

- The out-of-core loop uses one feature-method instance for every oversized ROI, while the in-RAM
  path creates a new instance per ROI in ``extract``. Reset every member at the start of
  ``osized_calculate`` and in ``cleanup_instance``, or one ROI's values leak into the next.
- ``osized_add_online_pixel`` is a pure virtual that no code path calls. Give it an empty body.

For a 3D feature, also add the class to ``is_3d_ooc_supported`` in ``phase3_3d.cpp``. Without it,
the out-of-core 3D path refuses the feature with an error. 3D out-of-core features read
``LR::raw_voxels_NT``.

See :ref:`devguide-ooc-rule`.

Step 7. Register the class
^^^^^^^^^^^^^^^^^^^^^^^^^^

Add one line to the ``FeatureManager`` constructor in ``feature_mgr_init.cpp``, in the 2D, 3D or
image-quality block:

.. code-block:: c++

   register_feature (new MyFeature());

See :ref:`devguide-registration`.

Step 8. Give the class a settings vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each feature class has its own ``Fsettings`` member on ``Environment``.

#. Add ``fsett_MyFeature`` to the ``Fsettings`` list in ``environment.h``.
#. Register it in ``Environment::compile_feature_settings`` (``env_features.cpp``):

   .. code-block:: c++

      f_settings_.push_back (fsett_MyFeature);
          feature2settings_ [typeid(MyFeature)] = f_settings_.size() - 1;

The out-of-core path looks the vector up by the class's ``typeid``. A class that is registered in
step 7 but missing here makes the run fail with "No feature settings are registered for ...". If the
feature needs its own parameter, see :ref:`devguide-settings`.

Step 9. Dispatch the feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The in-RAM, whole-slide and whole-volume paths call feature classes explicitly, in a fixed order, in
``reduce_trivial_rois.cpp``. Add a block to each path that applies:

- segmented 2D: ``reduce_trivial_2d``;
- whole-slide 2D: ``reduce_trivial_2d_wholeslide``;
- segmented 3D: ``reduce_trivial_3d``;
- whole-volume 3D: ``reduce_trivial_3d_wholevolume``.

.. code-block:: c++

   //==== My feature
   if (MyFeature::required (env.theFeatureSet))
   {
       STOPWATCH("Morphology/MyFeature/MF/#4aaaea", "\t=");
       const Fsettings& fst = env.fsett_MyFeature;
       runParallel (MyFeature::parallel_process_1_batch, n_threads, work_per_thread, job_size,
           &L, &env.roiData, fst, env.dataset);
   }

The whole-slide and whole-volume blocks call ``MyFeature::extract (r, s)`` on the single ROI
instead of ``runParallel``.

Place the block after the blocks of the features it depends on. If it reads helper data that
another feature builds, such as the contour or the convex hull, also add ``MyFeature::required``
to the condition of that feature's block. For example, the contour block runs when any of the
contour-based features is requested. Out of core, the feature manager runs the dependencies
first (:ref:`devguide-registration`).

If the feature compares ROIs with each other, it belongs in
``reduce_neighbors_and_dependencies_manual``, which runs once per image after its in-RAM ROIs are
computed and before its oversized ROIs are.

See :ref:`devguide-dispatch`.

Step 10. Add the feature to groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the class to the groups that should include it, in ``Environment::expand_2D_featuregroup``
(or ``expand_3D_featuregroup``) in ``env_features.cpp``. For example, a morphology feature goes into
the ``Fgroup2D::FG2_MORPHOLOGY`` branch:

.. code-block:: c++

   theFeatureSet.enableFeatures (MyFeature::featureset, enable);

``*ALL*`` and ``*3D_ALL*`` need no change. ``*ALL*`` enables every 2D code by range. ``*3D_ALL*``
lists each 3D class, so a new 3D class has to be added to it. See :ref:`devguide-groups`.

Step 11. Output more than one column per code, if needed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LR::fvals`` holds a vector per code. Most codes write element ``[0]`` and produce one output
column. A code that produces several values, like ``GABOR``, ``ZERNIKE2D``, ``FRAC_AT_D`` or
``HISTOGRAM``, resizes its vector in ``save_value`` and needs a branch in every writer that names the
columns and copies the values. The writers are in ``output_2_buffer.cpp`` and ``output_2_csv.cpp``;
search them for ``Feature2D::ZERNIKE2D`` to find every place. The per-angle GLCM and GLRLM codes use
the angle lists instead of a code branch.

Step 12. Add the sources to the build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add ``src/nyx/features/my_feature.cpp`` to the ``set(SOURCE ...)`` list in the top-level
``CMakeLists.txt``. Headers do not need to be listed. The same list builds the command-line
executable, the Python module and the test runner. See :ref:`devguide-build`.

Step 13. Test it
^^^^^^^^^^^^^^^^

Add tests that follow the vetting specification in ``tests/vetting/SPEC.md``:

#. An **oracle** test that compares the values with a reference tool or a closed form, in
   ``test_2d_myfamily_<oracle>.h`` (or ``.py``).
#. An **invariant** test that the out-of-core values equal the in-RAM values.
#. A **mechanics** test for anything the feature gates on: settings, defaults, refusals.
#. One row per assertion in ``tests/vetting/oracle_coverage.csv``, then regenerate the reports
   and run the checks.

See :ref:`devguide-testing`.

Step 14. Document it
^^^^^^^^^^^^^^^^^^^^

#. Add a one-line description of each code to ``docs/feature_descriptions.toml``.
#. If the family has a definition page in ``docs/source/Math``, add the codes to
   ``docs/feature_extras.toml`` and a definition with a ``.. _feature-<code>:`` label, the code in
   lower case with hyphens (``.. _feature-my-feature-a:``), to the page.
   A new family needs a new page, its worked-example script, and entries in the generator's family
   lists.
#. Update the feature-count sentence in ``README.md``, ``docs/source/index.rst`` and the other
   files listed in ``COUNT_PAGES`` in ``docs/tools/gen_feature_reference.py``; the generator
   checks them.
#. Regenerate the reference pages and build the docs with warnings as errors.

See :ref:`devguide-docs`.

Step 15. Commit
^^^^^^^^^^^^^^^

Use a Conventional Commit message, for example ``feat(morphology): add MY_FEATURE_A and
MY_FEATURE_B``. The type decides the release notes section and the version bump. See
:ref:`devguide-release`.


.. _devguide-feature-codes:

Feature codes and names
-----------------------

Enums
^^^^^

``featureset.h`` declares three ``enum class`` types in namespace ``Nyxus``:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Enum
     - First value
     - Contents
   * - ``Feature2D``
     - ``COV = 0``
     - 2D features, in families. The IBSI intensity-histogram codes (``IH_*``) come last.
   * - ``Feature3D``
     - ``COV = (int) Feature2D::_COUNT_``
     - 3D features.
   * - ``FeatureIMQ``
     - ``FOCUS_SCORE = (int) Feature3D::_COUNT_``
     - Image-quality features.

Each enum ends with ``_COUNT_`` and defines ``_FIRST_``. The three enums form one continuous range
of integers, so any code identifies its dimension. Code that is shared by all dimensions uses that
range:

- ``FeatureSet`` keeps one flag per code in ``m_enabledFeatures[(int) FeatureIMQ::_COUNT_]``;
- ``LR::initialize_fvals`` sizes ``fvals`` to ``FeatureIMQ::_COUNT_``;
- ``FeatureMethod`` stores provided codes and dependencies as plain ``int``.

The consequence is that inserting a 2D code shifts every 3D and image-quality code. Codes are
internal: users, output files, tests and documentation refer to features by name. Never hard-code
the numeric value of a code.

The feature groups use the same scheme: ``Fgroup3D`` starts at ``Fgroup2D::_COUNT_`` and
``FgroupIMQ`` at ``Fgroup3D::_COUNT_``.

Name maps
^^^^^^^^^

``featureset.cpp`` maps user-facing names to codes:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Map
     - Holds
   * - ``UserFacingFeatureNames``
     - 2D feature names to ``Feature2D``
   * - ``UserFacing_3D_featureNames``
     - 3D feature names to ``Feature3D``
   * - ``UserFacingIMQFeatureNames``
     - image-quality feature names to ``FeatureIMQ``
   * - ``UserFacing2dFeaturegroupNames``
     - 2D group names, such as ``*ALL_GLCM*``, to ``Fgroup2D``
   * - ``UserFacing3dFeaturegroupNames``
     - 3D group names, such as ``*3D_GLCM*``, to ``Fgroup3D``
   * - ``UserFacingIMQFeaturegroupNames``
     - ``*ALL_IMQ*`` to ``FgroupIMQ``

``FeatureSet::find_2D_FeatureByString``, ``find_3D_FeatureByString`` and their group and IMQ
counterparts look names up in these maps. A leading ``+`` or ``-`` on a 2D or 3D name is the
enable or disable operator; the lookup returns the code negated for ``-``. A code with no entry in
its map is still computed if a group enables it, but it is output under a placeholder name such as
``2D-feature<n>``, so every code needs a name.


.. _devguide-feature-method:

The FeatureMethod class
-----------------------

``FeatureMethod`` (``feature_method.h``) is the base class of every feature class. One class usually
computes a whole family, because the codes of a family share intermediate data.

Virtual interface
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Purpose
   * - ``calculate (LR&, const Fsettings&)``
     - Pure virtual. Computes the features of one in-RAM ROI from its cached data.
   * - ``osized_calculate (LR&, const Fsettings&, ImageLoader&)``
     - Pure virtual. Computes the features of one oversized ROI from its disk-backed data.
   * - ``osized_add_online_pixel (size_t x, size_t y, uint32_t intensity)``
     - Pure virtual, never called. Implement it with an empty body.
   * - ``save_value (std::vector<std::vector<double>>&)``
     - Pure virtual. Copies the computed values into the ROI's ``fvals``.
   * - ``osized_scan_whole_image (LR&, const Fsettings&, const Dataset&, ImageLoader&)``
     - The out-of-core entry point. The default calls ``osized_calculate`` and then
       ``save_value``. Classes that need the dataset's slide properties (intensity, intensity
       histogram) override it.
   * - ``cleanup_instance()``
     - Called after each oversized ROI. The default does nothing; override it to reset state.
   * - ``parallel_process (...)``
     - Virtual with an empty default; the dispatcher does not use it.

The non-virtual ``provide_features`` and ``add_dependencies`` each have an overload for
``Feature2D``, ``Feature3D`` and ``FeatureIMQ`` lists; ``provides`` and ``depends`` query them.
``osized_reduce`` is declared ``final`` with an empty body.

Static hooks
^^^^^^^^^^^^

The dispatcher in ``reduce_trivial_rois.cpp`` calls static members by class name, so every feature
class also defines:

- ``featureset``: a ``constexpr std::initializer_list`` of the codes it provides. The group
  expanders, the IBSI gate and the documentation generator read it.
- ``required (const FeatureSet&)``: whether any of its codes is enabled. Most classes return
  ``fs.anyEnabled (featureset)``.
- ``extract (LR&, const Fsettings&)``: creates an instance, calls ``calculate`` and ``save_value``.
  The whole-slide and whole-volume paths call it directly.
- A batch function, ``parallel_process_1_batch`` or ``reduce``, with the ``functype`` signature from
  ``parallel.h``:

  .. code-block:: c++

     void (size_t start, size_t end, std::vector<int>* ptrLabels,
           std::unordered_map<int, LR>* ptrLabelData,
           const Fsettings& s, const Dataset& ds);

  ``runParallel`` splits the ROI labels between threads and calls it once per slice.

- In GPU builds, optionally a ``gpu_process_all_rois`` function (:ref:`devguide-gpu`).

Where the files are
^^^^^^^^^^^^^^^^^^^

Feature classes live in ``features/``. 3D classes are in files prefixed ``3d_`` and are named
``D3_*``. Some families keep their out-of-core implementation in a separate ``*_nontriv.cpp``
file. Image-quality classes are ``focus_score``, ``power_spectrum``, ``saturation`` and
``sharpness``.


.. _devguide-registration:

Registration and the feature manager
------------------------------------

``FeatureManager`` (``feature_mgr.h``) owns one instance of every feature class.

- The constructor, in ``feature_mgr_init.cpp``, calls ``register_feature (new X())`` for every
  class: 2D first, then 3D, then image quality.
- ``compile()`` runs two checks. ``check_11_correspondence`` requires every 2D and 3D code to be
  provided by exactly one registered class; image-quality codes are not checked yet.
  ``gather_dependencies`` follows each class's dependencies through their providers and rejects a
  cycle.
- ``apply_user_selection (FeatureSet&)`` builds the ordered list of classes the run needs: for
  each enabled code, the providers of its direct dependencies, then its own provider, each once.
  Only direct dependencies are added, so if your dependency has dependencies of its own, list them
  too.
- ``init_feature_classes()`` runs one-time class initialisation. Today only
  ``GaborFeature::init_class`` uses it, and only the command-line entry point calls it.

The ordered list is what the out-of-core loops in ``phase3.cpp`` and ``phase3_3d.cpp`` iterate over.
The in-RAM paths do not use it: they call the classes explicitly in ``reduce_trivial_rois.cpp``.


.. _devguide-dispatch:

Dispatch paths
--------------

Which path computes an ROI depends on the run type and on the ROI's memory footprint.
``LR::get_ram_footprint_estimate`` estimates the footprint, and the workflow compares it with
``Environment::get_ram_limit()`` (set by ``--ramLimit`` or ``ram_limit``; by default half the memory
available at start-up). An ROI whose estimate reaches the limit is oversized.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Path
     - Entry point
     - What it does
   * - In-RAM (trivial), segmented
     - ``reduce_trivial_rois_manual`` in ``reduce_trivial_rois.cpp``, called from ``phase2_2d.cpp``,
       ``phase2_25d.cpp`` and ``phase2_3d.cpp``
     - Calls ``reduce_trivial_2d`` or ``reduce_trivial_3d``. Each feature block checks
       ``X::required`` and runs the class's batch function on all pending ROIs through
       ``runParallel``, with ``--reduceThreads`` threads. The order of the blocks is the
       execution order.
   * - Cross-ROI
     - ``reduce_neighbors_and_dependencies_manual`` in ``reduce_trivial_rois.cpp``
     - Runs once per image, after the in-RAM ROIs and before the out-of-core ones: neighbours, then
       hexagonality and polygonality, then
       the enclosing, inscribing and circumscribing circles. Single-threaded.
   * - Out-of-core (oversized)
     - ``processNontrivialRois`` in ``phase3.cpp`` (2D), ``processNontrivialRois_3D`` and
       ``run_3d_ooc_features`` in ``phase3_3d.cpp`` (3D)
     - Streams the ROI's pixels into ``LR::raw_pixels_NT`` (2D) or ``raw_voxels_NT`` (3D). Then,
       for each class in the feature manager's list, calls ``osized_scan_whole_image`` with the
       class's settings and then ``cleanup_instance``. One instance serves all oversized ROIs. In
       3D, only classes accepted by ``is_3d_ooc_supported`` run; any other raises an error.
   * - Whole-slide 2D
     - ``reduce_trivial_wholeslide`` in ``reduce_trivial_rois.cpp``, called from
       ``workflow_2d_whole.cpp``
     - Treats the whole image as one ROI and calls ``X::extract`` on it, class by class
       (``reduce_trivial_2d_wholeslide``). Neighbour features are not computed. A slide that
       exceeds the RAM limit is refused.
   * - Whole-volume 3D
     - ``reduce_trivial_3d_wholevolume``, called from ``workflow_3d_whole.cpp``
     - The 3D counterpart of whole-slide. An oversized volume goes through ``run_3d_ooc_features``.
   * - GPU
     - The ``#ifdef USE_GPU`` section of ``reduce_trivial_2d``
     - When the build has CUDA and the run has the GPU enabled, uploads ROIs in batches of
       ``gpu_batch_len`` with ``GpusideCache::send_roi_data_gpuside`` and calls
       ``gpu_process_all_rois`` for erosion, intensity moments, shape moments and Gabor. Otherwise
       those four run on the CPU like any other block. See :ref:`devguide-gpu`.

The ``*WHOLESLIDE*`` group (``Fgroup2D::FG2_WHOLESLIDE``) selects the families that make sense on a
whole slide: contour, intensity, the texture families except GLDZM, Gabor, intensity moments,
radial distribution and Zernike.

.. _devguide-ooc-rule:

The out-of-core rule
^^^^^^^^^^^^^^^^^^^^

**Out-of-core values must equal in-RAM values.** The same ROI must give the same feature row
whether it is computed in memory or streamed, apart from floating-point summation order. Users
cannot tell which path computed a row, and the oracle tests vet only the in-RAM path, so the
out-of-core path is correct only by being equal to it.

The rule is enforced by invariant tests that featurize the same input twice, once with a large RAM
limit and once with ``ram_limit=0`` or ``1``, and compare every column:
``tests/python/test_2d_ooc_invariant.py``, ``test_imq_ooc_invariant.py`` and the out-of-core cases
in ``test_ooc_mechanics.py`` and ``tests/test_all.cc``.

The ways to meet it, in order of preference:

#. Delegate. Rebuild the in-RAM data from the cloud with ``LR::rebuild_raw_pixels_from_cloud`` or
   ``LR::rebuild_aux_image_matrix_from_cloud`` and call ``calculate``. Most 2D families do this.
#. Stream with the same arithmetic as ``calculate``, when the ROI cannot be materialised. The 3D
   texture families stream over a window of planes.
#. Apply the same guards. A constant ROI, an empty ROI and a too-small ROI must take the same
   branch on both paths, and every member must be reset, because the out-of-core instance is
   reused.

Where a feature does not meet the rule yet, its definition page states the difference, and a
regression test pins it.


.. _devguide-settings:

Settings and metaparameters
---------------------------

Fsettings
^^^^^^^^^

``feature_settings.h`` defines ``Fsettings`` as ``std::vector<FeatureSetting>``, where
``FeatureSetting`` is a union of ``bool bval``, ``int ival`` and ``double rval``. The vector is
indexed by ``enum class NyxSetting``:

- common slots: ``SOFTNAN``, ``TINY``, ``SINGLEROI``, ``GREYDEPTH``, ``PIXELSIZEUM``,
  ``PIXELDISTANCE``, ``XYRES``, ``USEGPU``, ``VERBOSLVL``, ``IBSI``;
- family slots: ``GLCM_GREYDEPTH``, ``GLCM_OFFSET``, ``GLCM_NUMANG``, ``GLCM_SPARSEINTENS``,
  ``GLDM_GREYDEPTH``, ``NGTDM_GREYDEPTH``, ``NGTDM_RADIUS``, ``GLRLM_GREYDEPTH``,
  ``GLSZM_GREYDEPTH``;
- floating-point image slots: ``FPIMG_TARGET_DR``, ``FPIMG_MIN``, ``FPIMG_MAX``, ``FPIMG_ACTIVE``.

Features read the slots through macros, such as ``STNGS_NAN(s)``, ``STNGS_NGREYS(s)``,
``STNGS_IBSI(s)``, ``STNGS_USEGPU(s)`` and ``STNGS_GLCM_OFFSET(s)``. A feature never reads
``Environment`` directly.

``Environment::compile_feature_settings`` (``env_features.cpp``) registers one vector per feature
class, maps the class's ``typeid`` to it, and fills every vector the same way:

- the common slots from the run's options (``XYRES`` is left at 0);
- ``GLCM_OFFSET = 1``, ``GLCM_GREYDEPTH`` = the coarse grey depth, ``GLCM_NUMANG`` = the number of
  GLCM angles, ``NGTDM_RADIUS = 1``;
- the ``FPIMG_*`` slots from the floating-point image options.

The other family slots start at 0. ``Environment::get_feature_settings (typeid)`` returns a class's
vector and throws if the class has none.

Metaparameters
^^^^^^^^^^^^^^

A metaparameter changes one family slot of one class's settings after initialisation. They are set
from Python with ``set_metaparam ("<family>/<name>=<value>")`` and read with
``get_metaparam ("<family>/<name>")``. Both call ``Environment::set_metaparam`` and
``get_metaparam`` in ``env_metaparams.cpp``. There is no command-line form.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Slot written
   * - ``glcm/greydepth``, ``glcm/offset``
     - ``GLCM_GREYDEPTH``, ``GLCM_OFFSET`` of ``fsett_GLCM``
   * - ``3glcm/greydepth``, ``3glcm/offset``, ``3glcm/numang``, ``3glcm/sparseintensities``
     - the ``GLCM_*`` slots of ``fsett_D3_GLCM``
   * - ``3gldm/greydepth``
     - ``GLDM_GREYDEPTH`` of ``fsett_D3_GLDM``
   * - ``3ngtdm/greydepth``, ``3ngtdm/radius``
     - ``NGTDM_GREYDEPTH``, ``NGTDM_RADIUS`` of ``fsett_D3_NGTDM``
   * - ``3glrlm/greydepth``
     - ``GLRLM_GREYDEPTH`` of ``fsett_D3_GLRLM``
   * - ``3glszm/greydepth``
     - ``GLSZM_GREYDEPTH`` of ``fsett_D3_GLSZM``

Each function returns an error message for an unknown family, an unknown parameter or a value it
cannot parse, and Python raises it. ``set_metaparam`` accepts a key without ``/`` and ignores it;
``get_metaparam`` of such a key raises nothing and returns an undefined value. The two functions
are not symmetric: ``get_metaparam`` reads the ``3glcm/*`` keys from ``fsett_GLCM`` (the 2D vector),
a known defect (`#513 <https://github.com/PolusAI/nyxus/issues/513>`__), and knows the GLSZM key only as ``3glsz/greydepth``. ``glcm/greydepth`` set from Python is stored in
the 2D GLCM settings and is read only by the constant-ROI test: the 2D matrix is binned with the
common ``GREYDEPTH`` slot, so the metaparameter changes no value of a ROI whose binned minimum and
maximum differ. The ``set_metaparam`` and ``get_metaparam`` docstrings in ``nyxus.py`` describe the
same behaviour.

Adding a setting
^^^^^^^^^^^^^^^^

#. Add a slot to ``NyxSetting`` before ``__COUNT__`` and a ``STNGS_*`` macro for it in
   ``feature_settings.h``.
#. Give it a default in the loop at the end of ``compile_feature_settings``, if 0 is not the
   right default.
#. Expose it: as a metaparameter, with a branch in both ``set_metaparam`` and ``get_metaparam``, or
   as an option (:ref:`devguide-cli`, :ref:`devguide-python`) whose value the loop copies in.
#. Read it in the feature only through the macro.
#. Add a mechanics test that the setting reaches the feature, and document it in the family page's
   Settings section.


.. _devguide-groups:

Feature groups
--------------

A group is a name like ``*ALL_GLCM*`` that enables a set of codes. Asterisks are part of the name.

How groups are expanded
^^^^^^^^^^^^^^^^^^^^^^^

- ``Environment::spellcheck_raw_featurelist`` checks every requested name against the feature and
  group maps for the run's dimension, and switches on image-quality mode when an IMQ name is found.
- ``Environment::expand_featuregroups`` starts with every feature disabled and processes the names
  in order. A group name goes to ``expand_2D_featuregroup``, ``expand_3D_featuregroup`` or
  ``expand_IMQ_featuregroup``; anything else is looked up as a single feature and enabled with
  ``FeatureSet::enableFeature``. A ``-`` prefix disables instead of enabling, so
  ``*ALL*,-GABOR`` works. ``-COV`` and ``-*ALL*`` enable instead, because their codes are 0, and
  an image-quality name enables its feature whatever its prefix.
- Each expander branch calls ``theFeatureSet.enableFeatures (X::featureset, enable)`` for the
  classes in the group. ``*ALL*`` enables every ``Feature2D`` code by range; ``*ALL_IMQ*`` enables
  every image-quality code.
- After the loop, if IBSI mode is off, the ``IH_*`` codes are disabled again. They are available
  only with ``--ibsi``.

Adding a group
^^^^^^^^^^^^^^

#. Add an enumerator to ``Fgroup2D`` or ``Fgroup3D`` in ``featureset.h``, before ``_COUNT_``.
#. Add its name to ``UserFacing2dFeaturegroupNames`` or ``UserFacing3dFeaturegroupNames`` in
   ``featureset.cpp``.
#. Add a branch to the expander, in the same form as the others, because the documentation
   generator parses it:

   .. code-block:: c++

      if ((Fgroup2D)fgcode == Fgroup2D::FG2_MYGROUP)
      {
          theFeatureSet.enableFeatures (MyFeature::featureset, enable);
          theFeatureSet.enableFeatures (OtherFeature::featureset, enable);
          return true;
      }

   A group name that is in the map but has no branch passes the name check and then fails the
   run.
#. Run ``python docs/tools/gen_feature_reference.py`` so ``featuregroups.rst`` lists it.
#. Update the hand-written group lists: the docstrings of ``Nyxus`` and ``Nyxus3D`` in
   ``src/nyx/python/nyxus/nyxus.py`` and the ``features`` enum in ``plugin.json``.
#. Add a mechanics test that the group enables exactly the intended codes.


.. _devguide-lr:

The ROI record: LR and BasicLR
------------------------------

Each ROI has one ``LR`` record (``roi_cache.h``), derived from ``BasicLR`` (``roi_cache_basic.h``).
The workflow fills the scan fields before any feature runs; features then read them and write
``fvals``. These are the fields a feature uses:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Contents
   * - ``int label``
     - The ROI's label in the mask. ``BasicLR``.
   * - ``int slide_idx``
     - Index of the ROI's image in ``Dataset::dataset_props``; -1 when there is none. ``BasicLR``.
   * - ``AABB aabb``
     - Bounding box in pixels, as acquired. In the geometric pass of an anisotropic 2D run it is
       the box of the resampled ROI. ``BasicLR``.
   * - ``AABB ph_aabb``
     - Bounding box as scanned, before anisotropy. ``BasicLR``.
   * - ``unsigned int aux_area``
     - Number of ROI pixels (voxels in 3D).
   * - ``PixIntens aux_min, aux_max``
     - Smallest and largest grey level in the ROI. ``aux_min == aux_max`` identifies a constant
       ROI.
   * - ``std::vector<Pixel2> raw_pixels``
     - In-RAM 2D: every ROI pixel with its image coordinates and grey level.
   * - ``std::vector<Pixel3> raw_pixels_3D``
     - In-RAM 3D: every ROI voxel.
   * - ``std::unordered_map<int, std::vector<size_t>> zplanes``
     - In-RAM 3D: indices into ``raw_pixels_3D`` by z.
   * - ``OutOfRamPixelCloud raw_pixels_NT``
     - Out-of-core 2D: the disk-backed pixel cloud.
   * - ``OutOfRamVoxelCloud raw_voxels_NT``
     - Out-of-core 3D: the disk-backed voxel cloud, with z.
   * - ``ImageMatrix aux_image_matrix``
     - In-RAM 2D: the ROI's bounding-box image, built before the features run.
   * - ``SimpleCube<PixIntens> aux_image_cube``
     - In-RAM 3D: the ROI's bounding-box volume.
   * - ``std::vector<std::vector<Pixel2>> multicontour_``
     - The ROI's borders, written by ``ContourFeature``: one closed walk per outer border of each
       fragment and per hole, with ``multicontour_hole_`` marking the holes. Depend on
       ``PERIMETER`` to have it.
   * - ``std::vector<Pixel2> contour_pixels_``, ``std::vector<size_t> contour_border_ends_``
     - The contour pixels, each once, border after border, and where each border's run ends.
       ``ContourFeature`` fills them once per ROI with ``merge_multicontour()``; a feature reads
       them instead of merging the borders itself.
   * - ``std::vector<Pixel2> convHull_CH``
     - The convex hull, written by ``ConvexHullFeature``. Depend on ``CONVEX_HULL_AREA`` to have it.
   * - ``std::vector<int> aux_neighboring_labels``
     - Labels of neighbouring ROIs, written by ``NeighborsFeature``.
   * - ``std::vector<std::vector<StatsReal>> fvals``
     - Output: one vector per feature code, ``FeatureIMQ::_COUNT_`` vectors in all, each
       initialised to ``{0.0}``. ``save_value`` writes here.
   * - ``bool blacklisted``
     - The ROI was excluded with ``--skiproi`` or ``blacklist_roi``.

Helper methods: ``rebuild_raw_pixels_from_cloud`` and ``rebuild_aux_image_matrix_from_cloud``
materialise an oversized ROI's data for an out-of-core path that delegates to ``calculate``.
``get_fvals (code)`` returns a code's values. ``has_bad_data`` always returns ``false``; use
``aux_min == aux_max`` to detect a constant ROI.

``NestedLR``, also derived from ``BasicLR``, holds the parent-child relations for nested ROIs. It
does not hold feature values.


.. _devguide-build:

The build
---------

``CMakeLists.txt`` builds everything from one source list:

- ``set(SOURCE ...)`` lists the ``.cpp`` files shared by every target. A new source file must be
  added here. Headers are not listed.
- ``BUILD_CLI`` builds the ``nyxus`` executable from ``${SOURCE}`` plus ``main_nyxus.cpp``.
- ``BUILD_LIB`` builds the Python module ``backend`` with ``pybind11_add_module`` from
  ``${SOURCE}`` plus ``python/new_bindings_py.cpp``, ``python/nested_roi_py.cpp``,
  ``image_loader1x.cpp`` and ``nested_roi.cpp``.
- ``USEGPU`` defines ``USE_GPU`` and builds the ``nyxus_gpu`` library from ``GPU_SOURCE_FILES``.
- ``RUN_GTEST`` downloads GoogleTest and builds ``runAllTests`` from ``tests/CMakeLists.txt``,
  which reuses ``${SOURCE}``.
- ``RUN_BENCHMARKS`` builds the benchmark executable ``bench_convex_hull``.
- I/O backends: ``USE_Z5`` (OME-Zarr), ``USE_DCMTK`` (DICOM), ``USE_ARROW`` (Arrow and Parquet),
  or ``ALLEXTRAS`` for all three. ``NOEXTRAS`` is on by default and switches the three off, so a
  single backend also needs ``-DNOEXTRAS=OFF``; ``ALLEXTRAS`` overrides it.

The project is C++20. ``pip install .`` builds the Python module through ``setup.py``, which passes
``-DBUILD_LIB=ON -DALLEXTRAS=ON``; extra CMake arguments go in the ``CMAKE_ARGS`` environment
variable.


.. _devguide-cli:

Command-line options
--------------------

An option touches four places in the code and one in the docs.

- ``cli_option_constants.h`` defines each option as ``#define clo_<NAME> "--option"``. Options
  that exist only in some builds are inside ``#ifdef`` blocks, such as ``USE_GPU``.
- ``Environment`` (``environment.h``) holds the raw string of each option, either as its own member
  or inside an option object (``gaborOptions``, ``glcmOptions``, ``nestedOptions``,
  ``fpimageOptions``, ``anisoOptions``, ``resultOptions``, ``gpuOptions``). The option objects are
  in the ``cli_*_options.h/.cpp`` files.
- ``Environment::parse_cmdline`` (``environment.cpp``) matches each argument with
  ``find_string_argument``, which accepts ``--opt=value`` and ``--opt value``. An argument that
  matches no option is an error. After the loop it checks required options and converts and
  validates the values.
- ``Environment::show_cmdline_help`` prints the usage screen.
- ``docs/cli_options.toml`` describes every option for the command-line reference page.

To add an option:

#. Define ``clo_MYOPTION`` in ``cli_option_constants.h``.
#. Add a raw string member to ``Environment``, or a field to the matching ``cli_*_options`` class.
#. Add ``|| find_string_argument (i, clo_MYOPTION, member)`` to the chain in ``parse_cmdline``.
#. Validate and convert it after the loop, or in the option class's ``parse_input``.
#. Add it to ``show_cmdline_help``.
#. Add an ``[[option]]`` entry to ``docs/cli_options.toml`` with the keys ``name``, ``group``,
   ``required``, ``default``, ``default_at`` (``"file:target"``, the variable or call that is
   assigned a non-empty default, for example ``"environment.cpp:n_pixel_distance"``; ``"-"`` opts
   out), ``values``, ``build``, ``summary``, ``notes`` and ``evidence``.
   ``evidence`` names the source lines that apply the default.
#. Run ``python docs/tools/gen_cli_reference.py``. It rewrites ``docs/source/cli_reference.rst``,
   and its ``--check`` mode fails in CI if an option defined in the header has no TOML entry, or the
   other way round.
#. If Python users need it too, follow :ref:`devguide-python`.


.. _devguide-python:

Python bindings
---------------

``python/new_bindings_py.cpp`` defines the pybind11 module ``backend`` (``PYBIND11_MODULE(backend,
m)``). Its functions all take ``instid``, the Python object's ``id``, and look up a separate
``Environment`` per object, so several ``Nyxus`` objects can coexist. The main functions are:

- ``initialize_environment``: applies the constructor arguments, expands the feature groups,
  compiles the feature manager and the feature settings;
- ``set_environment_params_imp`` and ``get_params_imp``: back ``set_params`` and ``get_params``;
- ``set_metaparam_imp`` and ``get_metaparam_imp``: back the metaparameters;
- ``featurize_directory_imp``, ``featurize_montage_imp`` (in-memory arrays, through
  ``workflow_pythonapi.cpp``), ``featurize_fname_lists_imp`` and their 3D variants.

``src/nyx/python/nyxus/nyxus.py`` wraps them in the classes ``Nyxus``, ``Nyxus3D``,
``ImageQuality`` and ``Nested``, which ``nyxus/__init__.py`` exports.

To expose a new parameter:

#. Add an argument to ``initialize_environment`` at the end, with a default, and apply it to the
   ``Environment``.
#. If it can change after construction, add it to ``set_environment_params_imp`` with a sentinel
   value meaning "unchanged".
#. Report it in ``get_params_imp``.
#. In ``nyxus.py``, add the keyword to the constructor's accepted keys, read and validate it, and pass
   it to ``initialize_environment``. For ``set_params``, add it to the accepted parameter list.
#. Repeat for ``Nyxus3D`` and ``ImageQuality`` where it applies.
#. Document it in the class docstring; the API reference is generated from the docstrings.


.. _devguide-gpu:

GPU kernels
-----------

CUDA code is in ``src/nyx/gpu`` and is built only with ``-DUSEGPU=ON``, which defines ``USE_GPU``.
At run time the GPU is used when the user asks for it (``--useGpu=true`` and ``--gpuDeviceID`` on
the command line, ``use_gpu_device`` in Python). All CUDA-dependent C++ code is inside ``#ifdef USE_GPU``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Files
     - Contents
   * - ``gpu.h``, ``gpu_helpers.cu``, ``helper_cuda.h``
     - Error-checking macros (``CHECKERR``, ``OK``), which throw in the Python module and print
       elsewhere; ``NyxusGpu::gpu_initialize``, which selects the device.
   * - ``cache.cu``, ``gpucache.cu``
     - Device memory helpers of ``GpusideCache`` (``cache.h``), which holds the uploaded ROI clouds,
       contours and intermediate buffers. The cache is allocated per dataset in
       ``workflow_2d_segmented.cpp``.
   * - ``erosion.cu``
     - ``EROSIONS_2_VANISH``.
   * - ``geomoments_*.cu``, ``geomoments.cuh``
     - Shape and intensity geometric moments: raw, central, normalized, Hu, weighted.
   * - ``gabor.cu``, ``gabor.cuh``
     - Gabor filtering with cuFFT.
   * - ``glszm.cu``
     - The GLSZM sums, called from ``GLSZMFeature::calculate`` when ``STNGS_USEGPU(s)`` is set.
   * - ``reducers.cu``
     - Shared reductions.

Erosion, the two moment families and Gabor go through the batch path in ``reduce_trivial_2d``:
each has a static ``gpu_process_all_rois`` that reads the uploaded batch and writes ``fvals``.
GLSZM uses the GPU inside its ordinary ``calculate``. Every other feature runs on the CPU in GPU
builds too.

To add GPU support to a feature, add the ``.cu`` file to ``GPU_SOURCE_FILES`` in ``CMakeLists.txt``,
keep the CPU path, and add a test that the GPU and CPU values agree. The GPU path is held to the
same values as the CPU path.


.. _devguide-testing:

Testing
-------

C++ tests (GoogleTest)
^^^^^^^^^^^^^^^^^^^^^^

The C++ tests build into one executable, ``runAllTests``:

.. code-block:: bash

   cmake -B build-cli -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=ON -DRUN_GTEST=ON -DALLEXTRAS=ON
   cmake --build build-cli --config Release --parallel 2
   ./build-cli/tests/runAllTests      # build-cli\tests\Release\runAllTests.exe with a Visual Studio generator

- ``tests/test_all.cc`` holds every ``TEST`` case and the gtest ``main``. It includes every
  ``tests/test_*.h`` header (``test_2d_gabor_skimage.cc`` is the one test source compiled
  separately) and registers each test function with one ``TEST`` case in the suite
  ``TEST_NYXUS``:

  .. code-block:: c++

     #include "test_2d_myfamily_analytic.h"
     ...
     TEST(TEST_NYXUS, TEST_2D_MYFAMILY_DISK_ANALYTIC) { ASSERT_NO_THROW(test_2d_myfamily_disk_analytic()); }

  The case name is the function name in upper case, and each case calls one test function.
- Headers are not listed in ``tests/CMakeLists.txt``; only ``.cc`` files are.
- Image fixtures are in ``tests/data``.

Python tests (pytest)
^^^^^^^^^^^^^^^^^^^^^

Python tests are in ``tests/python`` and need an installed or importable ``nyxus`` module. Run the
whole directory, as CI does:

.. code-block:: bash

   python -m pytest tests/python/ -vv

Running single files can hide failures caused by state that one test leaves for another.
``pytest.ini`` defines the markers ``arrow`` (needs an Arrow build), ``serial`` and ``skip_ci``.

The vetting system
^^^^^^^^^^^^^^^^^^

Nyxus distinguishes a test that shows a value is correct from a test that only shows it did not
change. The rules are in ``tests/vetting/SPEC.md``, and the checks below enforce them.

Every test belongs to one of four kinds, kept in separate files:

.. list-table::
   :header-rows: 1
   :widths: 16 44 20 20

   * - Kind
     - Question it answers
     - Claims correctness
     - File suffix
   * - Oracle
     - Does Nyxus match an external tool or a closed form?
     - Yes
     - ``_<oracle>``
   * - Regression
     - Did Nyxus's output change?
     - No
     - ``_regression``
   * - Invariant
     - Does the output obey a required property or relation, such as out-of-core = in-RAM?
     - Partly
     - ``_invariant``
   * - Mechanics
     - Does the plumbing work: gating, defaults, I/O, refusals?
     - No
     - ``_mechanics``

The oracle tokens are ``pyradiomics``, ``radiomicsj``, ``skimage``, ``mirp``, ``matlab``,
``cellprofiler``, ``mitk``, ``feature2djava``, ``wndcharm``, ``imea``, ``imagej``, ``fraclac``,
``pydicom``, ``opencv``, ``ibsi`` and ``analytic`` (a closed form). Only oracle tests establish
that a feature is vetted, and only for the configuration they run.

**Naming.**

- Files: ``test_<dim>_<family>_<kind-or-oracle>.{h,py}``, for example ``test_2d_glcm_ibsi.h``.
  ``<dim>`` is ``2d`` or ``3d``. Image-quality files use ``imq`` and no dimension, and files with
  no image dimension (writers, environment) use none.
- Functions: ``test_<dim>_<family>[_<subject>]_<oracle-or-kind>``. The function's dimension and
  kind must match its file's.
- gtest cases: ``TEST(TEST_NYXUS, <FUNCTION NAME IN UPPER CASE>)``.
- Helpers: ``assert_*``, taking arguments. A helper called directly by a ``TEST`` is that case's
  test function and takes the ``test_`` name.
- Reference tables: ``<family>_<dim>_<oracle>_ref_vals``, declared ``const`` through the
  ``ref_vals_map`` alias, in the file whose assertions read them. ``_common.h`` headers hold
  fixtures, never reference values. A family's headers are included only from that family (and
  from ``test_all.cc``).
- Fixtures: ``bench_<shape><size>_<property>`` for new inline fixtures (standard phantoms keep
  their names), each listed in ``tests/vetting/benchmarks.md``.
- Every pinned oracle value records the tool, its version, the exact configuration and the
  generator script, ``tests/vetting/oracles/gen_<family>_<oracle>.*``.

**The registry.** ``tests/vetting/oracle_coverage.csv`` has one row per assertion, that is per
(feature, configuration, reference). Its columns are:

``dim, feature, family, status, oracle, agreement, config_recipe, tolerance, current_test,
target_test, candidate_oracle, flag, source, notes, test_name, benchmark``

- ``status`` is ``vetted``, ``regression``, ``invariant`` or ``untested``. A ``vetted`` row names
  an ``oracle``; other rows leave it empty.
- ``config_recipe`` names a ``## <id>`` heading in ``tests/vetting/config_recipes.md``: the Nyxus
  settings and the reference tool's settings that make the two comparable.
- ``tolerance`` is ``exact``, ``abs=<x>`` or ``rel=<x>`` (see SPEC section 7 for when each applies).
- ``test_name`` is the gtest case as ``TEST_NYXUS.<CASE>``; ``benchmark`` names a fixture in
  ``benchmarks.md``.
- ``source`` is ``in-tree``, ``tracker`` or ``audit``.

**Matrices.** ``tests/vetting/matrix/<family>.md`` lists, for a family, the settings that change its
values and, for each combination, whether a reference is valid there and which recipe and test
cover it.

**Adding a vetted feature** (SPEC section 8):

#. Pick the reference tool and the config recipe that makes Nyxus comparable; add the recipe if it
   is new.
#. Pick or define a benchmark fixture that exercises the feature.
#. Generate the reference values offline with a checked-in generator, and record the tool version
   and configuration where the values are defined.
#. Write the assertion in the oracle file, named as above, at the tolerance the SPEC gives.
#. If the same feature is also snapshotted, keep that in the ``_regression`` file.
#. Add the registry rows and regenerate the reports:

   .. code-block:: bash

      python tests/vetting/check_coverage.py --write
      python tests/vetting/report_features.py --write

#. Run the checks:

   .. code-block:: bash

      python tests/vetting/check_coverage.py --check
      python tests/vetting/check_test_names.py --check
      python tests/vetting/audit/run_scanners.py
      python tests/vetting/report_features.py --check

What the checks do:

- ``check_coverage.py`` validates the registry's columns and values, resolves every
  ``config_recipe``, ``benchmark`` and ``test_name``, and with ``--check`` fails if
  ``coverage_report.md`` is not what ``--write`` would produce.
- ``check_test_names.py`` checks file, function, case, helper and table names, file purity and
  cross-family includes. Exceptions live in lists in the script, each with its reason.
- ``audit/run_scanners.py`` runs the per-family scanners that find the assertions in the test tree.
- ``report_features.py`` joins each registry row with what the scanners found and, with
  ``--check``, fails when ``report_output.csv`` or ``report_output.md`` is stale, and on any row
  whose verdict is not ``agree`` or ``unscanned`` and is not in its ``ALLOWLIST``: a vetted row
  with no oracle assertion, a row that claims no oracle while an oracle test asserts the feature,
  a different oracle, the wrong kind of test, or a ``test_name`` that does not assert the feature.
  It also fails on an ``unscanned`` family that is not recorded in ``KNOWN_UNSCANNED`` (empty, so
  every family must be scanned), and on two rows that share the key dim, feature, oracle and
  recipe.

``tests/python/test_vetting_mechanics.py`` runs ``check_coverage.py`` and ``check_test_names.py``
against planted defects to show they report them, and runs the name checker on the tree.

Continuous integration
^^^^^^^^^^^^^^^^^^^^^^

These workflows in ``.github/workflows`` run on pull requests (the publish workflows run only on
demand or from a release):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Workflow
     - Checks
   * - ``build_and_test_ubuntu.yml``
     - Builds with ``-DBUILD_CLI=ON -DRUN_GTEST=ON -DALLEXTRAS=ON`` and runs ``runAllTests`` and
       the command-line tests in ``test_cli_mechanics.py``. Installs the Python package and runs
       ``pytest tests/python/``, then ``check_coverage.py --check``,
       ``check_test_names.py --check``, ``audit/scanlib_selftest.py``,
       ``audit/run_scanners.py`` and ``report_features.py --check``.
   * - ``build_and_test_mac.yml``, ``build_and_test_windows.yml``
     - ``runAllTests`` and ``pytest tests/python/``.
   * - ``asan.yml``
     - Builds ``runAllTests`` in Debug with ``-fsanitize=address,undefined`` and fails on any
       AddressSanitizer, LeakSanitizer or UndefinedBehaviorSanitizer report.
   * - ``docs.yml``
     - On changes to the docs, the Python package, the feature and option sources or the registry:
       both generators with ``--check``, then ``sphinx-build -W --keep-going`` for HTML and for
       ``linkcheck``.
   * - ``build_wheels.yml``, ``build_cuda11_wheels.yml``, ``build_cuda12_wheels.yml``
     - Build the wheels. The CPU wheels run ``pytest tests/python``; the CUDA wheels run
       ``pip check``.

``.pre-commit-config.yaml`` defines local hooks (file checks, Python and reST lint, codespell); CI
does not run them.


.. _devguide-benchmarks:

Benchmarks
----------

``benchmarks/README.md`` describes the benchmark programs and how to run them. They are built with
``-DRUN_BENCHMARKS=ON``:

.. code-block:: bash

   cmake -S . -B build-bench -DRUN_BENCHMARKS=ON -DNOEXTRAS=ON
   cmake --build build-bench --target bench_convex_hull -j
   ./build-bench/bench_convex_hull

To compare a change, build the benchmark from a clean checkout before and after it, and compare the
median times for the same arguments. The benchmark also prints values computed from the result, so
a change that alters the result shows up next to the timing.


.. _devguide-docs:

Documentation
-------------

The documentation is a Sphinx site in ``docs/source``. It does not need the compiled module:
``conf.py`` mocks ``nyxus.backend``. The pages describe the development head, so their title
carries no version number.

Building
^^^^^^^^

Python 3.11 or later is needed. Run the same steps as CI:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python docs/tools/gen_cli_reference.py --check
   python docs/tools/gen_feature_reference.py --check
   python -m sphinx -b html -W --keep-going docs/source docs/build/html

``-W`` turns every warning into an error, and Read the Docs builds with ``fail_on_warning: true``
too.

Generated pages
^^^^^^^^^^^^^^^

Some pages are written by scripts in ``docs/tools`` and must not be edited by hand. Run a script
without ``--check`` to rewrite its pages; with ``--check`` it fails if a page is out of date.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Script
     - Reads
     - Writes
   * - ``gen_feature_reference.py``
     - ``featureset.h``, ``featureset.cpp``, ``features/*.h``, ``env_features.cpp`` (through
       ``feature_registry.py``), ``docs/feature_descriptions.toml``,
       ``docs/feature_extras.toml``, ``tests/vetting/oracle_coverage.csv``
     - ``featurelist.rst``, ``featuregroups.rst``, ``Math/<family>_index.inc``
   * - ``gen_cli_reference.py``
     - ``cli_option_constants.h``, ``docs/cli_options.toml``
     - ``cli_reference.rst``

``gen_feature_reference.py`` fails, and writes nothing, when a code has no description, a
description names no code, a code belongs to no family, a family page lacks a code's entry,
``feature_extras.toml`` lists a code that is not a base code of that family, or a file in
``COUNT_PAGES`` does not state the current feature counts ("758 features in all, 539 for 2D
images, ...").

Documenting a new feature
^^^^^^^^^^^^^^^^^^^^^^^^^

#. Add a one-line description of every code to its family's table in
   ``docs/feature_descriptions.toml``. State what the code computes, checked against the source.
   Tables are named by the family's slug in ``FAMILIES``:

   .. code-block:: toml

      [my_family]
      MY_FEATURE_A = "What MY_FEATURE_A computes, in one sentence."

#. Add the class to its family's class list in ``FAMILIES`` in ``gen_feature_reference.py``. If it
   starts a new family, add a ``FAMILIES`` entry and a note to ``[family_notes]``.
#. If the family has a definition page, add each base code to ``docs/feature_extras.toml`` with its
   name, its IBSI identifier (empty if none) and its value in the page's worked example:

   .. code-block:: toml

      [my_family]
      MY_FEATURE_A = { name = "My feature A", ibsi = "", example = 1.25 }

#. Write the definition on the family page, under a ``.. _feature-my-feature-a:`` label.
#. Update the feature-count sentence in ``README.md``, ``docs/source/index.rst`` and the other
   files listed in ``COUNT_PAGES`` in ``docs/tools/gen_feature_reference.py``; the generator
   checks them.
#. Run ``python docs/tools/gen_feature_reference.py`` and commit the regenerated files.

Feature pages
^^^^^^^^^^^^^

Each family has one page, ``docs/source/Math/f_<family>.rst``, listed in ``Math.rst``. There is no
template file; ``f_erosion.rst`` is the reference example of the structure all pages follow:

#. Title and a short introduction.
#. A facts table: **Feature codes**, **Group**, **Dimensions**, **GPU**, **Out-of-core**, **IBSI**.
#. *How the features are computed* (*How the matrix is built* for texture families), ending with a
   *Worked example* that has a Python tab and a command-line tab.
#. *Notation*, *Settings*, *Output columns*.
#. *Feature index*: a ``.. _<family>-index:`` label and ``.. include:: <family>_index.inc``, the
   generated table.
#. *Features*: one subsection per base code with its ``.. _feature-<code>:`` label, ``:Code:`` and
   ``:Range:`` fields, the definition, and optional **Relations.** and **Possible issues.**
   paragraphs.
#. *Possible issues*, under a ``.. _<family>-possible-issues:`` label, for issues shared by the
   family.
#. *Validation* and *References*.

Describe what the code computes. Where it differs from IBSI or from a reference tool, say so and
give the size of the difference.

Worked examples
^^^^^^^^^^^^^^^

A page's example is a script, ``docs/examples/<family>_worked_example.py``. The page shows the part
between two marker lines:

.. code-block:: rst

   .. literalinclude:: ../../examples/myfamily_worked_example.py
      :language: python
      :start-after: # example-start
      :end-before: # example-end

After ``# example-end`` the script asserts every ``example`` value in ``feature_extras.toml``, so the
page cannot drift from the code. It reads those values from a JSON copy of the file, whose path is
in the ``NYXUS_DOCS_EXTRAS`` environment variable. A script that needs files from
``tests/python/data`` finds them through ``NYXUS_DOCS_DATA``.

``tests/python/test_docs_examples_mechanics.py`` picks up every script in ``docs/examples``
automatically. It checks that each has both markers and does not import ``tomllib``, which Python
3.10, the oldest supported version, lacks. It runs each script in a clean directory
with ``NYXUS_DOCS_DATA`` set to ``tests/python/data`` and ``NYXUS_DOCS_EXTRAS`` pointing at the
JSON copy, which it writes from ``feature_extras.toml``; the script must exit with 0. It runs in
the normal pytest job, which the wheel builds run on every supported Python version.


.. _devguide-release:

Commits and releases
--------------------

Commit messages follow `Conventional Commits <https://www.conventionalcommits.org>`_:
``<type>(<scope>): <summary>``, for example ``fix(glcm): ...`` or ``docs(erosion): ...``. Releases
are made by `release-please <https://github.com/googleapis/release-please>`_ from these messages:

#. On every push to ``main``, ``.github/workflows/release-please.yml`` runs release-please with
   ``release-please-config.json`` and ``.release-please-manifest.json``. It keeps a release pull
   request open that bumps the version in the manifest and in ``CITATION.cff``, whose version line
   carries the ``x-release-please-version`` marker. ``.version`` is also listed under
   ``extra-files`` but has no marker.
#. The commit types decide the bump. Before 1.0, a breaking change bumps the minor version and a
   ``feat`` bumps the patch version (``bump-minor-pre-major`` and
   ``bump-patch-for-minor-pre-major``). They also decide the release-notes section: ``feat``,
   ``fix``, ``perf``, ``revert``, ``docs``, ``refactor`` and ``build`` are listed; ``chore``,
   ``style``, ``test`` and ``ci`` are hidden.
#. Merging the release pull request creates the tag and the GitHub release. The workflow then runs
   ``python ci-utils/sync_version.py --version <version>``, which writes the version to
   ``.version``, ``plugin.json`` (``version`` and the container tag) and
   ``src/nyx/version.h`` (``PROJECT_VER``). It uploads ``.version`` to the release and starts
   ``publish_pypi.yml``, ``publish_cuda11_pypi.yml``, ``publish_cuda12_pypi.yml`` and
   ``publish_docker.yml`` on the tag.

The other version consumers:

- The Python package takes its version from the git tag through ``setuptools_scm``
  (``pyproject.toml``), and ``setup.py`` passes it to the C++ build as ``VERSION_INFO``.
- The wheel workflows run ``sync_version.py`` before building, so ``version.h`` and
  ``plugin.json`` in the build carry the tag's version. ``PROJECT_VER`` is printed in the
  command-line banner.

Without ``--version``, ``sync_version.py`` uses the latest tag from ``git describe --tags``.
