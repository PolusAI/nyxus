# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What Nyxus is

Nyxus is a scalable C++/Python library that computes engineered features
(intensity, texture, morphology, moments, digital-filter) from segmented and
whole-slide/whole-volume images. It computes 450+ 2D features and a large 3D
feature set at the ROI or whole-image level, and is designed to operate at any
scale by assembling ROIs that span multiple image tiles and files.

- **Python API**: classes `Nyxus` (2D), `Nyxus3D` (3D volumes), `Nested`
  (hierarchical parent/child ROIs), `ImageQuality`. Results come back as a
  pandas DataFrame; Arrow IPC and Parquet output are also supported.
- **CLI**: the `nyxus` executable (built with `-DBUILD_CLI=ON`).
- **IO**: 2D from OME-TIFF, OME-Zarr, DICOM; 3D from NIFTI (compressed and
  uncompressed); in-memory NumPy arrays via the Python API.
- Upstream is [PolusAI/nyxus](https://github.com/PolusAI/nyxus). C++17/C++20,
  CMake build system. Version is derived from git tags via `setuptools_scm`.

## Repository layout

```
CMakeLists.txt          top-level build; feature/IO/GPU options
pyproject.toml          Python packaging (package-dir = src/nyx/python)
environment.yml         conda deps
ci-utils/envs/          conda dep lists: conda_cpp.txt, conda_py.txt, conda_gpu.txt, ...
ci-utils/install_prereq_windows.bat / install_prereq_linux.sh   non-conda dep builders
.github/workflows/      CI: build_and_test_{windows,ubuntu,mac}.yml + wheel/publish jobs
docs/                   Sphinx docs (Read the Docs); docs/source/featurelist.rst
tests/python/           pytest suite (this is what CI runs)
tests/vetting/          feature-validation "oracle" framework (see below)
src/nyx/                all C++ sources
```

### `src/nyx/` — the C++ core

- **Entry points**: `main_nyxus.cpp` (CLI `main()`), `python/new_bindings_py.cpp`
  (pybind11 module), `python/nested_roi_py.cpp`.
- **Feature dispatch / registration** (the backbone — read these first when
  adding or debugging a feature):
  - `featureset.h` / `featureset.cpp` — the `Feature2D`, `Feature3D`, and
    `FeatureIMQ` (image-quality) enums (canonical feature codes) and the
    code↔name maps. Every feature has an enum entry here.
  - `feature_method.h/.cpp` — `FeatureMethod` abstract base every feature class
    derives from. Each feature declares `provide_features(...)` /
    `add_dependencies(...)` and implements `calculate()` (trivial/in-RAM ROI)
    and `osized_*()` (out-of-core ROI).
  - `feature_mgr.cpp`, `feature_mgr.h` — the `FeatureManager`: registration,
    1:1 code-correspondence check, cyclic-dependency resolution, user-selection
    compilation, `init_feature_classes()`.
  - **`feature_mgr_init.cpp`** — the concrete registration list
    (`register_feature(new GLCMFeature()); ...` for every 2D/3D feature).
    **Start here to see what is actually wired in.**
  - `env_features.cpp`, `environment*.cpp/h`, `feature_settings.h` — parsing the
    requested feature set / groups (e.g. `*ALL*`, `*ALL_GLCM*`, `*WHOLESLIDE*`,
    `*3D_ALL*`) and all run parameters.
- **Feature implementations**: `src/nyx/features/` (~100 files). Naming:
  - 2D feature classes (e.g. `glcm.*`, `glrlm.*`, `basic_morphology.*`,
    `2d_geomoments.*`, `gabor.*`, `caliper_*`, `convex_hull*`, `contour.*`,
    `erosion.*`, `radial_distribution.*`, `zernike.*`).
  - 3D feature classes are prefixed `3d_` (e.g. `3d_glcm.*`, `3d_intensity.*`,
    `3d_surface.*`, `3d_gldzm.*`).
  - Image-quality features (exposed via the `ImageQuality` Python class):
    `focus_score.*`, `power_spectrum.*`, `saturation.*`, `sharpness.*`.
  - `*_nontriv*.cpp` files hold the **out-of-core ("non-trivial ROI")** variant
    of a feature — the streaming implementation used when an ROI is too large to
    hold in RAM as a dense buffer. The in-RAM ("trivial") path and the
    out-of-core path must produce identical values.
- **ROI model**: `roi_cache*.cpp/h` (`LR` = labeled ROI accumulators),
  `nested_roi.*` (parent/child aggregation), `roi_blacklist.*` (`--skiproi`),
  `cache.*` / `gpucache.cpp`. In-RAM ROI image in `features/image_matrix.*`;
  out-of-core `OutOfRamPixelCloud` in `features/image_matrix_nontriv.*`; 3D in
  `features/image_cube.h`.
- **Analysis pipeline** (three phases, declared in `globals.h`, dispatched from
  `main_nyxus.cpp`): dataset drivers `workflow_2d_segmented.cpp`,
  `workflow_2d_whole.cpp`, `workflow_3d_segmented.cpp`, `workflow_3d_whole.cpp`,
  `workflow_pythonapi.cpp`; then `phase1.cpp` (gather ROI metrics, decide
  trivial vs oversized), `phase2_2d.cpp` / `phase2_25d.cpp` / `phase2_3d.cpp`
  (feature calc), `phase3.cpp` (output). ROIs over the memory limit go through
  `processNontrivialRois()` + each feature's streaming
  `osized_scan_whole_image()` / `osized_add_online_pixel()`; in-RAM ROIs go
  through `processTrivialRois*()`. See also `reduce_trivial_rois.cpp`,
  `pixel_feed.cpp`, `features_calc_workflow.cpp`.
- **Image loading**: `image_loader*.cpp/h`, `raw_image_loader.*`, and format
  backends `raw_tiff.h`, `raw_omezarr.h`, `raw_dicom.h`, `raw_nifti.*`,
  `grayscale_tiff.h`, `abs_tile_loader.h`; NIFTI support under `io/nifti/`.
- **GPU (CUDA)**: `src/nyx/gpu/*.cu/.cuh` — GPU kernels for the compute-heavy
  features (Gabor, erosion, geometric moments). Guarded by `-DUSEGPU=ON` /
  `USE_GPU`. Enabled at runtime with `--useGpu=true` (CLI) or the constructor.
- **Arrow/Parquet output**: `arrow_helpers.cpp`, `arrow_output_stream.*`
  (guarded by `USE_ARROW`).
- **Helpers / 3rd-party**: `helpers/` (thread pool, FFT, timing, least-squares),
  `3rdparty/` (quickhull, Jacobi eigensolver).

### `src/nyx/python/` — Python layer

- `new_bindings_py.cpp` — pybind11 bindings exposing the C++ backend as
  `nyxus.backend`.
- `nyxus/__init__.py` — public exports: `Nyxus`, `Nyxus3D`, `Nested`,
  `ImageQuality`, `gpu_is_available`, `get_gpu_properties`.
- `nyxus/nyxus.py` — the Python-facing wrapper classes (`featurize`,
  `featurize_directory`, `featurize_files`, `get_params`/`set_params`,
  `blacklist_roi`, …).
- `nyxus/functions.py` — GPU helper functions.

## Building

C++17/20 compiler + CMake ≥ 3.20 required. Key CMake options:

| Option | Meaning |
|---|---|
| `-DBUILD_CLI=ON` | build the `nyxus` command-line executable |
| `-DBUILD_LIB=ON` | build the Python extension (`backend` target) |
| `-DALLEXTRAS=ON` | enable **all** IO: Zarr (z5) + DICOM (dcmtk) + Arrow |
| `-DNOEXTRAS=OFF` + `-DUSE_ARROW/USE_Z5/USE_DCMTK=ON` | enable IO backends selectively (NOEXTRAS defaults ON and force-disables them) |
| `-DUSEGPU=ON` | build CUDA GPU kernels (requires a CUDA toolkit matching the host compiler) |

Deps are easiest via conda (`ci-utils/envs/conda_cpp.txt` + `conda_py.txt`); set
`NYXUS_DEP_DIR=$CONDA_PREFIX` (or `$CONDA_PREFIX/Library` on Windows) so CMake
finds them. The README "Building from source" section has full Linux/Windows and
Docker recipes.

### On this Windows machine (verified recipe)

Use **MSVC**, not gcc — conda-forge Windows packages are MSVC-ABI, so a gcc
build cannot link them. Full-feature + CUDA build:

1. Create the env (Miniforge at `C:\Users\dvladi\miniforge3`):
   ```
   conda create -n nyxus_build -c conda-forge python=3.12 \
       --file ci-utils/envs/conda_cpp.txt --file ci-utils/envs/conda_py.txt
   ```
2. **Pin z5py to 2.1.2**: the unpinned list pulls z5py 3.x, which renamed
   `z5/multiarray/xtensor_access.hxx` → `array_access.hxx`; the source still
   includes the old name (fatal C1083). Fix without touching source:
   `conda install -n nyxus_build -c conda-forge z5py=2.1.2`.
3. Configure with the **Ninja** generator so `nvcc` can use `cl` as its host
   compiler (no CUDA MSBuild integration needed): run VS2022 `vcvars64.bat`
   first, then `cmake -G Ninja -DCMAKE_CXX_COMPILER=cl -DBUILD_CLI=ON
   -DALLEXTRAS=ON -DUSEGPU=ON -DCMAKE_PREFIX_PATH=<env>\Library
   -DNYXUS_DEP_DIR=<env>\Library ..` then `cmake --build . --config Release`.
   MSVC / CUDA toolchain paths are recorded in the session memory
   `windows-toolchains.md`.
4. To run, put `<env>\Library\bin` (dependency DLLs) and the CUDA `bin` on PATH.

**Fast Python-extension build** for running the test suite (tiff-only CPU,
~2 min): same as above but `-DBUILD_LIB=ON -DALLEXTRAS=OFF -DUSEGPU=OFF
-DPYTHON_EXECUTABLE=<env>\python.exe -Dpybind11_DIR=<env>\Lib\site-packages\pybind11\share\cmake\pybind11`
and `--target backend`, with `-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE` pointed at
`src\nyx\python\nyxus` so the `.pyd` lands in the package. With `ALLEXTRAS=OFF`
the 7 `@pytest.mark.arrow` tests fail with "Apache Arrow functionality is not
available" — expected, not a regression. To exercise Arrow/Parquet without a slow
CUDA build, use `-DNOEXTRAS=OFF -DUSE_ARROW=ON -DUSE_Z5=OFF -DUSE_DCMTK=OFF`.

> Gotcha: in a Git Bash / MSYS shell, bare `cmd` resolves to an MSYS shim that
> silently no-ops `.bat` files and returns exit 0. Invoke `.bat` files via
> `& "$env:ComSpec" /c <bat>` from the PowerShell tool, or full-path
> `C:\Windows\System32\cmd.exe`.

## Running tests

**Python (`tests/python/`)** — the suite CI runs
(`.github/workflows/build_and_test_*.yml`):

```
python -m pytest tests/python/ -vv
```

Run the **whole directory**, not selected files — some bugs only appear as
cross-test state pollution (e.g. a mutable-default-argument leak surfaced only
because an alphabetically-earlier test seeded shared state). Selecting a single
test can hide such failures. Requires an importable `nyxus.backend` (`.pyd` on
`PYTHONPATH`, dependency DLLs on `PATH`). Markers (`pytest.ini`): `arrow`,
`serial`, `skip_ci`. Key files: `test_nyxus.py` (main API) plus the oracle /
regression suites listed under "Feature validation" below.

**C++ (GoogleTest, `tests/`)** — target `runAllTests` (enabled from the
top-level `CMakeLists.txt`; gtest fetched via `tests/CMakeLists.txt` +
`CMakeLists.txt.gtest`). `tests/test_all.cc` is the single gtest `main` and
`#include`s the many `test_*.h` headers (they are intentionally **not** listed
in CMake — see the note in `tests/CMakeLists.txt`). Headers are grouped by
feature and by oracle: `test_<feat>_ibsi.h` (IBSI conformance),
`test_<feat>_pyradiomics.h`, `test_<feat>_regression.h`,
`test_<feat>_mechanics.h`, and `test_3d_<feat>*.h`; fixtures in `test_data.h`.
Images live in `tests/data/`.

## Feature validation ("oracle") framework

Correctness of feature values is validated against external reference
implementations ("oracles"), not just regression snapshots. This is a
first-class concern in this project.

**Tests must conform to the framework specified in `tests/vetting/`.** The
governing documents are:

- **`tests/vetting/SPEC.md`** — the authoritative spec. Defines the vocabulary,
  the four test kinds (**oracle** / **regression** / **invariant** /
  **mechanics**, kept in separate files), the rule that *vetting is a property of
  a (feature × config × reference) assertion* (only oracle tests establish
  vetting; regression tests never claim it), the tolerance policy, and the
  authoring checklist for adding a vetted feature. Any new or changed test must
  follow it — taxonomy, naming, tolerances, and registry update included.
- `tests/vetting/oracle_coverage.csv` — the single source of truth: one row per
  assertion (`feature × config_recipe × oracle`) with its `outcome`
  (`vetted` / `regression` / `invariant` / `not_tested` / `not_implemented` /
  `dropped_invalid`), tolerance, backing test, and benchmark. A feature counts
  as vetted iff it has ≥1 `vetted` oracle row.
- `tests/vetting/config_recipes.md` — the exact Nyxus + tool settings that make
  a feature directly comparable to a reference (referenced by id from the
  registry). `tests/vetting/matrix/<family>.md` — the config-point matrix and
  per-cell verdict. `tests/vetting/README.md`, `TOOLS.md`, `MIGRATION.md` —
  supporting guides.
- Allowed oracle tokens (SPEC §4): pyradiomics, radiomicsj, skimage, mirp,
  matlab, cellprofiler, mitk, feature2djava, wndcharm, imea, imagej, fraclac,
  pydicom, ibsi, analytic. Oracle goldens are generated **offline** by a
  checked-in generator (`tests/vetting/oracles/gen_<family>_<oracle>.*`) and
  pinned with full provenance — **reference tools are never CI runtime
  dependencies**.
- `tests/vetting/check_coverage.py` validates the registry and regenerates
  `coverage_report.md` (stdlib only). Naming: test files are
  `test_<family>_<kind-or-oracle>.{h,py}` (e.g. `test_glcm_pyradiomics.py`,
  `test_glcm_regression.h`, `test_glcm_mechanics.h`); functions carry the oracle
  suffix so vetting status is self-evident.

When you change a feature's math, re-vet it against its oracle at the recipe's
config and update the registry row — don't just re-baseline a snapshot. A
tolerance loose enough to pass a known-bad value is itself a test bug.

## Conventions for changes

- **Branches** are cut from and named after their base branch as
  `main-<topic>` (not `fix/` or `feat/`).
- **Annotate changed source lines**: when fixing a defect, add a brief comment
  on the changed line(s) explaining the defect and the fix, so the reason
  survives in the code.
- Match the density, naming, and idiom of the surrounding code.
- **Run `pytest tests/python/` (the full directory) locally before proposing a
  push** — it mirrors CI and catches cross-test pollution.
- The trivial (in-RAM) and non-trivial (out-of-core) implementations of a
  feature must return identical values; if you touch one, check the other.
- **Any new or modified test must conform to the framework in
  `tests/vetting/SPEC.md`** — correct test kind (oracle/regression/invariant/
  mechanics), naming, tolerance policy, and an updated `oracle_coverage.csv`
  row. See "Feature validation" above.

### Git

- **Claude must never run `git commit` or `git push`.** Instead, generate the
  exact commands for the user to run themselves.
- **Never add a `Co-Authored-By:` trailer** (or any similar attribution line)
  to commit messages.

## Docs

Feature reference: `docs/source/featurelist.rst`. Rendered docs at
https://nyxus.readthedocs.io. The README has extensive Python/CLI usage
examples (directory/file/NumPy featurization, whole-slide, 3D, nested ROIs,
Hounsfield/CT handling, anisotropy, Arrow/Parquet output).
