# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Nyxus uses CMake 3.20+ and C++20. Dependencies are best managed through conda.

### Python package (most common)

```bash
# With conda (Linux/Mac)
conda create -n nyxus_build python=3.12
conda activate nyxus_build
mamba install -y -c conda-forge --file ci-utils/envs/conda_cpp.txt --file ci-utils/envs/conda_py.txt
export NYXUS_DEP_DIR=$CONDA_PREFIX
CMAKE_ARGS="-DALLEXTRAS=ON -DPython_ROOT_DIR=$CONDA_PREFIX -DPython_FIND_VIRTUALENV=ONLY" pip install . -vv

# Minimal build (TIFF + Python only, no Zarr/DICOM/Arrow):
pip install . -vv
```

### CLI binary

```bash
mkdir build && cd build
cmake -DBUILD_CLI=ON -DALLEXTRAS=ON ..
make -j4
# Binary: build/nyxus
```

### Key CMake flags

| Flag | Default | Effect |
|---|---|---|
| `BUILD_CLI=ON` | OFF | Build `nyxus` CLI binary |
| `BUILD_LIB=ON` | OFF | Build Python `backend` module (set by setup.py) |
| `ALLEXTRAS=ON` | OFF | Enable Zarr, DICOM, Arrow/Parquet I/O |
| `NOEXTRAS=ON` | ON | Minimal build (TIFF + pybind11 only) |
| `USEGPU=ON` | OFF | CUDA GPU acceleration |
| `RUN_GTEST=ON` | OFF | Download and build Google Test suite |

## Tests

### Python tests

```bash
# All tests
pytest tests/python/

# Single test
pytest tests/python/test_nyxus.py::TestNyxus::test_featurize_filelist_masked

# Skip Arrow-dependent tests (if Arrow not built)
pytest tests/python/ -m "not arrow"
```

### C++ (Google Test)

```bash
mkdir build && cd build
cmake -DBUILD_CLI=ON -DRUN_GTEST=ON ..
make -j4
./tests/runAllTests
```

## Architecture

### Processing pipeline

The core three-phase pipeline is driven by `workflow_*.cpp` entry points:

1. **Phase 1** (`phase1.cpp`) — Scans image tiles, discovers ROI labels, accumulates per-ROI pixel/metric data into the `roiData` hashmap (`unordered_map<int, LR>`).

2. **Phase 2** (`phase2_2d.cpp`, `phase2_3d.cpp`, `phase2_25d.cpp`) — For each ROI (`LR`), calls feature calculation methods in dependency order as managed by `FeatureManager`. Oversized ROIs (exceeding RAM limit) are processed using out-of-core `OutOfRamPixelCloud`.

3. **Phase 3** (`phase3.cpp`) — Aggregates results into CSV, Arrow IPC, or Parquet output.

Workflow entry points:
- `workflow_2d_segmented.cpp` — 2D paired intensity+mask images
- `workflow_2d_whole.cpp` — 2D whole-slide (single ROI per image)
- `workflow_3d_segmented.cpp` / `workflow_3d_whole.cpp` — 3D volumes
- `workflow_pythonapi.cpp` — Python API dispatch

### Key data structures

- **`LR`** (`roi_cache.h`) — "Label Record", one instance per ROI label. Holds raw pixels, contour, convex hull, image matrix, and all computed feature values. `roiData` is the global `unordered_map<int, LR>`.
- **`Environment`** (`environment.h`) — All configuration: CLI-parsed or Python-set parameters, output options, metaparameters. The Python API maintains one `Environment` per `Nyxus` instance in `pynyxus_cache` (keyed by instance ID).
- **`FeatureSet`** (`featureset.h`) — Enumerates all feature codes as `Feature2D`, `Feature3D`, and `FeatureIMQ` enums. Feature group nicknames (e.g., `*ALL_GLCM*`) are expanded in `env_features.cpp`.

### Feature module system

Each feature (or feature group) is a class that:
1. Inherits from `FeatureMethod` (`feature_method.h`)
2. Declares its feature codes via `static constexpr featureset` (an `initializer_list<Feature2D>` or `Feature3D`)
3. Declares dependencies on other features via `add_dependencies()`
4. Implements `calculate(LR& roi, ...)` for in-RAM ROIs and `osized_calculate(...)` for oversized ROIs
5. Implements `save_value()` to write results into the output table

All feature classes are registered in `feature_mgr_init.cpp` via `FeatureManager::register_feature()`. `FeatureManager::compile()` resolves the dependency graph, detects cycles, and produces the ordered execution list for Phase 2.

Files with a `_nontriv.cpp` suffix implement the out-of-core (oversized ROI) path for the corresponding feature.

### Python API

The Python package lives in `src/nyx/python/nyxus/` and exposes four classes:
- `Nyxus` — 2D feature extraction
- `Nyxus3D` — 3D volume feature extraction
- `ImageQuality` — Image quality metrics
- `Nested` — Hierarchical (parent-child) ROI analysis

The pybind11 C++ bindings are in `src/nyx/python/new_bindings_py.cpp`. Each Python `Nyxus` instance maps to a separate `Environment` object stored in the C++ `pynyxus_cache` map, allowing multiple independent configurations in the same process.

### Configuration subsystem

`environment.cpp` (~33K LOC) handles CLI argument parsing. `env_features.cpp` (~23K LOC) manages feature name recognition and group expansion. `env_metaparams.cpp` handles metaparameters (algorithm-level settings like GLCM angles, Gabor filter parameters). Adding a new CLI parameter requires touching both `environment.h` and `environment.cpp`.

### IBSI compliance

When `ibsi=True` (Python) or `--ibsi=1` (CLI), feature calculations follow the Image Biomarker Standardisation Initiative specification. IBSI feature codes are noted in comments in the feature headers (e.g., `// IBSI # QWB0`).
