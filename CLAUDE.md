# Nyxus - Claude Code Context

## What is Nyxus?
Scalable C++17/Python library for computing 450+ engineered features (intensity, texture, morphology) from segmented/whole-slide images and 3D volumes. Published on PyPI/Conda, Docker-containerized, integrated into WIPP.

## Git Remotes
- `origin` → sameeul/nyxus (fork, push here for PRs)
- `polus_origin` → PolusAI/nyxus (upstream, PRs target this)
- `andre_origin` → friskluft/nyxus
- `jesse_origin` → JesseMckinzie/nyxus

## Build
```bash
# Python module (most common for development)
cmake -DBUILD_LIB=ON ..
# CLI binary
cmake -DBUILD_CLI=ON ..
# Full build with all optional I/O formats
cmake -DALLEXTRAS=ON ..
# GPU support
cmake -DUSEGPU=ON ..
```
- CMake 3.20+, C++17 required
- Core deps: libtiff, pybind11
- Optional: z5/xtensor (OME-Zarr), dcmtk (DICOM), arrow (Parquet/Arrow), CUDA 10-12

## Testing
```bash
pytest -vv tests/python/
```
- `test_nyxus.py` - main test suite
- `test_data.py`, `test_tissuenet_data.py` - test fixtures
- Tests marked `@pytest.mark.arrow` require pyarrow
- Tests marked `@pytest.mark.skip_ci` are GPU tests
- CI uses cibuildwheel: `.github/workflows/build_wheels.yml`

## Architecture
### 3-Phase Pipeline
1. **Phase 1** (`phase1.cpp`): Tile loading, ROI extraction from intensity+segmentation image pairs
2. **Phase 2** (`phase2_2d.cpp`, `phase2_3d.cpp`, `phase2_25d.cpp`): Feature calculation per ROI
3. **Phase 3** (`phase3.cpp`): Output aggregation (CSV, Arrow, Parquet, or buffer for Python)

### Workflow Entry Points
- `workflow_2d_segmented.cpp` - standard 2D with segmentation masks
- `workflow_2d_whole.cpp` - whole-slide (intensity dir == segmentation dir, single ROI per slide)
- `workflow_3d_segmented.cpp` / `workflow_3d_whole.cpp` - 3D equivalents
- `workflow_pythonapi.cpp` - Python API entry point

### Key Data Structures
- `LR` (Label Record): per-ROI data (pixels, contour, image matrix, feature values)
- `Environment`: all configuration, feature sets, results cache, dataset properties
- `roiData`: hashmap label→LR
- `uniqueLabels`: set of ROI labels
- `ResultsCache` (`results_cache.h`): accumulates results for Python API return

### Feature System
- 56+ feature modules in `src/nyx/features/`
- Feature manager (`feature_mgr.h/cpp`): dependency-tracked dispatch with cycle detection
- Feature settings (`Fsettings`): per-feature-family config stored in Environment (`env.fsett_*`)
- `env_features.cpp` (~23K LOC): feature selection/expansion from user input like `*ALL*`, `*ALL_GLCM*`

### Threading Model
- `n_reduce_threads` / `n_feature_calc_threads`: controls parallelism
- Whole-slide workflows use `std::async` with batched thread dispatch
- Output functions (`output_2_buffer.cpp`, `output_2_csv.cpp`, `output_2_apache.cpp`) use static mutexes for thread safety

### Python Bindings
- `src/nyx/python/new_bindings_py.cpp` - pybind11 bindings
- `src/nyx/python/nyxus/nyxus.py` - Python classes (Nyxus, Nyxus3D, Nested, ImageQuality)
- Multiple Environment instances tracked by Python object `id()` via `findenv()`

## Key Source Files (by size/importance)
| File | LOC | Purpose |
|------|-----|---------|
| `environment.cpp` | ~33K | CLI parsing, configuration |
| `env_features.cpp` | ~23K | Feature selection/expansion |
| `env_metaparams.cpp` | ~11K | Metaparameter management (e.g. `glcm/greydepth=100`) |
| `cache.cpp` | ~10K | Memory caching, LRU tile cache |
| `grayscale_tiff.h` | ~27K | OME-TIFF tile loader |
| `reduce_trivial_rois.cpp` | ~800 | Feature extraction dispatch (calls Feature::extract per feature family) |

## Conventions
- Ticket IDs: POL-prefixed (e.g. POL5890)
- Branch naming: `fix/description` or `feature/POL####-description`
- PRs go from `origin` (sameeul fork) to `polus_origin` (PolusAI upstream)
- `gh` CLI is not installed; use GitHub web UI or API for PR creation
- **Shell scripts must use Unix (LF) line endings**, never Windows (CRLF). The `.gitattributes` enforces `*.sh text eol=lf`.

## Known Issues / Past Fixes
- **Dangling `rval` reference in whole-slide threading** (fixed on branch `fix/wsi-rval-dangling-ref`):
  `int rval` was scoped per inner-loop iteration but referenced by `std::async` tasks that outlived it.
  Caused segfault on Python 3.11 Linux. Fix: use `std::vector<int> rvals` with stable lifetime + explicit `future.get()` synchronization.
  Affected files: `workflow_2d_whole.cpp`, `workflow_3d_whole.cpp`.

## Image Format Support
TIFF, OME-TIFF, OME-Zarr, DICOM, NIFTI

## Output Formats
CSV, Apache Arrow IPC, Parquet, in-memory buffer (Python API default)
