# Contributing to Nyxus

Thank you for helping improve Nyxus. This page covers the conventions a change has to follow to be
merged: branches and pull requests, commit messages, building, testing, the feature-validation
framework, line endings and the documentation. For how the code is organized (feature classes,
registration, the analysis pipeline, the GPU kernels and the out-of-core path) read the
[developer guide](docs/source/devguide.rst), published at
<https://nyxus.readthedocs.io/en/latest/devguide.html>.

Bugs and feature requests go to the [issue tracker](https://github.com/PolusAI/nyxus/issues).

## Branches and pull requests

- `main` is protected. Never commit or push to it directly; the repository's pre-commit
  configuration (`.pre-commit-config.yaml`) also refuses commits to `main` locally.
- Cut a branch from `main` and name it after its base: `main-<topic>`, for example
  `main-glcm-offset` (not `fix/...` or `feat/...`).
- Open a pull request against `main` of [PolusAI/nyxus](https://github.com/PolusAI/nyxus). Review
  and merge are done by the maintainers.
- Before you ask for review, run the full Python suite and, for a C++ change, the gtest suite
  (see [Running the tests](#running-the-tests)). CI runs both on Linux, Windows and macOS.

## Commit messages

Releases are made by [release-please](https://github.com/googleapis/release-please), which reads the
commit subjects on `main` to choose the next version and write `CHANGELOG.md`. Every commit that
lands on `main` is read, so **every commit subject, and the pull request title, is a
[Conventional Commit](https://www.conventionalcommits.org/)**:

```
<type>(<optional scope>): <summary in the imperative>
```

for example `fix(imq): score oversized ROIs like in-RAM ones` or
`docs(glcm): correct the angle-averaging formula`. The types and where they appear in the
changelog (`release-please-config.json`):

| type | changelog section |
|------|-------------------|
| `feat` | Features |
| `fix` | Bug Fixes |
| `perf` | Performance Improvements |
| `revert` | Reverts |
| `docs` | Documentation |
| `refactor` | Code Refactoring |
| `build` | Build System |
| `chore`, `style`, `test`, `ci` | not shown |

A breaking change is marked with `!` after the type (`feat(io)!: ...`) or a `BREAKING CHANGE:`
footer. While the version is below 1.0, a breaking change bumps the minor version and a `feat`
bumps the patch version (`bump-minor-pre-major`, `bump-patch-for-minor-pre-major`).

Do not edit version numbers or `CHANGELOG.md` by hand. When the release pull request that
release-please keeps open is merged, the release workflow (`.github/workflows/release-please.yml`)
tags the release and starts the PyPI and Docker publish workflows. The release and wheel builds
run `ci-utils/sync_version.py` to stamp the version into `.version`, `src/nyx/version.h` and
`plugin.json` of the build. The Python
package takes its version from the release tag through `setuptools_scm`.

## Building

Nyxus needs a C++20 compiler and CMake 3.20 or newer. The dependencies are easiest to get from
conda-forge, using the lists in `ci-utils/envs/`:

```bash
conda create -n nyxus-dev -c conda-forge python=3.12 pytest bfio imagecodecs \
    --file ci-utils/envs/conda_cpp.txt --file ci-utils/envs/conda_py.txt
conda activate nyxus-dev
```

On Apple Silicon Macs and Linux aarch64 use `conda_cpp_mac.txt` in place of `conda_cpp.txt`; on
Linux also add `--file ci-utils/envs/conda_linux_compiler.txt`, as the CI builds do.

**Python package.** Build and install the extension into the environment, as CI does
(`setup.py` always configures with `-DALLEXTRAS=ON`):

```bash
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX" python -m pip install . -vv
```

**Command-line tool and C++ tests.**

```bash
cmake -S . -B build-cli -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=ON -DRUN_GTEST=ON -DALLEXTRAS=ON \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build-cli --config Release --parallel
```

Use a directory other than `build`, which `pip` uses for the Python package.

The main CMake options:

| Option | Meaning |
|---|---|
| `-DBUILD_CLI=ON` | build the `nyxus` command-line executable |
| `-DBUILD_LIB=ON` | build the Python extension (`backend` target) |
| `-DRUN_GTEST=ON` | fetch GoogleTest and build the `runAllTests` executable |
| `-DALLEXTRAS=ON` | enable all optional I/O: OME-Zarr (z5), DICOM (dcmtk) and Arrow/Parquet |
| `-DNOEXTRAS=OFF` with `-DUSE_ARROW` / `-DUSE_Z5` / `-DUSE_DCMTK=ON` | enable the optional I/O selectively (`NOEXTRAS` defaults to `ON` and turns all three off) |
| `-DUSEGPU=ON` | build the CUDA kernels; needs a CUDA toolkit that supports the host compiler |
| `-DRUN_BENCHMARKS=ON` | build the microbenchmarks in [`benchmarks/`](benchmarks/README.md) |

**Windows.** Use MSVC, not gcc: conda-forge's Windows packages are built with MSVC, so a gcc build
cannot link them. Run the Visual Studio `vcvars64.bat` first and configure with the Ninja generator,
which also lets `nvcc` use `cl` as its host compiler for a CUDA build:

```bat
cmake -S . -B build-cli -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=Release ^
      -DBUILD_CLI=ON -DALLEXTRAS=ON -DCMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library
cmake --build build-cli
```

Pass `-DCMAKE_BUILD_TYPE=Release`: with MSVC, CMake presets `Debug` for Ninja, and `--config` has
no effect on a single-configuration generator.

If CMake still misses a dependency, set the environment variable `NYXUS_DEP_DIR` to the same
prefix before configuring; `CMakeLists.txt` reads it from the environment, not as a `-D` option.

To run the result, put `%CONDA_PREFIX%\Library\bin` (the dependency DLLs), and for a GPU build the
CUDA toolkit's `bin`, on `PATH`.

Without conda, `ci-utils/install_prereq_linux.sh` and `ci-utils/install_prereq_win.bat` build the
dependencies from source. The "Building from source" section of the
[installation guide](https://nyxus.readthedocs.io/en/latest/installation.html#building-from-source)
has the full recipes.

## Running the tests

**Python.** Run the whole directory, as CI does:

```bash
python -m pytest tests/python/ -vv
```

Run the full directory rather than selected files: some defects only show as state carried from one
test to the next, and running a single file hides them. The markers in `pytest.ini` are `arrow`,
`serial` and `skip_ci`. In a build without Arrow (`ALLEXTRAS=OFF`) the `arrow` tests fail with
"Apache Arrow functionality is not available"; that is expected.

**C++.** Build with `-DRUN_GTEST=ON` and run `runAllTests` from the build directory
(`build-cli/tests/runAllTests`, or `build-cli\tests\Release\runAllTests.exe` with a
multi-config generator such as Visual Studio).
`tests/test_all.cc` is the single gtest `main`; it `#include`s the `test_*.h` headers, which are
deliberately not listed in `tests/CMakeLists.txt`. Test images are in `tests/data/`.

**Validation checks.** CI also runs these, and they need only the Python standard library:

```bash
python tests/vetting/check_coverage.py --check
python tests/vetting/check_test_names.py --check
python tests/vetting/audit/scanlib_selftest.py
python tests/vetting/audit/run_scanners.py
python tests/vetting/report_features.py --check
```

## Tests and feature validation

Nyxus validates feature values against independent reference implementations ("oracles") and
closed-form values, not only against snapshots of its own output. Every new or changed test follows
[`tests/vetting/SPEC.md`](tests/vetting/SPEC.md); [`tests/vetting/README.md`](tests/vetting/README.md)
lists the rest of the framework. In short:

- **Four test kinds, in separate files.** *Oracle* tests compare with an external tool or a closed
  form and are the only tests that establish that a feature is vetted. *Regression* tests pin
  current output to catch drift and never claim correctness. *Invariant* tests assert a required
  relation, such as the out-of-core result equalling the in-memory one. *Mechanics* tests cover
  plumbing: gating, defaults and I/O.
- **Naming.** Test files are `test_<dim>_<family>_<kind-or-oracle>.{h,py}`, for example
  `test_2d_glcm_pyradiomics.py`, `test_3d_glcm_regression.h` or `test_2d_glcm_mechanics.h`. Test
  functions follow the same shape, `test_<dim>_<family>[_<subject>]_<kind-or-oracle>`, and a gtest
  case is `TEST(TEST_NYXUS, <UPPER-CASE function name>)`. The `2d`/`3d` token is mandatory; a file
  whose subject has no image dimensionality is listed in `DIM_AGNOSTIC` in
  `tests/vetting/check_test_names.py` with its reason. Helpers that take arguments are named
  `assert_*`. A family's shared fixtures live in `test_<dim>_<family>_common.h`.
- **Oracle tokens** (SPEC §4): `pyradiomics`, `radiomicsj`, `skimage`, `mirp`, `matlab`,
  `cellprofiler`, `mitk`, `feature2djava`, `wndcharm`, `imea`, `imagej`, `fraclac`, `pydicom`,
  `opencv`, `ibsi`, `analytic`.
- **Goldens are generated offline.** A checked-in generator,
  `tests/vetting/oracles/gen_<family>_<oracle>.*`, runs the reference tool, and its numbers are
  pinned in the test with the tool, version, settings and generator recorded beside them. Reference
  tools are never CI dependencies.
- **The registry.** `tests/vetting/oracle_coverage.csv` has one row per assertion
  (`feature × config_recipe × oracle`) with its `status`: `vetted`, `regression`, `invariant` or
  `untested`. Only a `vetted` row names an `oracle`. A feature counts as vetted when it has at least
  one `vetted` row. Update the row with the test, and regenerate the generated reports:

  ```bash
  python tests/vetting/check_coverage.py --write
  python tests/vetting/report_features.py --write
  ```

- **Config recipes and benchmarks.** A row's `config_recipe` must be defined in
  `tests/vetting/config_recipes.md` and its `benchmark` in `tests/vetting/benchmarks.md`.
- **Tolerances.** Use the tolerance policy in SPEC §7 and give every non-exact tolerance a one-line
  reason. A tolerance loose enough to pass a known-bad value is a bug in the test.

When you change a feature's math, re-vet it against its oracle at the recipe's configuration and
update its registry row; do not just re-baseline a snapshot. A fix comes with a test that fails
without it.

## Code conventions

- Match the naming, density and idiom of the surrounding code.
- A feature's in-memory ("trivial ROI") and out-of-core ("oversized ROI", `osized_*`)
  implementations must return identical values. If you change one, check the other.
- Comments describe what the code does and why it does it that way. The history of a change belongs
  in the commit message.
- Never commit machine-specific content: no absolute paths, user names, personal install locations
  or references to files that exist on one machine only. Use placeholders such as
  `$CONDA_PREFIX`, `<repo>` or `~`.

## Line endings

The repository stores LF line endings everywhere, and `.gitattributes` enforces it: git normalizes
text files on check-in, whatever your editor or `core.autocrlf` setting does. The one exception is
Windows scripts (`*.bat`, `*.cmd`, `*.ps1`), which are checked out with CRLF in the working tree but
stored as LF. Test images and other binary fixtures are marked `binary` so they are never rewritten.

## Documentation

The documentation is built with Sphinx from `docs/source` and published on
[Read the Docs](https://nyxus.readthedocs.io). It builds without the compiled extension, because
`docs/source/conf.py` mocks `nyxus.backend`:

```bash
python -m pip install -r docs/requirements.txt
python docs/tools/gen_cli_reference.py --check
python docs/tools/gen_feature_reference.py --check
python -m sphinx -b html -W --keep-going docs/source docs/build/html
python -m sphinx -b linkcheck -W --keep-going docs/source docs/build/linkcheck
```

These are the steps the docs CI workflow (`.github/workflows/docs.yml`) runs; warnings are
errors. The command-line and feature references are generated from the code: after changing a CLI
option or a feature, run the two generators without `--check` to rewrite their pages.

## Citing Nyxus

If you use Nyxus in your work, please cite it as described in [`CITATION.cff`](CITATION.cff).

## License

By contributing, you agree that your contributions are licensed under the repository's
[MIT License](LICENSE).
