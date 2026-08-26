# Regenerating the IMQ goldens

Concrete steps to reproduce every pinned number in `tests/test_imq_{opencv,cellprofiler,regression}.h`
from scratch. Two oracles and one snapshot; the fixture is the same 8×12 ROI for all three.

## The fixture

`im_quality_intensity` + `im_quality_mask` in `tests/test_data.h`, benchmark
`bench_imq_quality_roi`. Nothing needs exporting: every script below parses the two arrays out of
`test_data.h` itself and rebuilds the ROI image matrix the way Nyxus does — the axis-aligned bounding
box of the masked pixels, with positions inside the box but never assigned left at 0.

That last clause is load-bearing. Rows y=7..9 of the intensity literal repeat the *coordinates* of
rows 1..3 for x=3..8, so those 18 positions are never written and stay 0. The ROI's observed minimum
is therefore 0 and not 1, which is what `MIN_SATURATION` counts. A hand-transcribed copy of the ROI
that "fixed" the repeated rows would produce different goldens for four of the six features.

The resulting matrix, 8 wide × 12 tall:

```
1 4 4 1 1 4 1 1        4 4 6 4 1 6 4 1
1 4 6 1 1 6 1 1        1 4 0 0 0 0 0 0
4 1 6 4 1 6 4 1        1 4 0 0 0 0 0 0
4 4 6 4 1 6 4 1        4 1 0 0 0 0 0 0
4 4 6 4 1 6 4 1        4 4 6 4 1 6 4 1
4 4 6 4 1 6 4 1        4 4 6 4 1 6 4 1
                       4 4 6 4 1 6 4 1
```

## `FOCUS_SCORE`, `LOCAL_FOCUS_SCORE` — OpenCV

```
conda activate nyxus_mirp        # already has cv2 4.13.0, numpy 2.4.6, python 3.11.15
python tests/vetting/oracles/gen_imq_opencv.py
```

No dedicated environment: `nyxus_mirp` carries `opencv-python` and that is now recorded in
`TOOLS.md`. The generator prints one line per check and exits non-zero on any failure.

Recipe `imq.laplacian_ksize1_zeropad`. What the tool is asked for:

| | |
|---|---|
| filter | `cv2.Laplacian(roi_float64, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT)` |
| statistic | `ndarray.var()` — population variance, `ddof=0` |
| `LOCAL_FOCUS_SCORE` | the same call on each tile `get_local_focus_score()` visits, summed, divided by `scale*scale` with `scale=2` |

**Name mapping**: none. OpenCV computes a filtered image, not a named feature; the two Nyxus names
are the two ways this generator reduces it.

**Conventions to account for:**

- `ksize=1` in cv2 selects the plain `[[0,1,0],[1,-4,1],[0,1,0]]` stencil. cv2's `ksize=3` is a
  *different, Sobel-derived* Laplacian, scaled differently — using it would produce a value that
  looks close and is not the same statistic.
- `BORDER_CONSTANT` is what matches Nyxus dropping out-of-range taps. `BORDER_REPLICATE`, cv2's
  default, does not.
- `ndarray.var()` is the population variance. `numpy.var(ddof=1)` would be off by `N/(N-1)`.
- **The tile count is part of the recipe, not an implementation detail.** `get_local_focus_score()`
  loops `y < height - M`, so at `scale=2` it visits one 4×6 tile and divides by 4. The generator
  asserts `len(tiles) == 1`; if the loop bound is ever fixed, that check fails and the golden has to
  be re-derived rather than quietly meaning something else.

## `MIN_SATURATION`, `MAX_SATURATION` — CellProfiler

```
conda activate nyxus_cellprofiler
set JAVA_HOME=%CONDA_PREFIX%\Library\lib\jvm
set JDK_HOME=%JAVA_HOME%
set PATH=%JAVA_HOME%\bin;%JAVA_HOME%\bin\server;%PATH%
cd %TEMP%
python <repo>/tests/vetting/oracles/gen_imq_cellprofiler.py
```

Environment build-out is in `TOOLS.md`. Three things that will otherwise waste an hour:

- **`JAVA_HOME`/`JDK_HOME` must be `<env>\Library\lib\jvm`**, and `<jvm>\bin` plus
  `<jvm>\bin\server` must be on `PATH` so `jvm.dll` resolves. Without them the process dies with a
  bare **exit 127** and no Python traceback, which reads as a missing interpreter rather than a
  missing JVM.
- Run from a directory on the **same drive** as the CellProfiler install. Importing
  `cellprofiler.modules.measureimagequality` pulls in `cellprofiler.gui.help.content`, which calls
  `os.path.relpath()` against the cwd and raises `path is on mount 'C:', start on mount 'D:'`.
- `cellprofiler_core.preferences.set_headless()` must run **before** `Measurements()` is
  constructed, or it imports wx.

Recipe `imq.saturation_observed_extremum`. What the tool is asked for:

| | |
|---|---|
| module | `MeasureImageQuality`, `images_choice = O_SELECT`, `check_saturation = True`, everything else off |
| call | `module.calculate_saturation(group, workspace)` on one grayscale image, no mask |
| outputs | `Image_ImageQuality_PercentMinimal`, `Image_ImageQuality_PercentMaximal` |

**Name mapping and the one conversion:**

| CellProfiler | Nyxus | conversion |
|---|---|---|
| `Image_ImageQuality_PercentMinimal` | `MIN_SATURATION` | ÷ 100 |
| `Image_ImageQuality_PercentMaximal` | `MAX_SATURATION` | ÷ 100 |

**That division is why `MAX_SATURATION` is pinned at `0.16666666666666669` and not
`...666`.** Both tools count 16 of 96 pixels, but CP computes `100.0*16/96` and the generator divides
by 100, which lands one ulp above `16/96`. The pin carries CellProfiler's digits. `MIN_SATURATION`
has no gap because 18/96 = 0.1875 is exactly representable.

**Other conventions:** the metric compares pixels to the image's *own* extremum, not to a fixed
bit-depth threshold, so it is scale-invariant — the generator checks that by running the raw and
max-normalized image and requiring the same answer, which is what lets CP's `[0,1]` floats vet Nyxus'
integer pixels. Do **not** point this generator at CP's `FocusScore`/`LocalFocusScore`: those names
exist in CP but denote the normalized variance of the raw image, a different statistic from Nyxus'
variance-of-Laplacian. The generator fails if either name appears in the header's table.

## `POWER_SPECTRUM_SLOPE`, `SHARPNESS` — no oracle

Recipe `imq.regression_quality_roi`. These are Nyxus' own output pinned at full `%.17g` precision as
drift guards, so "regenerating" them means reading them out of a build:

```
cmake --build build-test --target runAllTests
build-test/tests/runAllTests.exe --gtest_filter=*IMQ*
```

and, if a pin has to move, printing the value with `%.17g` rather than copying what a failure
message rounds to. A pin truncated to five digits — which is where `SHARPNESS`' `2.19047` came from
— starts 3.8e-7 away from the value it guards.

**`POWER_SPECTRUM_SLOPE` cannot be regenerated from the algorithm on this fixture**, because on this
fixture the algorithm does not run: `rps()` returns early unless `floor(min(h,w)/8) >= 3` and the ROI
is 8 px wide. The pinned 0 is the guard's return value, which is why its band is `abs=0`. Vetting the
feature needs a benchmark at least 24 px on its short side *and* the radial-binning defect fixed
first — see `matrix/imq.md`.

**`SHARPNESS` has a candidate oracle that has been measured and refuted.** Re-run the comparison with

```
conda activate nyxus_mirp
python tests/vetting/audit/imq_sharpness_reference_dom.py
```

which also re-checks that its Python port still reproduces the C++ pin — the thing that keeps
`imq_pydom_sharpness_vetting_report.md` describing the shipped algorithm.

## Checking the family after any change

```
python tests/vetting/check_coverage.py --check
python tests/vetting/check_test_names.py --check
python tests/vetting/audit/scan_imq_coverage.py --check
```

`scan_imq_coverage.py` also diffs each golden table's keys against the assertions that read them and
against the `TEST()` registrations, and cross-checks the registry's six IMQ features against
`FeatureIMQ` in `featureset.cpp` in both directions.
