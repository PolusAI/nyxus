# Audit: IMQ focus scores vs a fresh OpenCV run

**Verdict: both quantities reproduce**, each to 7.1e-15 absolute — and the *filtered image*
they are computed from is equal to OpenCV's cell for cell, which is the stronger of the two results.

Covers `tests/test_imq_opencv.h` (goldens + assertions), `tests/test_imq_common.h` (fixture) and
`tests/vetting/oracles/gen_imq_opencv.py` (generator).

## Method

- **Tool**: OpenCV **4.13.0** (`opencv-python`), numpy 2.4.6, Python 3.11.15, in the existing
  `nyxus_mirp` conda env — no new environment was needed, which is now recorded in `TOOLS.md`.
- **Config**: `cv2.Laplacian(src, ddepth=cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT)`
  followed by `ndarray.var()` (population variance, `ddof=0`). Nyxus side: recipe
  `imq.laplacian_ksize1_zeropad`. Neither side has a setting to match — `FocusScoreFeature` reads no
  `NyxSetting` at all.
- **Fixture**: `im_quality_intensity` / `im_quality_mask` exactly as `tests/test_data.h` stores
  them, parsed out of that header by the generator — so the generator, the C++ test and cv2 are fed
  one copy of the pixels and cannot drift apart. One 8×12 ROI, 96 pixels, grey values {0, 1, 4, 6}.
- **Command**: `python tests/vetting/oracles/gen_imq_opencv.py`; full steps in `imq_golden_regen.md`.

## Result table

| feature | pinned (OpenCV) | Nyxus | abs | rel | verdict |
|---|---|---|---:|---:|---|
| `FOCUS_SCORE` | 34.956597222222221 | 34.956597222222229 | 7.1e-15 | 2.0e-16 | vetted |
| `LOCAL_FOCUS_SCORE` | 28.341145833333336 | 28.341145833333343 | 7.1e-15 | 2.5e-16 | vetted |

Both assert at SPEC §7's exact tier, an **absolute** 1e-9 band via `ASSERT_NEAR`. The tier applies
for the reason the SPEC gives it: nothing but float summation order separates the two sides. It is
agreement, not bit identity — both residuals are non-zero.

## The convolution is proved, not inferred

A matching variance is weak evidence that two implementations filter an image the same way: the
variance is a scalar and many different filtered images share one. So the generator compares the
filtered images directly, before it compares any scalar:

```
max |cv2.Laplacian(img, CV_64F, ksize=1, BORDER_CONSTANT) - nyxus_laplacian(img)| = 0.0
```

exactly 0 over all 96 cells. Nyxus' hand-rolled `laplacian()` uses the ksize=1 stencil
`[[0,1,0],[1,-4,1],[0,1,0]]` and drops out-of-range taps, which is zero padding; that is what
`ksize=1` plus `BORDER_CONSTANT` means in cv2. With the convolution settled, the only thing the
scalar comparison tests is the variance step, and the residual size says so.

The generator also prints the raw Laplacian's mean, **−0.9583333333333334**. That number is the
reason the variance step is worth an assertion at all: a variance taken over `|x|` rather than `x`
differs from the true variance by exactly `E[|X|]² − E[X]²`, which vanishes only when the mean is 0.
Zero padding at the ROI border keeps it away from 0 here.

## Per element, not just the aggregate — why there is nothing to intercept here

The standing rule is that a test averaging several slices, angles or ROIs must pin the per-element
values too, because two errors that cancel leave a mean unmoved. This family has no such structure,
and that is checked rather than assumed:

- `FOCUS_SCORE` is one variance over one ROI. There is no partition.
- `LOCAL_FOCUS_SCORE` is the mean over the `scale² = 4` tiles, so the per-element rule applies. The
  generator asserts the tile count is 4 and prints each tile's score:

  | tile (x, y) | `var(cv2.Laplacian(tile))` |
  |---|---:|
  | (0, 0) | 30.305555555555561 |
  | (1, 0) | 28.623263888888889 |
  | (0, 1) | 32.831597222222221 |
  | (1, 1) | 21.604166666666668 |

  Their mean is the pinned 28.341145833333336. The per-tile values are not reachable from the C++
  side, which returns only the mean, so the per-element assertion is made analytically instead:
  `test_imq_analytic.h` puts a single spike, whose tile score has the closed form `20v²/P`, in each
  tile in turn and asserts every placement contributes it. A tile that dropped out of the mean, or
  was counted twice, fails there rather than hiding inside an average.
- Both saturations are counts over the whole ROI, again unpartitioned.

What this family has instead of a per-element table is the **filtered image** comparison above,
which is the same idea one level lower: it checks all 96 cells rather than the one number they
reduce to.

## What the two assertions do not cover

**`ksize > 1` as a score.** `focus_score.cpp` carries a second kernel, `{{2,0,2},{0,-8,0},{2,0,2}}`,
selected when `ksize != 1`. `calculate()` never selects it, so no score is computed with it, and the
cell is recorded as INVALID in `matrix/imq.md` rather than left implied.

The kernel itself is vetted. It is `cv2.Laplacian`'s ksize=3 aperture exactly: the generator asserts
cv2's ksize=3 filtered fixture equal to Nyxus' cell for cell, and `test_imq_opencv.h` pins both
stencils from cv2's response to a unit spike. The two kernels are constant arrays and `laplacian()`
picks one per call from its `ksize` argument, so no call changes the kernel a later call sees;
`test_imq_focus_score_kernel_per_call_opencv` runs a ksize=3 call and then a ksize=1 one and asserts
each against cv2's stencil.

**The tile grid is Nyxus' definition, reproduced rather than taken from the tool.** cv2 supplies
each tile's Laplacian and variance. Which sub-arrays are tiles — a `scale × scale` grid of
`(height/scale) × (width/scale)` tiles, with any leftover row or column in no tile — is Nyxus'
own convention, and no tool publishes it. The generator reproduces the grid and asserts its tile
count; `test_imq_analytic.h` pins which tiles take part, and `test_imq_mechanics.h` the two
conventions no closed form gives: the leftover row and column, and the soft-NaN of an ROI too thin
for any tile.

**Negative control: the pin rejects a truncated tiling.** A tile loop bounded by `y < height - M`
stops after the first tile on an even side while still dividing by `scale²`. On this 8×12 fixture
that visits one tile of four and gives **7.5763888888888902** against the full **28.341145833333336**
— 73% apart. The generator asserts that gap: were it to vanish, the `LOCAL_FOCUS_SCORE` pin could not
tell the full tiling from the truncated one.

**The out-of-core path.** `osized_calculate()` scores an oversized ROI with the same
`laplacian_variance()` and `get_local_focus_score()` templates `calculate()` uses, reading the
disk-backed image instead of the in-RAM one. It is not run against cv2; `test_imq_invariant.h` and
`tests/python/test_imq_ooc_invariant.py` hold it bit-equal to the in-RAM path, which is.

## Reproduction

```
conda activate nyxus_mirp                     # cv2 4.13.0, numpy 2.4.6, python 3.11.15
python tests/vetting/oracles/gen_imq_opencv.py
```

The generator parses the fixture out of `test_data.h` and the pins out of `test_imq_opencv.h`,
re-verifies every pin against the fresh cv2 run, and exits non-zero on a mismatch, on a pin it
cannot produce, or on a value it produces that the header pins nothing for. A golden table kept
inside the generator would only ever have compared the script against its own copy.
