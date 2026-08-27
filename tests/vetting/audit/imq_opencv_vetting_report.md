# Audit: IMQ focus scores vs a fresh OpenCV run

**Verdict: both quantities reproduce**, to 7.1e-15 and 3.6e-15 absolute — and the *filtered image*
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
| `LOCAL_FOCUS_SCORE` | 7.5763888888888902 | 7.5763888888888937 | 3.6e-15 | 4.7e-16 | vetted |

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
- `LOCAL_FOCUS_SCORE` is a sum over the tiles `get_local_focus_score()` visits, divided by
  `scale²`. The generator **asserts that the tile count is 1**, so the "aggregate" and its single
  element are the same number up to the constant divisor, and a per-element table would be the
  scalar table again. If the loop bound is ever fixed the count becomes 4, that assertion fails,
  and the golden has to be re-derived — at which point the per-element rule starts applying and the
  generator is the thing that says so.
- Both saturations are counts over the whole ROI, again unpartitioned.

What this family has instead of a per-element table is the **filtered image** comparison above,
which is the same idea one level lower: it checks all 96 cells rather than the one number they
reduce to.

## What the two assertions do not cover

**`ksize > 1`.** `focus_score.cpp` carries a second kernel, `{{2,0,2},{0,-8,0},{2,0,2}}`, selected
when `ksize != 1`. It has no `cv2.Laplacian` counterpart, and `calculate()` never selects it —
`laplacian()` is only ever called with the default. Out of scope, and recorded as INVALID in
`matrix/imq.md` rather than left implied.

Worth noting where that kernel lives: `FocusScoreFeature::kernel[9]` is a *mutable static*, and the
`ksize != 1` branch overwrites it in place without restoring it. A single call with `ksize != 1`
would change the kernel for every later ROI in the process. Nothing reaches that branch today, which
makes it latent rather than live — the same shape as the `NGTDMFeature::n_levels` static the 2D
NGTDM pass fixed.

**`LOCAL_FOCUS_SCORE` visits one tile, not four.** `get_local_focus_score()` loops
`for (y = 0; y < height - M; y += M)` with `M = height/scale`. On a 12-row ROI at `scale=2` that is
`0 < 6` → true, `6 < 6` → false, and the same in x: exactly **one** 4×6 tile, while the final
division is still by `scale*scale = 4`. The golden reproduces that, and the generator asserts the
tile count is 1 — so a change to the loop bound fails in the generator rather than silently
redefining what the golden means.

Two claims are entangled there and this report settles neither: the bound is `<` where `<=` would
visit all `scale²` tiles, and `docs/source/Math/f_image_quality.rst` says "the mean and median
values of the tiles are returned" where the code returns one sum over one tile. Both are recorded as
open in `matrix/imq.md`.

**How much of the feature that leaves unvetted, measured.** The tile count alone only checks that
the tiling this generator reproduces still matches the one Nyxus walks; it says nothing about the
size of the gap. So the generator carries the SPEC §4 negative control for a partial-pipeline
oracle: scoring the same feature over all `scale² = 4` tiles — the tiling the `/scale²` divisor
already assumes — gives **28.341145833333336** against the pinned one-tile **7.5763888888888902**.
**73% of `LOCAL_FOCUS_SCORE` is outside this oracle's reach**, and the control asserts that gap
rather than printing it: were it to vanish, the one-tile pin would not be a partial value and this
scope note would be overstated.

**The out-of-core path.** `FocusScoreFeature::get_focus_score_NT()` is reached by
`osized_calculate()` and by no assertion in the tree. Reading it (not measuring — nothing exercises
it): it passes `(width, height)` to `laplacian()` where the signature's first size parameter is the
row count; it takes the variance over the whole `conv_buffer`, sized `(winY+2)*(winX+2)*2 = 2048`
and larger than the region any pixel writes; and in the branch taken when the ROI is smaller than
one 30×30 window it fills `W`, sized `winY*winX = 900`, with `W[row*width + col]` over the full ROI
— for a 100×20 ROI that is 2000 entries into 900. It also declares a `tile_variance` vector, whose
comment still reads "0: abs sum of tile", that nothing ever touches. Recorded in `matrix/imq.md` and
`not_covered.md`; out of scope for a vetting pass.

## Reproduction

```
conda activate nyxus_mirp                     # cv2 4.13.0, numpy 2.4.6, python 3.11.15
python tests/vetting/oracles/gen_imq_opencv.py
```

The generator parses the fixture out of `test_data.h` and the pins out of `test_imq_opencv.h`,
re-verifies every pin against the fresh cv2 run, and exits non-zero on a mismatch, on a pin it
cannot produce, or on a value it produces that the header pins nothing for. A golden table kept
inside the generator would only ever have compared the script against its own copy.
