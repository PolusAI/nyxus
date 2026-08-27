# IMQ config matrix

Axes = the settings the four image-quality feature classes actually read; verdicts are measured, not
assigned (SPEC §5.1).

**There are none.** `FocusScoreFeature`, `SaturationFeature`, `PowerSpectrumFeature` and
`SharpnessFeature` each take `const Fsettings&` in `calculate()` and none of the four reads a single
`NyxSetting` — grep the four `.cpp` files for `NyxSetting`, `STNGS_` or `theEnvironment` and nothing
comes back. Every knob these features have is a compile-time default on a private static method, so
the cross-product is over defaults rather than over configuration, and a recipe here names a fixture
and a set of defaults instead of a settings bundle. So the cells below are separated by INPUT, not by
settings: three of the five IMQ recipes describe the same 8×12 ROI, and the other two exist because
a constant ROI, a narrow mask and a 24 px short side are the only ways to reach the remaining
branches at all.

Every verdict below is one of SPEC §5.1's three dispositions — VALID, VALID-BUT-PRODUCTION-ONLY,
INVALID — and each carries the test it maps to, because the disposition *is* the claim about which
test exists. A reachable production cell no external tool reproduces is
VALID-BUT-PRODUCTION-ONLY and gets a regression guard; recording a defect is not a substitute for
one, and a cell whose guard is still outstanding says so in its own row rather than being labelled
something outside the vocabulary.

| feature | knob | value | verdict | recipe / oracle / test |
|---|---|---|---|---|
| `FOCUS_SCORE` | `ksize` | 1, the only value `calculate()` passes | VALID | `imq.laplacian_ksize1_zeropad` — opencv, SPEC §7 exact tier |
| `FOCUS_SCORE` | `ksize` | >1, kernel `{{2,0,2},{0,-8,0},{2,0,2}}` | INVALID | unreachable from `calculate()` and no `cv2.Laplacian` counterpart |
| `LOCAL_FOCUS_SCORE` | `scale` | 2, the only value `calculate()` passes | VALID | `imq.laplacian_ksize1_zeropad` — opencv, one tile (see below) |
| `LOCAL_FOCUS_SCORE` | `scale` | ≠2 | INVALID | no config reaches it: `calculate()` hardcodes 2, the parameter has a default and no plumbing, and nothing else calls `get_local_focus_score()` |
| `MIN`/`MAX_SATURATION` | — | in-RAM path | VALID | `imq.saturation_observed_extremum` — cellprofiler, SPEC §7 exact tier |
| `MIN`/`MAX_SATURATION` | ROI | constant (`min == max`) | VALID-BUT-PRODUCTION-ONLY | CellProfiler computes something else here (below), so no oracle claim — `test_imq_{min,max}_saturation_constant_roi_regression`, pinned 0 and 1 |
| `MIN`/`MAX_SATURATION` | mask | narrower than the bounding box | VALID-BUT-PRODUCTION-ONLY | Nyxus counts in-box out-of-mask zeros and CellProfiler does not (below) — `test_imq_{min,max}_saturation_narrow_mask_regression`, pinned 11/16 and 1/16 |
| `POWER_SPECTRUM_SLOPE` | ROI short side | < 24 px | VALID-BUT-PRODUCTION-ONLY | `imq.regression_quality_roi` — the pin is the guard's return value, `test_imq_power_spectrum_slope_regression` |
| `POWER_SPECTRUM_SLOPE` | ROI short side | ≥ 24 px | VALID-BUT-PRODUCTION-ONLY | the algorithm's only reachable cell, and it is defective (below) — `test_imq_power_spectrum_slope_large_roi_regression`, pinned 1.7837481542489078 on a 24×24 ROI |
| `SHARPNESS` | `width` | 2 | VALID-BUT-PRODUCTION-ONLY | `imq.regression_quality_roi` — the reference DOM measure does not reproduce it (below), `test_imq_sharpness_regression` |
| any | out-of-core (`osized_calculate`) | — | VALID-BUT-PRODUCTION-ONLY | reachable and **still unguarded** — the one open row; needs an oversized-ROI harness, see below and `not_covered.md` |

## Three things that look like knobs and are not

- **`static int ksize;`** is declared as "User interface" in all four feature headers and is
  **defined nowhere and referenced nowhere**. Nothing sets it, and a translation unit that tried to
  would fail to link. It is dead API surface, not an axis.
- **`FocusScoreFeature::kernel[9]`** is a *mutable* static. `laplacian()` and `get_focus_score_NT()`
  overwrite it in place when `ksize != 1`, and never restore it — so a single call with `ksize != 1`
  would change the kernel for every later ROI in the process. Nothing reaches that branch today
  because `calculate()` hardcodes `ksize=1`, which is what makes it latent rather than live. Same
  shape as the `NGTDMFeature::n_levels` static the 2D NGTDM pass fixed.
- **`Fsettings`** itself. The three test files pass a default-constructed one because
  `assert`-style helpers need something to pass, not because any value in it is read.

## `LOCAL_FOCUS_SCORE` reaches one tile of four

`get_local_focus_score()` loops `for (y = 0; y < height - M; y += M)` with `M = height/scale`. At
`scale=2` on a 12-row ROI that is `0 < 6` → true, `6 < 6` → false, and the same in x, so exactly
**one** 4×6 tile is visited — while the final division is still by `scale*scale = 4`. The pinned
golden is `var(Laplacian(top-left tile)) / 4`, and `gen_imq_opencv.py` asserts the tile count so a
change to the bound fails there rather than silently redefining the golden.

Two separate claims are entangled in that loop and neither is settled here: the bound is `<` where
`<=` would visit all `scale²` tiles, and `docs/source/Math/f_image_quality.rst` says "the mean and
median values of the tiles are returned" where the code returns one sum over one tile. **Open, not
endorsed.** The assertion pins the current behaviour so a fix is visible as a moved golden.

## `POWER_SPECTRUM_SLOPE` is pinned twice: at the guard, and at the algorithm behind it

`rps()` returns `{0.}` unless `floor(min(h, w) / 8) >= 3`. The fixture is 8 px wide, so
`min(12,8)/8 = 1`, the guard fires, and `power_spectrum_slope()` returns the literal `0` its
`accumulate(magnitude) > 0` test falls through to. `test_imq_power_spectrum_slope_regression` covers
that path and nothing beyond it.

The cell past the guard is reachable production, so it is snapshotted too rather than described and
left unpinned: `test_imq_power_spectrum_slope_large_roi_regression` runs a deterministic 24×24
modular ramp — the smallest ROI that clears `floor(min(h,w)/8) >= 3` — and pins
**1.7837481542489078**. The pin endorses nothing; it exists so that fixing either defect below moves
a golden instead of passing unnoticed.

What the algorithm does in that cell, measured (`PROBE_PS` instrumentation, not committed):

- `power_spectrum_slope()` loops `i` over `magnitude.size()` and reads `raw_radii[i]` inside it, with
  no bound relating the two. On the pinned 24×24 fixture `magnitude.size() = 1024` (the 32×32
  power-of-2 padded FFT) against `raw_radii.size() = 24`, the largest index reached was **3**, and 3
  points survived to the fit — so the read stays in range here and the pin is a defined value. On a
  synthetic 32×32 ROI it was `raw_radii.size() = 32`, 4 surviving bins, largest index 5. Nothing in
  the code keeps the index below `raw_radii.size()`; both inputs happen to stay under it.
- The radius axis is `std::floor(std::sqrt(image_invariant[i])) + 1`, i.e. a function of the FFT
  **coefficient at bin i**, not of the frequency radius `sqrt(kx² + ky²)` the log-log power-spectrum
  fit is defined over. `sqrt` of a negative coefficient is NaN, and a NaN `label_index` fails both
  bounds tests and is dropped.
- Earlier synthetic runs returned 1.3518845575419998 (32×32) and −0.1408723598022707 (24×24) on
  fixtures that were not committed; the pinned figure above is the one this tree reproduces.

So the cell is reachable, produces a number, and that number is not a radial power-spectrum slope.
Vetting it needs the radial binning rewritten and the index bounded; the candidate oracle is
CellProfiler's `centrosome.radial_power_spectrum.rps`, which is the implementation this was ported
from.

## `SHARPNESS` is not the reference DOM measure

Nyxus 2.1904708385718963 against the published reference's 0.54592951157710823 on the same fixture —
a factor of four, and structural rather than numerical. Six differences, all measured by
`audit/imq_sharpness_reference_dom.py`:

1. **Aggregation.** The reference counts pixels whose sharpness reaches `sharpness_threshold=2`
   (28 and 16 here); Nyxus sums the sharpness values themselves (157.83 and 19.68) and has no
   threshold parameter at all.
2. **Sy runs down the wrong axis.** The reference computes `Sy` column-wise, summing `domy` over a
   column window; Nyxus reuses the row-wise pass for both.
3. **The edge maps are swapped.** The reference's `edgex` comes from the *column* convolution and
   `edgey` from the row one; Nyxus assigns them the other way round. Measured: Nyxus `edge_x`=73,
   `edge_y`=56 against the reference's `edgex`=56, `edgey`=73 — the same two numbers, exchanged.
4. **Normalization.** The reference divides each smoothed image by its own maximum; Nyxus divides
   both by the row-convolved one's.
5. **No final masking.** The reference multiplies `Sx`/`Sy` by the edge maps again before
   aggregating; Nyxus masks only the contrast terms.
6. **Column coverage.** Nyxus writes `Sx`/`Sy` only for `k < cols - width`, leaving the last two
   columns at 0; the reference fills every column.

Also: `contrast()` uses the *forward* difference `|Im[i+1] - Im[i]|` where the reference uses the
backward `|Im[i] - Im[i-1]|`, which shifts the contrast field one row/column against the DOM field;
and `median_blur()` pads by `(rows, cols)` rather than by `(ksize-1)/2`, builds a 3× image, and its
`remove_padding()` ends with an `erase()` that is a no-op, so the blurred vector keeps a 768-element
tail nothing reads. Neither changes the six above.

`SHARPNESS` therefore stays `regression` with no oracle claim. The registry's
`candidate_oracle = "reference DOM sharpness (Kumar et al. 2012)"` is now measured and refuted
rather than untried; promotion needs the six differences resolved first. Report:
`audit/imq_pydom_sharpness_vetting_report.md`.

## The out-of-core paths are the one row still without a guard

`phase3.cpp` calls `osized_scan_whole_image()` on every registered feature method for an oversized
ROI, and `FocusScoreFeature` and `SaturationFeature` are both registered, so this cell is reachable
production — VALID-BUT-PRODUCTION-ONLY, not "not covered". It is also the one cell in this matrix
whose regression guard is **outstanding**: reaching `osized_calculate()` needs an oversized-ROI
harness (a disk-backed `raw_pixels_NT` and `WriteImageMatrix_nontriv`), which the gtest fixture here
does not build, and the 2D out-of-core path is being repaired on its own branch. Stated here as an
open row rather than closed by relabelling; `not_covered.md` carries the same entry.

What follows is therefore read off the source rather than measured. The two feature classes not
listed below, `PowerSpectrumFeature` and `SharpnessFeature`, carry their own out-of-core defect,
tracked outside this vetting pass.

- `SaturationFeature::get_percent_max_pixels_NT()` uses two independent `if`s where the in-RAM
  `get_percent_max_pixels()` uses `else if`. On a constant ROI those two disagree by construction —
  and `osized_calculate()` returns before either runs when `aux_max == aux_min`, leaving both
  saturations at 0. Three behaviours for one input.
- `FocusScoreFeature::get_focus_score_NT()` calls `laplacian(W, conv_buffer, width, height, ksize)`,
  passing width where the signature's `m_image` (rows) is; takes the variance over the whole
  `conv_buffer`, which is sized `(winY+2)*(winX+2)*2 = 2048` and larger than the region any pixel
  writes; and fills `W`, sized `winY*winX = 900`, with `W[row*width + col]` over the full ROI in the
  branch taken when the ROI is smaller than one 30×30 window — which for a 100×20 ROI writes 2000
  entries into 900. Its `tile_variance` vector, whose comment still says "abs sum of tile", is
  declared and never used.

## Measured agreement at the two VALID points

| feature | oracle | oracle value | Nyxus | abs | rel |
|---|---|---|---:|---:|---:|
| `FOCUS_SCORE` | opencv 4.13.0 | 34.956597222222221 | 34.956597222222229 | 7.1e-15 | 2.0e-16 |
| `LOCAL_FOCUS_SCORE` | opencv 4.13.0 | 7.5763888888888902 | 7.5763888888888937 | 3.6e-15 | 4.7e-16 |
| `MIN_SATURATION` | cellprofiler 4.2.8 | 0.1875 | 0.1875 | 0 | 0 |
| `MAX_SATURATION` | cellprofiler 4.2.8 | 0.16666666666666669 | 0.16666666666666666 | 2.8e-17 | 1.7e-16 |

Both files assert at SPEC §7's exact tier, an **absolute** 1e-9 band. On the focus scores the tier
applies because the two sides filter the image identically — `gen_imq_opencv.py` asserts the
filtered images are equal cell for cell — and differ only in the order the variance is summed. On the
saturations both tools count the same 18 and 16 of 96 pixels; the one-ulp gap on `MAX_SATURATION` is
CellProfiler reporting a percentage that the generator divides by 100.

Before this pass all four asserted at `rel=1e-3`, which was not a measurement: no call site passed a
tolerance and `assert_feature`'s signature ends `double frac_tolerance = 1000`. The registry read
`rel=1e-3` because that default did.

## No GPU axis, and no coverage sweep

None of the four features has a GPU path, an IBSI mode, or a 3D twin — `FeatureIMQ` is its own
enum and `dim=IMQ` is its own registry dimension. IMQ is also the one family with no
`*_coverage.h` sweep to retire: every feature has a named test, and the features whose matrix has
more than one reachable cell have one test per cell — eleven assertions over six features.
