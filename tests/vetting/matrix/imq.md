# IMQ config matrix

Axes = the settings the four image-quality feature classes actually read; verdicts are measured, not
assigned (SPEC §5.1).

**There is one, and it is not a knob.** `FocusScoreFeature` reads the soft-NaN setting
(`STNGS_NAN`), the value it publishes for a `LOCAL_FOCUS_SCORE` no tile fits; `SaturationFeature`,
`PowerSpectrumFeature` and `SharpnessFeature` read no `NyxSetting` at all — grep the four `.cpp`
files for `NyxSetting`, `STNGS_` or `theEnvironment`. Every knob these features have is a
compile-time default on a private static method, so the cross-product is over defaults rather than
over configuration, and a recipe here names a fixture
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

**Scope: this matrix vets the in-RAM paths.** Every VALID row below is a cell `calculate()` reaches.
The out-of-core cell — `osized_calculate()`, entered when a ROI exceeds the RAM limit — is split by
feature. The two focus scores share their scoring code with `calculate()` and are held bit-equal to
it by an invariant test. The other three features' out-of-core cells are classified and described
here but still unguarded, and the follow-up that closes them is scoped at the end of this file.

| feature | knob | value | verdict | recipe / oracle / test |
|---|---|---|---|---|
| `FOCUS_SCORE` | `ksize` | 1, the only value `calculate()` passes | VALID | `imq.laplacian_ksize1_zeropad` — opencv, SPEC §7 exact tier |
| `FOCUS_SCORE` | `ksize` | >1, kernel `{{2,0,2},{0,-8,0},{2,0,2}}` | INVALID | unreachable from `calculate()`; the stencil itself is `cv2.Laplacian`'s ksize=3 aperture, pinned by `test_imq_focus_score_kernel_per_call_opencv` |
| `LOCAL_FOCUS_SCORE` | `scale` | 2, the only value `calculate()` passes | VALID | `imq.laplacian_ksize1_zeropad` — opencv, the mean over the 2×2 tile grid (see below) |
| `LOCAL_FOCUS_SCORE` | ROI | each tile carrying the texture in turn, both side parities | VALID | analytic, `test_imq_local_focus_score_{each_tile,all_tiles}_analytic` — `20v²/(P·scale²)` per tile (see below) |
| `LOCAL_FOCUS_SCORE` | ROI side | not a multiple of `scale` | VALID-BUT-PRODUCTION-ONLY | mechanics, `test_imq_local_focus_score_remainder_mechanics` — the leftover row/column is in no tile, pinned 0; a Nyxus convention, not a closed form |
| `LOCAL_FOCUS_SCORE` | ROI side | shorter than `scale` (a 1 px thin ROI) | VALID-BUT-PRODUCTION-ONLY | mechanics, `test_imq_local_focus_score_thin_roi_mechanics` — no tile fits, the score is undefined and reads as the soft-NaN setting; `test_imq_local_focus_score_smallest_tiled_roi_mechanics` pins the 2×2 ROI on the other side of that guard at a defined 0 |
| `LOCAL_FOCUS_SCORE` | `scale` | ≠2 | INVALID | no config reaches it: `calculate()` hardcodes 2, the parameter has a default and no plumbing, and nothing else calls `get_local_focus_score()` |
| `MIN`/`MAX_SATURATION` | — | in-RAM path | VALID | `imq.saturation_observed_extremum` — cellprofiler, SPEC §7 exact tier |
| `MIN`/`MAX_SATURATION` | ROI | constant (`min == max`) | VALID-BUT-PRODUCTION-ONLY | CellProfiler computes something else here (below), so no oracle claim — `test_imq_{min,max}_saturation_constant_roi_regression`, pinned 0 and 1 |
| `MIN`/`MAX_SATURATION` | mask | narrower than the bounding box | VALID-BUT-PRODUCTION-ONLY | Nyxus counts in-box out-of-mask zeros and CellProfiler does not (below) — `test_imq_{min,max}_saturation_narrow_mask_regression`, pinned 11/16 and 1/16 |
| `POWER_SPECTRUM_SLOPE` | ROI short side | < 24 px | VALID-BUT-PRODUCTION-ONLY | `imq.regression_quality_roi` — the pin is the guard's return value, `test_imq_power_spectrum_slope_regression` |
| `POWER_SPECTRUM_SLOPE` | ROI short side | ≥ 24 px | VALID-BUT-PRODUCTION-ONLY | the algorithm's only reachable cell, and it is defective (below) — `test_imq_power_spectrum_slope_large_roi_regression`, pinned 1.7837481542489078 on a 24×24 ROI |
| `SHARPNESS` | `width` | 2 | VALID-BUT-PRODUCTION-ONLY | `imq.regression_quality_roi` — the reference DOM measure does not reproduce it (below), `test_imq_sharpness_regression` |
| `FOCUS_SCORE`, `LOCAL_FOCUS_SCORE` | out-of-core (`osized_calculate`) | — | VALID-BUT-PRODUCTION-ONLY | invariant, `test_imq_focus_score_out_of_core_invariant` in `test_imq_invariant.h` and `tests/python/test_imq_ooc_invariant.py` — bit-equal to the in-RAM scores |
| `MIN`/`MAX_SATURATION`, `POWER_SPECTRUM_SLOPE`, `SHARPNESS` | out-of-core (`osized_calculate`) | — | VALID-BUT-PRODUCTION-ONLY | reachable and **still unguarded** — the open row; see below and `not_covered.md` |

## Two things that look like knobs and are not

- **The Laplacian kernel.** `laplacian()` picks one of two constant stencils per call from its
  `ksize` argument — `{{0,1,0},{1,-4,1},{0,1,0}}` for 1, `{{2,0,2},{0,-8,0},{2,0,2}}` otherwise — so
  no call changes the kernel a later one sees. `calculate()` always passes 1. The two stencils are
  `cv2.Laplacian`'s ksize=1 and ksize=3 apertures exactly, and
  `test_imq_focus_score_kernel_per_call_opencv` runs a `ksize=3` call and then a `ksize=1` one on a
  single spike and asserts each against cv2's own unit-spike response, re-verified by
  `gen_imq_opencv.py`.
- **`Fsettings`** itself. The test files pass one laid out like the production settings, with the
  soft-NaN set to a value no focus score can take, so the thin-ROI assertion can tell it from a
  computed score. Only that one slot is read.

## `LOCAL_FOCUS_SCORE` is the mean over a `scale × scale` tile grid

`get_local_focus_score()` cuts the ROI into `scale²` non-overlapping tiles of `M = height/scale` by
`N = width/scale` pixels, tile `(tx, ty)` starting at `(tx·N, ty·M)`, and returns the mean of their
focus scores. The last `height % scale` rows and `width % scale` columns belong to no tile. A side
shorter than `scale` gives `M` or `N` = 0: no tile fits, the score is undefined, and it reads as the
soft-NaN setting (`--noval`, 0 by default) - the same no-value substitute the texture families
publish - rather than as a computed score. At the `scale=2`
`calculate()` passes, the 8×12 fixture has four 4×6 tiles, and `gen_imq_opencv.py` asserts that
count and prints each tile's score.

The opencv assertion vets the mean on the fixture. Which tiles take part is asserted separately,
analytically, because every tile of the fixture is textured and a mean over them cannot say which
ones contributed. A single spike of height `v` clear of the tile edge has a ksize=1 Laplacian of
`-4v` at the spike and `+v` at its four neighbours, so its tile scores exactly `20v²/P` with `P`
the tile's pixel count, and every all-zero tile scores 0. `test_imq_analytic.h` puts the spike in
each tile in turn on 8×12, 9×12, 8×13 and 9×13 ROIs — all a 2×2 grid of 4×6 tiles — and asserts
`20·36/24/4 = 7.5` for every placement. Both parities matter: a `y < height - M` loop bound stops
after the first tile on an even side but still reaches the second on an odd one, so only the
even-sided ROIs separate the two bounds.

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

## The out-of-core paths

`phase3.cpp:112` calls `osized_scan_whole_image()` on every registered feature method for an
oversized ROI, and all four IMQ feature methods are registered in `feature_mgr_init.cpp`, so every
IMQ out-of-core cell is reachable production — VALID-BUT-PRODUCTION-ONLY, not "not covered". The
feature methods are long-lived: one instance per class serves every oversized ROI of a run, and
nothing resets it between ROIs.

### The focus scores are held equal to the in-RAM path

`FocusScoreFeature::osized_calculate()` builds the same bounding-box image `calculate()` reads, from
the ROI's disk-backed pixels, and passes it to the same two scoring templates,
`laplacian_variance()` and `get_local_focus_score()`. They read the image only through a
`px(row, col)` accessor and filter it three rows at a time, so an oversized ROI is never held whole,
and the two paths share every arithmetic step. Both members are assigned on every call, constant
ROIs included, so nothing carries over from one ROI to the next.

Two invariant tests assert the out-of-core scores **equal** to the in-RAM ones, not close to them:

- `test_imq_focus_score_out_of_core_invariant` in `test_imq_invariant.h` calls `osized_calculate()`
  directly, through one feature instance, on pseudo-random ROIs larger than 30 px on both sides and
  on one side only, a constant ROI, a 1 px thin one and the im_quality fixture.
- `tests/python/test_imq_ooc_invariant.py` drives `ImageQuality(..., ram_limit=0)` through
  `featurize_directory`, so the real oversized-ROI loop runs, on one slide carrying a textured, a
  constant and a thin ROI in that order.

Equality with the in-RAM path is the whole claim: it establishes no vetting of its own.

### The other three are still without a guard

Read off the source, not measured:

- **`PowerSpectrumFeature::osized_calculate()` is empty** — `{}` at `power_spectrum.h:28`, overriding
  the base's pure virtual. `FeatureMethod::osized_scan_whole_image()` (`feature_method.cpp:49`) calls
  it and then `save_value()` unconditionally, so `POWER_SPECTRUM_SLOPE` is published from `slope_`
  without anything having computed it.
- **`SharpnessFeature::osized_calculate()` is empty** — `{}` at `sharpness.h:32`, the same shape,
  publishing `sharpness_`.
- **Their members have no default initializer.** `slope_`, `sharpness_`, `max_saturation_` and
  `min_saturation_` are bare `double x;`, no constructor assigns them, and `cleanup_instance()` is
  `virtual void cleanup_instance() {}` (`feature_method.h:43`) with no override in any of the three
  classes. Combined with the two items above, the first oversized ROI publishes an
  **indeterminate** double rather than a zero.
- **The saturation early return leaks the previous ROI's values.** `SaturationFeature::osized_calculate()`
  (`saturation.cpp:58`) returns early when `aux_max == aux_min`, but the base calls `save_value()`
  regardless, so the second oversized constant ROI publishes the first one's numbers. Same shape as
  the `NGTDMFeature::n_levels` static the 2D NGTDM pass fixed.
- **`SaturationFeature::get_percent_max_pixels_NT()` uses two independent `if`s** (`saturation.cpp`
  lines 125-126) where the in-RAM `get_percent_max_pixels()` uses `else if` (lines 87-89), so on a
  constant ROI the two paths disagree by construction — and on that ROI the early return above means
  neither of them runs. One input, three answers.
- **`PowerSpectrumFeature::featureset` names the wrong feature** (`power_spectrum.h:17`):
  `{ FeatureIMQ::FOCUS_SCORE }` where the constructor provides `POWER_SPECTRUM_SLOPE`. Latent today —
  nothing reads it, and `required()` tests the enum directly — but `SaturationFeature::required()` is
  written as `anyEnabled(featureset)`, so aligning this class with that pattern would gate
  `POWER_SPECTRUM_SLOPE` on whether `FOCUS_SCORE` was requested.

### What the follow-up carries

One PR for the three features above:

1. one matrix row per feature in place of the shared open row, each with its own SPEC §5.1
   disposition and its own assertion — the two focus-score invariant tests are the pattern, and
   their harnesses (a disk-backed `raw_pixels_NT` in gtest, `ImageQuality(..., ram_limit=0)` in
   Python) already exist;
2. a fix for every defect listed above.

It changes what these features publish on the out-of-core path, so it is a source change and lands
on its own branch under the standing rule.

Not in it: the `POWER_SPECTRUM_SLOPE` radial-binning defect described earlier. That one is an
**in-RAM** defect the out-of-core path merely inherits, it is already pinned by
`test_imq_power_spectrum_slope_large_roi_regression`, and closing it needs an oracle
(`centrosome.radial_power_spectrum.rps`) rather than a harness — so the two stay separable.

## Measured agreement at the two VALID points

| feature | oracle | oracle value | Nyxus | abs | rel |
|---|---|---|---:|---:|---:|
| `FOCUS_SCORE` | opencv 4.13.0 | 34.956597222222221 | 34.956597222222229 | 7.1e-15 | 2.0e-16 |
| `LOCAL_FOCUS_SCORE` | opencv 4.13.0 | 28.341145833333336 | 28.341145833333343 | 7.1e-15 | 2.5e-16 |
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
more than one reachable cell have one test per cell.
