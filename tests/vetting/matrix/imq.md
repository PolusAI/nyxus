# IMQ config matrix

Axes = the settings the four image-quality feature classes actually read; verdicts are measured, not
assigned (SPEC §5.1).

**There are none.** `FocusScoreFeature`, `SaturationFeature`, `PowerSpectrumFeature` and
`SharpnessFeature` each take `const Fsettings&` in `calculate()` and none of the four reads a single
`NyxSetting` — grep the four `.cpp` files for `NyxSetting`, `STNGS_` or `theEnvironment` and nothing
comes back. Every knob these features have is a compile-time default on a private static method, so
the cross-product is over defaults rather than over configuration, and a recipe here names a fixture
and a set of defaults instead of a settings bundle. That is why all three IMQ recipes describe the
same 8×12 ROI.

| feature | knob | value | verdict | recipe / oracle |
|---|---|---|---|---|
| `FOCUS_SCORE` | `ksize` | 1, the only value `calculate()` passes | VALID | `imq.laplacian_ksize1_zeropad` — opencv, SPEC §7 exact tier |
| `FOCUS_SCORE` | `ksize` | >1, kernel `{{2,0,2},{0,-8,0},{2,0,2}}` | INVALID | unreachable from `calculate()` and no `cv2.Laplacian` counterpart |
| `LOCAL_FOCUS_SCORE` | `scale` | 2, the only value `calculate()` passes | VALID | `imq.laplacian_ksize1_zeropad` — opencv, one tile (see below) |
| `LOCAL_FOCUS_SCORE` | `scale` | ≠2 | NOT REACHABLE | no caller sets it; the parameter has a default and no plumbing |
| `MIN`/`MAX_SATURATION` | — | in-RAM path | VALID | `imq.saturation_observed_extremum` — cellprofiler, SPEC §7 exact tier |
| `MIN`/`MAX_SATURATION` | ROI | constant (`min == max`) | VALID-but-divergent | not asserted; Nyxus and CellProfiler disagree (below) |
| `MIN`/`MAX_SATURATION` | mask | narrower than the bounding box | VALID-but-divergent | not asserted; Nyxus counts in-box out-of-mask zeros, CellProfiler does not |
| `POWER_SPECTRUM_SLOPE` | ROI short side | < 24 px | VALID-prod-only | `imq.regression_quality_roi` — the pin is the guard's return value |
| `POWER_SPECTRUM_SLOPE` | ROI short side | ≥ 24 px | NOT PINNED | the algorithm's only reachable cell, and it is defective (below) |
| `SHARPNESS` | `width` | 2 | VALID-prod-only | `imq.regression_quality_roi` — the reference DOM measure does not reproduce it (below) |
| any | out-of-core (`osized_calculate`) | — | NOT COVERED | no assertion reaches either NT path (below) |

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

## `POWER_SPECTRUM_SLOPE` is pinned at the guard, not at the algorithm

`rps()` returns `{0.}` unless `floor(min(h, w) / 8) >= 3`. The fixture is 8 px wide, so
`min(12,8)/8 = 1`, the guard fires, and `power_spectrum_slope()` returns the literal `0` its
`accumulate(magnitude) > 0` test falls through to. The pin covers that path and nothing beyond it.

Measured on a synthetic 32×32 ROI, where the guard does not fire (`PROBE_PS` instrumentation, not
committed):

- `magnitude.size() = 1024`, `raw_radii.size() = 32`. `power_spectrum_slope()` loops `i` over
  `magnitude.size()` and reads `raw_radii[i]` inside it, with no bound relating the two. On this
  input only 4 bins passed the `magnitude[i] > 0` test and the largest index reached was 5, so the
  read stayed in range — but nothing in the code keeps it there.
- The radius axis is `std::floor(std::sqrt(image_invariant[i])) + 1`, i.e. a function of the FFT
  **coefficient at bin i**, not of the frequency radius `sqrt(kx² + ky²)` the log-log power-spectrum
  fit is defined over. `sqrt` of a negative coefficient is NaN, and a NaN `label_index` fails both
  bounds tests and is dropped.
- With 4 surviving points the least-squares fit ran and returned 1.3518845575419998 (32×32) and
  −0.1408723598022707 (24×24).

So the cell "short side ≥ 24 px" is reachable, produces a number, and that number is not a radial
power-spectrum slope. Vetting it needs the radial binning rewritten and a second benchmark at least
24 px on its short side; the candidate oracle is CellProfiler's
`centrosome.radial_power_spectrum.rps`, which is the implementation this was ported from.

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

## The out-of-core paths are uncovered, and one of them is not a port of the in-RAM path

No assertion in the tree reaches `osized_calculate()` for any IMQ feature, so what follows is read
off the source rather than measured, and is recorded here so it is tracked rather than implied by
the word "uncovered".

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
`*_coverage.h` sweep to retire: all six features have named tests, one each.
