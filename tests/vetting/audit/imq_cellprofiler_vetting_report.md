# Audit: IMQ saturation vs a fresh CellProfiler run

**Verdict: both quantities reproduce.** `MIN_SATURATION` bit for bit; `MAX_SATURATION` to one ulp,
and the ulp is a unit conversion rather than a disagreement about the statistic.

Covers `tests/test_imq_cellprofiler.h` (goldens + assertions), `tests/test_imq_common.h` (fixture)
and `tests/vetting/oracles/gen_imq_cellprofiler.py` (generator).

## Method

- **Tool**: CellProfiler **4.2.8** (module package) on cellprofiler-core 4.2.8.1, centrosome 1.2.3,
  numpy 1.26.4, Python 3.9, in the `nyxus_cellprofiler` conda env (`TOOLS.md`).
  `cellprofiler.modules.measureimagequality.MeasureImageQuality.calculate_saturation()`, run
  in-process and headless on a single grayscale image with no mask.
- **Config**: recipe `imq.saturation_observed_extremum`. Neither side has a setting to match —
  `SaturationFeature` reads no `NyxSetting`, and the CP module's only relevant switch is
  `check_saturation`.
- **Fixture**: `im_quality_intensity` / `im_quality_mask` exactly as `tests/test_data.h` stores
  them, parsed out of that header by the generator. One 8×12 ROI, 96 pixels, min 0, max 6, **18** at
  the min and **16** at the max.
- **Command**: `python tests/vetting/oracles/gen_imq_cellprofiler.py`; full steps in
  `imq_golden_regen.md`.

## Result table

| feature | pinned (CellProfiler) | Nyxus | abs | rel | verdict |
|---|---|---|---:|---:|---|
| `MIN_SATURATION` | 0.1875 | 0.1875 | 0 | 0 | vetted |
| `MAX_SATURATION` | 0.16666666666666669 | 0.16666666666666666 | 2.8e-17 | 1.7e-16 | vetted |

Both assert at SPEC §7's exact tier, an **absolute** 1e-9 band via `ASSERT_NEAR`.

## Why `MAX_SATURATION` is pinned at …669 and not …666

The two tools count the same pixels. CellProfiler's `calculate_saturation()` is

```python
number_pixels_maximal = numpy.sum(pixel_data == numpy.max(pixel_data))     # 16
percent_maximal       = 100.0 * number_pixels_maximal / pixel_count        # 96 pixels
```

and Nyxus' `get_percent_max_pixels()` returns `max_pixel_count / image.size()` directly. Both agree
that 16 of the 96 pixels sit at the maximum. But CP reports a **percentage**, so the fraction reaches
the generator as `100.0*16/96` divided by `100.0`, which is one ulp above `16/96`:

```
100.0*16/96/100.0 -> 0.16666666666666669   565555555555c53f
16/96             -> 0.16666666666666666   555555555555c53f
```

`MIN_SATURATION` has no such gap because 18/96 = 0.1875 is exactly representable in binary and the
round trip through 18.75% lands back on it.

The pin carries CellProfiler's digits, which is what the pin of an oracle table should carry. The
previous pin carried Nyxus' `0.16666666666666666` under a comment reading "CellProfiler … = 16/96;
tolerance rel=1e-3 (agreement is exact)". Three things were wrong with that at once: the value was
not CellProfiler's, the agreement was not exact, and a `rel=1e-3` band is five orders looser than
what "exact" would justify. The old generator could not have caught it — it compared CP against a
literal `NYXUS = {...}` dict in its own source at `TOL=1e-6`, so a 2.8e-17 gap was three orders
below anything it looked at.

## Range and identity checks

Run mechanically by the generator over both pins rather than eyeballed:

- each value in [0, 1];
- each value equals its own integer count over the pixel total — `MIN` = 18/96, `MAX` = 16/96 — so a
  pin that had drifted off a whole-pixel fraction would fail even if it stayed inside the band;
- `MIN + MAX ≤ 1`, which holds because the two pixel sets are disjoint whenever `min != max`;
- **scale invariance**: both metrics are unchanged when the image is divided by its own maximum.
  This is the property that lets CellProfiler, whose images are `[0,1]` floats, vet Nyxus, whose
  pixels are integer `PixIntens` — so it is asserted rather than assumed.

## What the two assertions do not cover

Both cells below are reachable production configs, so under SPEC §5.1 they are
VALID-BUT-PRODUCTION-ONLY rather than gaps: CellProfiler computes a different quantity on each, so
neither can carry an oracle claim, and each is pinned as a drift guard in `test_imq_regression.h`
instead. What is uncovered is the *agreement*, not the code path.

**A constant ROI (`min == max`).** CellProfiler counts minimal and maximal independently and reports
100% for both. Nyxus' `get_percent_max_pixels()` uses `else if`, so a pixel equal to both extrema is
counted only as maximal. On a constant 4×4 ROI Nyxus returns `MIN_SATURATION = 0`,
`MAX_SATURATION = 1` — now pinned by `test_imq_{min,max}_saturation_constant_roi_regression`, recipe
`imq.saturation_production_only`.

That case has a third behaviour as well. `SaturationFeature::osized_calculate()` returns early when
`r.aux_max == r.aux_min`, leaving both values at 0 — and the out-of-core
`get_percent_max_pixels_NT()` it would otherwise call uses two independent `if`s, i.e. CellProfiler's
convention rather than the in-RAM path's. One input, three answers, depending on the ROI size and
which code path runs. Recorded in `matrix/imq.md` and `not_covered.md`.

**A mask narrower than the bounding box.** Nyxus computes over the ROI's bounding-box image matrix,
in which in-box out-of-mask pixels are 0 and *do* take part in the extremum; CellProfiler restricts
to `image.mask` when one is present. The two coincide here only because `im_quality_mask` covers the
whole 8×12 box. `test_imq_{min,max}_saturation_narrow_mask_regression` supplies a 4×4 AABB whose mask
holds 5 of its 16 pixels and pins Nyxus at `MIN_SATURATION = 11/16`, `MAX_SATURATION = 1/16` — the 11
in-box out-of-mask zeros counted into both. Vetting the cell, as opposed to guarding it, would need
CellProfiler and Nyxus to agree on which pixels the ROI contains, which they do not.

**CellProfiler's `FocusScore` / `LocalFocusScore` are not an oracle for Nyxus' features of those
names.** CP's `FocusScore` is the normalized variance of the *raw* image, `sum((x-mean)²)/(N·mean)`,
and its `LocalFocusScore` is `var(local_norm_var)/median(local_norm_var)` over a grid. Nyxus
implements the Pech-Pacheco variance-of-*Laplacian* instead, vetted against OpenCV in
`imq_opencv_vetting_report.md`. The generator fails if either name appears in the CellProfiler
table, so the two cannot be wired up by a later edit.

## Reproduction

```
conda activate nyxus_cellprofiler
set JAVA_HOME=%CONDA_PREFIX%\Library\lib\jvm         # and JDK_HOME; see TOOLS.md
cd %TEMP%                                            # same drive as the CellProfiler install
python <repo>/tests/vetting/oracles/gen_imq_cellprofiler.py
```

The generator parses the fixture out of `test_data.h` and the pins out of
`test_imq_cellprofiler.h`, re-verifies every pin against the fresh CP run, and exits non-zero on a
mismatch, on a pin it cannot produce, or on a value it produces that the header pins nothing for.
