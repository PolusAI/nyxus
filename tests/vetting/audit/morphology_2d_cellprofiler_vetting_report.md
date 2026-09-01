# 2D morphology vs CellProfiler — vetting report

Closes the six `oracle=cellprofiler` rows that carried no evidence: `not_covered.md` §C listed them
as the one entry where not even the *provenance* of the values was established — no version, no
config, no generator, so nothing in the tree distinguished a CellProfiler number from a Nyxus one.
Running the tool proved five of them and disproved the sixth.

## Tool and configuration

| | |
|---|---|
| Tool | CellProfiler 4.2.8 (module package), cellprofiler-core 4.2.8.1, centrosome 1.2.3, python 3.9, env `nyxus_cellprofiler` (conda) |
| Module | `MeasureObjectIntensity`, one image and one object set, all settings at their defaults |
| Generator | `tests/vetting/oracles/gen_morphology_cellprofiler.py` |
| Recipe | `morphology.cellprofiler_edge_intensity` |
| Fixture | `shape2d_morphology_{mask,intensity}`, read out of `tests/test_data.h`, background-padded |
| Benchmark | `bench_shape8_concave_holed` |
| Test | `test_2d_morphology_cellprofiler.h` |
| Tolerance | `rel=1e-6` |

```
python tests/vetting/oracles/gen_morphology_cellprofiler.py
```

Verifies every golden pinned in the test header and exits non-zero on mismatch.
Current run: **5 verified, 0 failed, 0 unproducible, 0 unpinned**, identity holds.

## Results

| feature | Nyxus | CellProfiler | rel | source |
|---|---|---|---|---|
| MASS_DISPLACEMENT | 0.634476074243407 | 0.6344760644215962 | 9.8e-09 | `Intensity_MassDisplacement` |
| EDGE_MEAN_INTENSITY | 41.8333333333333 | 41.833334217468895 | 2.1e-08 | `Intensity_MeanIntensityEdge` |
| EDGE_MAX_INTENSITY | 68.0 | 68.00000354647636 | 5.2e-08 | `Intensity_MaxIntensityEdge` |
| EDGE_MIN_INTENSITY | 12.0 | 12.000000234693289 | 2.0e-08 | `Intensity_MinIntensityEdge` |
| EDGE_INTEGRATED_INTENSITY | 753.0 | 753.0000159144402 | 2.1e-08 | `Intensity_IntegratedIntensityEdge` |
| EDGE_STDDEV_INTENSITY | 16.7691944455582 | 16.296727949274324 | **2.9e-02** | `Intensity_StdIntensityEdge` — see below |

**The agreement is exact rather than approximate, because both tools select the same pixels.**
CellProfiler's edge is `skimage.segmentation.find_boundaries(mode="inner")` (connectivity 1): an
object pixel is an edge pixel unless all four of its N/S/E/W neighbours share its label. On this ROI
that is 18 of the 26 pixels, summing to 753 against the ROI's 1048 — the same set Nyxus walks, which
is why the integrated value agrees to the digit rather than merely closely.

The residual is not a disagreement. CellProfiler measures on `[0,1]` and stores the image as
float32, so a raw value round-trips as `raw/255 → float32 → ×255` and returns within ~1 ulp of
float32. The band is `rel=1e-6`: above the measured 5.2e-8, far below any real divergence.
Mutation-checked — an 8.8e-6 perturbation of the mean fails.

## EDGE_STDDEV_INTENSITY: a definitional gap, demoted to regression

Over the **identical 18 pixels** the two tools use different estimators. Nyxus divides the variance
by n-1 (`Moments4::std()` returns `sqrt(M2/(n-1))`, a helper shared across features, so this is a
house convention rather than a local slip); CellProfiler divides by n. The values therefore differ
by exactly `sqrt(n/(n-1))`:

| quantity | value |
|---|---|
| Nyxus | 16.7691944455582 |
| sample std, ddof=1 | 16.769194445558234 |
| CellProfiler | 16.296727949274324 |
| population std, ddof=0 | 16.296727687892847 |
| ratio | 1.028991510855053 |
| `sqrt(18/17)` | 1.0289915108550531 |

No tolerance should absorb that, so the row reads `regression` with
`candidate_oracle=cellprofiler (MeasureObjectIntensity)` and `flag=estimator-divergence`, and its
snapshot lives in `test_2d_morphology_regression.h` — the `_cellprofiler` file holds only assertions
CellProfiler actually backs. The generator checks the relationship as an **identity** rather than
printing it, so it cannot drift unnoticed in either direction.

Which estimator Nyxus means to report is a `src/nyx` question and is left open; see
`not_covered.md` §C.

## The band was not what let the old assertion pass

Worth recording, because the intuitive explanation is wrong. The previous assertion used the shared
snapshot helper's 0.1% relative band, and it is tempting to say that band was loose enough to admit
the estimator gap. It was not: the gap is **2.9% relative, 29 times the band**, so 0.1% would have
caught CellProfiler's value comfortably.

It passed because the golden it compared against was the **Nyxus** number. A test named
`_cellprofiler` was judging Nyxus against itself — which is exactly the condition `not_covered.md`
§C exists to surface, and a sharper problem than a loose tolerance. Tightening the band to the
measured residual is still right; it fixes something else.

## The generator is bound to the fixture and the pins

A generator holding its own copy of either would only ever compare itself against itself. This one
parses the fixture out of `test_data.h` and both reference tables out of the headers, and enforces
key equality in **both** directions — every pin must be a feature the recipe vets and must match the
run, and every feature CellProfiler vets must be pinned. Mutation-checked; each exits 1:

| mutation | caught as |
|---|---|
| edit a pin in the header | `FAIL EDGE_MEAN_INTENSITY … rel=1.57e-06` |
| delete a pin | `UNPINNED EDGE_MIN_INTENSITY: CP vets it but the header pins nothing` |
| add a pin CellProfiler does not back | `EXTRA EDGE_STDDEV_INTENSITY … this recipe does not vet it` |
| edit the Nyxus stddev pin | `IDENTITY BROKEN -- investigate` |
| edit the fixture in `test_data.h` | `FAIL EDGE_INTEGRATED_INTENSITY … rel=0.0106` |

## Scope

These five rows vet **one** production cell: `ContourFeature::calculate()` at `SINGLEROI=false`,
the in-RAM segmented contour, for the `EDGE_*` statistics — and
`BasicMorphologyFeatures::calculate()` for `MASS_DISPLACEMENT`, which reads no contour and has its
own `osized_calculate()`, so it does not inherit a contour-builder cell.

Not covered here: `SINGLEROI=true` (`buildWholeSlideContour`) and the out-of-core
`osized_calculate()` path. `matrix/morphology.md` records those cells.

## Reproduction

The CellProfiler env is offline-only — CI never invokes it, and CellProfiler is not a runtime
dependency. Two environment notes cost time and are recorded in the generator's docstring:
activate the env rather than calling its `python.exe` by path (otherwise importing
`cellprofiler_core.image` dies on a DLL ordinal lookup with nothing on stderr), and run from a
working directory on the same drive as the env (`cellprofiler.modules` pulls in
`cellprofiler.gui.help.content`, which calls `os.path.relpath()` against the CWD).
