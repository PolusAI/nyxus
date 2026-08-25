# 2D neighbour vs CellProfiler — vetting report

`NUM_NEIGHBORS` and `CLOSEST_NEIGHBOR1_DIST` have been marked `vetted` against CellProfiler since PR
#390. This re-vet re-ran the tool rather than trusting that label, and the values hold — **exactly**,
not approximately. What did not hold up was everything around them: the assertion band, the
generator's self-check, and where the `PERCENT_TOUCHING` bound assertions lived.

## Tool and configuration

| | |
|---|---|
| Tool | CellProfiler 4.2.8 (module package) / cellprofiler-core 4.2.8.1 / centrosome 1.2.3, Python 3.9 |
| Module | `cellprofiler.modules.MeasureObjectNeighbors`, `distance_method=Adjacent`, `neighbors_are_objects=True` |
| Env | `nyxus_cellprofiler` (conda) — see `TOOLS.md`; needs `JDK_HOME`/`JAVA_HOME` set or javabridge fails to load the JVM |
| Generator | `tests/vetting/oracles/gen_neighbor_cellprofiler.py` |
| Recipe | `neighbor.scene2d_radius1` |
| Fixture | `neighborhood2d_scene_labels` (`tests/test_data.h`), five ROIs, padded by 3 px so no ROI touches the border |
| Nyxus config | `PIXELDISTANCE=1`, `PIXELSIZEUM=1`, `XYRES=1`, `IBSI=false` |
| Test | `test_2d_neighbor_cellprofiler.h` |
| Tolerance | `abs=1e-9` (SPEC §7 exact tier, which is an absolute band) |
| Benchmark | `bench_scene7_5roi_enclosed` |
| Config matrix | `matrix/neighbor.md` |

## Result

Per ROI, never aggregated — the fixture is five ROIs and each carries its own value:

| feature | ROI | CellProfiler | Nyxus | rel |
|---|---|---|---|---|
| NUM_NEIGHBORS | 1 | 4.0 | 4.0 | 0 |
| NUM_NEIGHBORS | 2–5 | 1.0 | 1.0 | 0 |
| CLOSEST_NEIGHBOR1_DIST | 1 | 2.5 | 2.5 | 0 |
| CLOSEST_NEIGHBOR1_DIST | 2 | 2.5495097567963922 | 2.5495097567963922 | 0 |
| CLOSEST_NEIGHBOR1_DIST | 3 | 2.5495097567963922 | 2.5495097567963922 | 0 |
| CLOSEST_NEIGHBOR1_DIST | 4 | 2.5 | 2.5 | 0 |
| CLOSEST_NEIGHBOR1_DIST | 5 | 2.5495097567963922 | 2.5495097567963922 | 0 |

**All ten residuals are exactly 0** — the same double, not a close one. That is what sets the band at
the exact tier.

## What did not hold up

### The band was ~9 orders of magnitude looser than the agreement

The assertion was `ASSERT_NEAR(..., 1e-4)` — an **absolute** tolerance, against a measured residual
of **0** on both features here. Nine orders of magnitude of slack over an exact match; across the
family the same 1e-4 sat over `CLOSEST_NEIGHBOR1_ANG`'s 258.69, a relative 3.9e-7. The registry's `tolerance` column was **empty** on all nine rows,
so nothing recorded what was being claimed either.

Both are now set from the measurement: `abs=1e-9` in the header and in the registry — SPEC §7
spells the exact tier "`exact` (abs 1e-9)", so the band is absolute and the assertion is
`ASSERT_NEAR`, matching `test_2d_gldzm_mirp.h`, the other file at this tier. On values of 2.5 a
relative 1e-9 would have been the looser of the two anyway, at 2.5e-9.

### The generator validated itself, not the header

`gen_neighbor_cellprofiler.py` compared its CP run against a `CP_VETS` dict **hardcoded inside the
generator**, at `TOL = 1e-4`. Editing a golden in `test_2d_neighbor_cellprofiler.h` would not have
been caught by anything: the generator never read the header. Its Nyxus-side `PERCENT_TOUCHING`
divergence table was a second hardcoded copy, already stale to 6 significant figures.

It now parses both headers — the pins it feeds and the regression file it reports divergence
against — verifies every pin, checks that no vetted feature has silently lost its pin, and exits
non-zero on any failure. It also reads the scene itself out of `tests/test_data.h` rather than
carrying a transcription of it, so a fixture edit reaches CellProfiler instead of leaving the
generator driving the old scene. Current run: **10 verified, 0 failed, 0 unproducible, 0 unpinned**,
every residual `abs = 0`.

## Range, identity and cross-table checks on the pinned goldens

Run mechanically over **all 45 pins in the family** — 30 analytic, 10 CellProfiler, 5 regression —
not spot-checked:

- **Range.** `NUM_NEIGHBORS` a non-negative integer; both distances ≥ 0; all four angle features in
  `[0, 360)`; `ANG_BW_NEIGHBORS_STDDEV` ≥ 0; `PERCENT_TOUCHING` in `[0, 100]`.
- **Identity.** For every single-neighbour ROI — which is four of the five — the mean of one angle
  *is* that angle (`ANG_BW_NEIGHBORS_MEAN == CLOSEST_NEIGHBOR1_ANG`, exactly), its sample standard
  deviation is 0, and both `CLOSEST_NEIGHBOR2_*` are the structural zero. The mode must equal
  `round()` of a pinned angle. `CLOSEST_NEIGHBOR2_DIST` is never nearer than
  `CLOSEST_NEIGHBOR1_DIST` where it exists.
- **Cross-table.** Every table covers exactly labels 1–5, and **no feature is pinned in more than one
  table** — the check that confirms the duplicate snapshot assertions really were removed rather than
  merely moved.

All pass. These catch a rotted pin instantly; they cannot catch a wrong definition, so they are a
floor and not the vetting.

## PERCENT_TOUCHING: a definition gap, and a misfiled test

CP and Nyxus measure different things:

- **Nyxus** — distinct contour pixels 8-adjacent to a neighbour, over contour length.
- **CellProfiler** — object outline pixels overlapping a `disk(distance+0.5)`-dilated neighbour,
  over perimeter.

Measured on this fixture, Adjacent method:

| ROI | CP | Nyxus | verdict |
|---|---|---|---|
| 1 | 100.0000 | 100.0000 | agree (both saturate: the ROI is fully enclosed) |
| 2 | 50.0000 | 66.6667 | **diverge** |
| 3 | 50.0000 | 66.6667 | **diverge** |
| 4 | 50.0000 | 50.0000 | agree |
| 5 | 50.0000 | 33.3333 | **diverge** |

Three of five ROIs, up to 33.3 percentage points. No CP distance method (Adjacent / Expand / Within)
reproduces Nyxus. This is a convention difference, not a defect, so the feature stays `regression`
with its values pinned rather than compared.

**The misfiled test.** The bound assertions for this feature lived in `test_2d_neighbor_analytic.h`,
in a function named `test_2d_neighbor_percent_touching_enclosed_analytic`. Every coverage scanner in
this tree — `check_test_names.py`, `report_feature_tests.py`, the per-family `scan_*` scripts —
attributes an oracle from the test function's **name suffix**. So a feature the registry correctly
records as having no oracle was being asserted under a name that reads as an oracle claim, and a
scan would have credited `PERCENT_TOUCHING` with `oracle=analytic`.

By SPEC §4.4 these are invariants — "does output obey a required property/bound/relation" — not
oracle comparisons: `0 ≤ PT ≤ 100` is a bound, and "a fully enclosed ROI has every contour pixel
adjacent to a neighbour, so PT = 100 exactly" is a closed form of the definition, not a value
produced by a reference implementation. They now live in `test_2d_neighbor_invariant.h` under
`_invariant` names, which is also why `check_coverage.py` still shows this feature with no oracle —
correctly, since `invariant` is not an allowed `status` there.

## Reproduction

```
conda activate nyxus_cellprofiler
set JAVA_HOME=%CONDA_PREFIX%\Library\lib\jvm      # javabridge finds the JVM through this
set JDK_HOME=%JAVA_HOME%
python tests/vetting/oracles/gen_neighbor_cellprofiler.py
```

Nyxus side: build `runAllTests` with `-DRUN_GTEST=ON` and run
`--gtest_filter=*NEIGHBOR*`. Regenerating the goldens: `neighbor_2d_golden_regen.md`.
