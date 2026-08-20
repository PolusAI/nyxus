# Regenerating the 2D neighbour goldens

Concrete steps for each of the family's three tables. Everything runs offline; CI never invokes a
reference tool.

All three share one fixture and one recipe (`neighbor.scene2d_radius1`): the
`neighborhood2d_scene_labels` scene from `tests/test_data.h`, five ROIs, `PIXELDISTANCE=1`. The scene
is duplicated as a `SCENE` list in both generators — if `test_data.h` ever changes, both copies must
be updated with it.

## 1. `test_2d_neighbor_analytic.h` — the analytic oracle

**Environment.** numpy only; the `nyxus_build` env is enough.

```
python tests/vetting/oracles/gen_neighbor_analytic.py        # from the repository root
```

Prints a paste-ready `{label, {{"FEATURE", value}, ...}}` block per ROI, then re-verifies every pin
already in the header and exits non-zero on a mismatch, on a pin it cannot produce, or on a feature
it produces that the header fails to pin. Paste the printed blocks into
`neighbor_2d_analytic_ref_vals_by_label` and run it again: it must report
`30 verified, 0 failed, 0 unproducible, 0 unpinned`.

**Formulas** (all from the ROI centroids, given the neighbour graph):

| quantity | definition |
|---|---|
| direction angle | `degrees(atan2(dy, dx))`, mapped into `[0, 360)` |
| closest / second-closest | by centroid distance; ties keep ascending-label push order |
| `CLOSEST_NEIGHBOR2_DIST` / `_ANG` | **0** when fewer than two neighbours lie within the radius |
| `ANG_BW_NEIGHBORS_STDDEV` | **sample** (n−1) standard deviation; 0 for a single angle |
| `ANG_BW_NEIGHBORS_MODE` | most frequent `round(angle)` bucket, lowest bucket winning a tie |

**Known convention differences to account for:** the sample-vs-population standard deviation (n−1,
not n) and the in-radius restriction on the second-closest neighbour. Both are places where a
plausible reimplementation would differ from Nyxus and the disagreement would look like a defect.

## 2. `test_2d_neighbor_cellprofiler.h` — the CellProfiler oracle

**Environment.** The `nyxus_cellprofiler` conda env; the build recipe is in `TOOLS.md`. A full
`pip install cellprofiler` does not build on Windows — the env installs `cellprofiler-core` +
`centrosome` and then the module package with `--no-deps`.

The JVM must be reachable or javabridge fails at import with a bare exit 127:

```
conda activate nyxus_cellprofiler
set JAVA_HOME=%CONDA_PREFIX%\Library\lib\jvm
set JDK_HOME=%JAVA_HOME%
python tests/vetting/oracles/gen_neighbor_cellprofiler.py
```

Prints one paste-ready line per ROI, re-verifies all 10 pins, prints the documented
`PERCENT_TOUCHING` divergence table (reading Nyxus' side out of the regression header, not a copy),
and exits non-zero on failure. Must report `10 verified, 0 failed, 0 unproducible, 0 unpinned`.

**Config mapping.**

| Nyxus | CellProfiler |
|---|---|
| `PIXELDISTANCE=1` | `distance_method=Adjacent` (the pixel-adjacency definition) |
| one scene, five labelled ROIs | `Objects().segmented = <label image>`, indexed `[row=y, col=x]` |
| — | the label image is padded by **3 px**, because CP treats border-touching objects specially |
| `NUM_NEIGHBORS` | `Neighbors_NumberOfNeighbors_Adjacent` |
| `CLOSEST_NEIGHBOR1_DIST` | `Neighbors_FirstClosestDistance_Adjacent` |

CellProfiler must be put in headless mode (`cellprofiler_core.preferences.set_headless()`) **before**
`Measurements()` is constructed, or it imports wx and fails.

**Only two features are CP-vettable.** `SecondClosestDistance` and `PercentTouching` are different
quantities, not disagreements — see `neighbor_2d_cellprofiler_vetting_report.md`. Do not "fix" them
by widening a band.

## 3. `test_2d_neighbor_regression.h` — the PERCENT_TOUCHING drift pins

Not an oracle table: it records what Nyxus itself computes for the one feature with no promotable
reference. To re-record after a deliberate change, print the per-ROI value at this recipe and paste
the full `%.17g` value — do not round, since the band is `agrees_gt`'s `rel=1e-3` default and a pin
truncated to six digits eats most of it before the test starts.

Re-recording these is a **deliberate act**, not a fix for a red test. A drift guard going red means
either the change was intended — say so in the commit message — or it was not, in which case the pin
is the finding.

## What is deliberately not in any table

`test_2d_neighbor_invariant.h` holds no reference data at all. It asserts `0 ≤ PERCENT_TOUCHING ≤
100` and that a fully enclosed ROI gives exactly 100 — properties of the definition, checked without
comparing against anything. It carries `_invariant` names rather than `_analytic` ones on purpose:
the coverage scanners read the oracle token off the function name, so an oracle-suffixed name there
would credit the feature with an oracle it does not have.
