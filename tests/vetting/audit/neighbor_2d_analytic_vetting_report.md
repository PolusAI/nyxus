# 2D neighbour vs the analytic recomputation — vetting report

Six features — the second-closest distance and the five angle statistics — are vetted against an
independent numpy recomputation of their documented closed forms. This re-vet re-ran that
recomputation, measured the residual for the first time, and found the generator was not checking
what it claimed to check.

## Tool and configuration

| | |
|---|---|
| Tool | analytic — an independent numpy recomputation, no external package |
| Generator | `tests/vetting/oracles/gen_neighbor_analytic.py` |
| Recipe | `neighbor.scene2d_radius1` |
| Fixture | `neighborhood2d_scene_labels` (`tests/test_data.h`), five ROIs |
| Nyxus config | `PIXELDISTANCE=1`, `PIXELSIZEUM=1`, `XYRES=1`, `IBSI=false` |
| Test | `test_2d_neighbor_analytic.h` |
| Tolerance | `rel=1e-9` (SPEC §7 exact tier) |

## Why an analytic oracle is legitimate here, and why it is not circular

`analytic` is the weakest oracle token in SPEC §4 precisely because a recomputation of the
implementation's own algorithm can be circular — a match then proves only that two copies of the
same mistake agree. That is the trap `revet.txt` §3 warns about, and it is worth stating exactly why
it does not apply:

- The six features are **closed forms of the ROI centroids given the neighbour graph**. The
  arithmetic is `atan2`, a mean, a sample standard deviation and a mode — textbook formulas, not
  Nyxus-specific procedures.
- **The graph itself is vetted independently.** `NUM_NEIGHBORS` and `CLOSEST_NEIGHBOR1_DIST` are
  reproduced exactly by CellProfiler (`neighbor_2d_cellprofiler_vetting_report.md`), so the input the
  closed forms are evaluated on is externally confirmed. CellProfiler supplies the graph; the
  recomputation supplies the arithmetic.
- The recomputation is written from the documented formulas, not transcribed from `neighbors.cpp`.

What would make it circular is recomputing a Nyxus-specific *procedure* — the histogram-interpolation
percentile case in `revet.txt` §3 — and that is not what happens here.

## Why CellProfiler cannot vet these six

Not disagreement — different quantities:

- CP's **`AngleBetweenNeighbors`** is the angle *subtended at an object by two of its neighbours*.
  Nyxus' `CLOSEST_NEIGHBOR*_ANG` and `ANG_BW_NEIGHBORS_*` are *absolute* `atan2` direction angles in
  `[0, 360)`. Different measurement entirely.
- CP's **`SecondClosestDistance`** ranges over **any object**; Nyxus' `CLOSEST_NEIGHBOR2_DIST` is the
  second-closest neighbour *within the search radius*, hence 0 when an ROI has fewer than two
  in-radius neighbours. On this fixture CP reports 2.83–3.20 where Nyxus reports 0, for exactly that
  reason.

Comparing either pair would be a category error, not a vetting failure.

## Result

Per ROI, never aggregated — all 30 values, every one asserted:

| feature | ROI | Nyxus | analytic oracle | rel |
|---|---|---|---|---|
| `CLOSEST_NEIGHBOR2_DIST` | 1 | 2.5495097567963922 | 2.5495097567963922 | 0 |
| `CLOSEST_NEIGHBOR2_DIST` | 2 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR2_DIST` | 3 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR2_DIST` | 4 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR2_DIST` | 5 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR1_ANG` | 1 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR1_ANG` | 2 | 11.309932474020213 | 11.309932474020213 | 0 |
| `CLOSEST_NEIGHBOR1_ANG` | 3 | 78.69006752597979 | 78.69006752597979 | 0 |
| `CLOSEST_NEIGHBOR1_ANG` | 4 | 180.0 | 180.0 | 0 |
| `CLOSEST_NEIGHBOR1_ANG` | 5 | 258.69006752597977 | 258.69006752597977 | 0 |
| `CLOSEST_NEIGHBOR2_ANG` | 1 | 191.3099324740202 | 191.3099324740202 | 0 |
| `CLOSEST_NEIGHBOR2_ANG` | 2 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR2_ANG` | 3 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR2_ANG` | 4 | 0.0 | 0.0 | 0 |
| `CLOSEST_NEIGHBOR2_ANG` | 5 | 0.0 | 0.0 | 0 |
| `ANG_BW_NEIGHBORS_MEAN` | 1 | 132.17251688149494 | 132.17251688149494 | 0 |
| `ANG_BW_NEIGHBORS_MEAN` | 2 | 11.309932474020213 | 11.309932474020213 | 0 |
| `ANG_BW_NEIGHBORS_MEAN` | 3 | 78.69006752597979 | 78.69006752597979 | 0 |
| `ANG_BW_NEIGHBORS_MEAN` | 4 | 180.0 | 180.0 | 0 |
| `ANG_BW_NEIGHBORS_MEAN` | 5 | 258.69006752597977 | 258.69006752597977 | 0 |
| `ANG_BW_NEIGHBORS_STDDEV` | 1 | **115.23001801020592** | **115.23001801020591** | **1.2e-16** |
| `ANG_BW_NEIGHBORS_STDDEV` | 2 | 0.0 | 0.0 | 0 |
| `ANG_BW_NEIGHBORS_STDDEV` | 3 | 0.0 | 0.0 | 0 |
| `ANG_BW_NEIGHBORS_STDDEV` | 4 | 0.0 | 0.0 | 0 |
| `ANG_BW_NEIGHBORS_STDDEV` | 5 | 0.0 | 0.0 | 0 |
| `ANG_BW_NEIGHBORS_MODE` | 1 | 0.0 | 0.0 | 0 |
| `ANG_BW_NEIGHBORS_MODE` | 2 | 11.0 | 11.0 | 0 |
| `ANG_BW_NEIGHBORS_MODE` | 3 | 79.0 | 79.0 | 0 |
| `ANG_BW_NEIGHBORS_MODE` | 4 | 180.0 | 180.0 | 0 |
| `ANG_BW_NEIGHBORS_MODE` | 5 | 259.0 | 259.0 | 0 |

| | |
|---|---|
| values compared | 30 (6 features × 5 ROIs) |
| bit-identical to the recomputation | **29** |
| worst residual | **1.23e-16** — `ANG_BW_NEIGHBORS_STDDEV` on ROI 1, one ulp of 115.23001801020591 |

Eighteen of the thirty goldens are **structural zeros**: `CLOSEST_NEIGHBOR2_DIST`/`_ANG` are 0 when
fewer than two neighbours lie within the radius, and the sample standard deviation of a single angle
is 0. `agrees_gt` computes `tolerance = golden / frac_tolerance`, so a zero golden demands **bit-exact
equality** — for those eighteen the assertion is as strong as it can be, and it passes.

## What did not hold up

### The band, again

`ASSERT_NEAR(..., 1e-4)` absolute against a measured 1.2e-16, with the registry's `tolerance` column
empty. Now `rel=1e-9` in both.

### The generator compared itself against itself

`gen_neighbor_analytic.py` validated its run against a `PINNED` dict **hardcoded inside the
generator** — and the comment above that dict, plus the banner it printed, named
`test_2d_neighbor_regression.h`, a file those goldens had **already been moved out of**. So the
self-check was doubly detached from reality: it read no header at all, and it named the wrong one.

It now parses `test_2d_neighbor_analytic.h`, verifies all 30 pins, reports any feature it produces
that the header fails to pin, and exits non-zero on failure. Current run: **30 verified, 0 failed, 0
unproducible, 0 unpinned**.

**Negative control.** Perturbing the pinned `CLOSEST_NEIGHBOR1_ANG` on ROI 5 by 1e-5 absolute — a
change the previous ±1e-4 band would have passed — now fails the gtest naming
`ANALYTIC__CLOSEST_NEIGHBOR1_ANG__L5`, and independently fails the generator with
`rel=3.87e-08 ... SOME CHECKS FAILED` and exit 1. The old generator could not have caught it at all.

### The pins were truncated

Every golden had been recorded to ~15 significant digits (`2.54950975679639` for
`2.5495097567963922`), sitting 1.1e-15 from the oracle's own value. Harmless under the old band,
but it eats real margin under `rel=1e-9`. All 30 are re-pinned at full `repr` precision, straight
from the generator's paste-ready output.

### A parser bug, caught by a guard added in this same pass

The new header parser initially used a non-greedy regex to split label blocks, which swallows the
closing brace of each block's **last** entry. On the multi-line analytic table it happened to work;
on the single-line CellProfiler table it silently dropped `CLOSEST_NEIGHBOR1_DIST` from every ROI.
The "every vetted feature must actually be pinned" check written alongside it is what caught this —
it reported five `UNPINNED` rows rather than a clean pass. Both generators now split label blocks by
**counting braces**, which is layout-independent.

Worth keeping: the analytic table parsed correctly only because of where its trailing commas and
newlines fell. It was right by accident, one reformat away from being wrong, and no assertion would
have noticed.

## Reproduction

```
python tests/vetting/oracles/gen_neighbor_analytic.py     # numpy only, no special environment
```

Regenerating the goldens: `neighbor_2d_golden_regen.md`.
