# 2D morphology vs imea — vetting report

This family's imea rows were not part of the planned gap, but running the oracle showed the tree
disagreed with the registry in a way no naming check could catch. This report records what was
measured and what changed.

## Tool and configuration

| | |
|---|---|
| Tool | imea 0.3.5, numpy, env `nyxus_mirp` (conda) |
| Generator | `tests/vetting/oracles/gen_morphology_imea.py` |
| Recipes | `morphology.shape2d_native` (ISO transforms), `morphology.caliper_ellipse` (calipers) |
| Test | `test_2d_morphology_imea.h` |

```
python tests/vetting/oracles/gen_morphology_imea.py
```

Verifies every golden in both tables, prints the two evidence blocks below, and exits non-zero on
mismatch. Current run: **20 verified, 0 failed, 0 unproducible**.

## Finding 1 — an imea-named table held Nyxus snapshots

`morphology_2d_imea_shape2d_ref_vals` pinned 19 caliper/chord statistics on the 8×8 `shape2d` mask,
read by a helper named `assert_caliper_imea()` inside `test_2d_morphology_caliper_stats_imea()`.
Those values had never been compared to imea. Running imea on the same mask
(`dalpha=10`, matching Nyxus' sweep):

| feature | pinned (Nyxus) | imea | rel |
|---|---|---|---|
| STAT_FERET_DIAM_MIN | 4.47301 | 6.00000 | 34.1% |
| STAT_FERET_DIAM_MAX | 6.32220 | 8.00000 | 26.5% |
| STAT_MARTIN_DIAM_MIN | 4.25885 | 4.00000 | 6.1% |
| STAT_MARTIN_DIAM_MAX | 6.12801 | 7.00000 | 14.2% |
| STAT_NASSENSTEIN_DIAM_MIN | 1.67316 | 3.00000 | 79.3% |
| STAT_NASSENSTEIN_DIAM_MAX | 6.24165 | 6.00000 | 3.9% |

The 8×8 raster is too coarse for the hull-vs-raster conventions to converge, which is why the real
imea comparison was always done on a clean ellipse instead. The 19 snapshots now live in
`test_2d_morphology_regression.h` under a `_regression`-suffixed function and claim nothing.

**Why this matters beyond the rows.** An oracle-named test asserting a snapshot table passes every
automated check in the repo: `check_test_names.py`, `report_feature_tests.py` and
`scan_morphology_coverage.py` all attribute an oracle from the function-name suffix. Only running
the tool finds it. This is the case revet.txt step 3 warns about.

## Finding 2 — the ellipse goldens were a mix of angular steps

The ellipse table is genuinely imea-derived (11 of its 12 values sat closer to imea than to Nyxus),
but the values came from different `dalpha` settings and one — `STAT_NASSENSTEIN_DIAM_MIN` = 16.0 —
matched no run at any step tested (5, 9, 10, 15, 18, 20, 30, 36; the closest was 17.0).

Nyxus sweeps its calipers at a fixed 10° step (`rot_angle_increment`, `src/nyx/features/caliper.h`),
so `dalpha=10` is the matched configuration. Regenerated at that step, all 17 goldens come from one
run and the worst Nyxus-vs-imea residual falls from 8.9% to **4.99%**, which let the assertion
tolerance tighten from `reltol=0.10` to `0.06`. The old 10% bound existed to accommodate the stale
pin.

| feature | imea (dalpha=10) | Nyxus | rel |
|---|---|---|---|
| STAT_MARTIN_DIAM_MIN | 19.0 | 19.870567 | 4.58% |
| STAT_MARTIN_DIAM_MAX | 41.0 | 39.933333 | 2.60% |
| STAT_MARTIN_DIAM_MEAN | 27.5 | 27.114361 | 1.40% |
| STAT_MARTIN_DIAM_MEDIAN | 25.5 | 24.999570 | 1.96% |
| STAT_MARTIN_DIAM_STDDEV | 7.197607627229728 | 6.898214 | 4.16% |
| STAT_NASSENSTEIN_DIAM_MIN | 18.0 | 17.423218 | 3.20% |
| STAT_NASSENSTEIN_DIAM_MAX | 41.0 | 40.000000 | 2.44% |
| STAT_NASSENSTEIN_DIAM_MEAN | 24.833333333333332 | 24.239812 | 2.39% |
| STAT_NASSENSTEIN_DIAM_MEDIAN | 21.0 | 20.208603 | 3.77% |
| STAT_NASSENSTEIN_DIAM_STDDEV | 7.365459931328117 | 7.384585 | 0.26% |
| STAT_FERET_DIAM_MIN | 21.0 | 20.000000 | 4.76% |
| STAT_FERET_DIAM_MAX | 41.0 | 40.000000 | 2.44% |
| STAT_FERET_DIAM_MEAN | 31.555555555555557 | 31.150595 | 1.28% |
| STAT_FERET_DIAM_MEDIAN | 32.0 | 32.966415 | 3.02% |
| STAT_FERET_DIAM_STDDEV | 6.7595382996689475 | 7.096753 | 4.99% |
| ALLCHORDS_MIN | 1.0 | 1.000000 | 0 |
| DIAMETER_MIN_ENCLOSING_CIRCLE | 40.00019836425781 | 40.000198 | 0 |

The 4.99% floor is `STAT_FERET_DIAM_STDDEV`. Nyxus' Feret sweep runs `theta = 0..180` **inclusive**
(`caliper_feret.cpp:89`) so it counts the 0/180 direction twice, while Martin and Nassenstein stop at
`theta < 180`. That tilts the spread slightly; it is a sampling-definition difference, not a
precision loss.

## Finding 3 — the MODE statistics cannot be vetted, at any tolerance

`STAT_{FERET,MARTIN,NASSENSTEIN}_DIAM_MODE` carried `status=vetted, oracle=imea` with no imea
comparison anywhere. Attempting one shows why none is possible: the mode of a caliper distribution is
an artifact of the angular sampling step, so the oracle disagrees with *itself* by more than it
disagrees with Nyxus.

| dalpha | feret_mode | martin_mode | nassenstein_mode |
|---|---|---|---|
| 5 | 21 | 19 | 20 |
| 9 | 21 | 21 | 24 |
| 10 | 22 | 20 | 20 |
| 15 | 23 | 22 | 19 |
| 18 | 24 | 20 | 21 |
| 30 | 21 | 22 | 21 |
| **Nyxus** | **20** | **19** | **18** |

A tolerance wide enough to admit these would be wide enough to admit a genuinely wrong value, which
SPEC §7 calls a test bug. All three are demoted to `status=regression` and drift-pinned in
`test_2d_morphology_regression.h`.

The standard deviations behave the opposite way — stable to ~3% in the oracle across that same sweep
and agreeing with Nyxus to 0.26–5.0% — so they, and `ALLCHORDS_MIN` (exact), are newly vetted.

## Finding 4 — three goldens were pinned twice, and had already drifted

`DIAMETER_EQUAL_PERIMETER`, `GEODETIC_LENGTH` and `THICKNESS` each existed in the ref-vals table
*and* as a literal inside the assertion, with the assertions reading the literals — so the table
entries were dead and only the generator checked them. The two copies of
`DIAMETER_EQUAL_PERIMETER` had already diverged: `8.57365809435587` in the table against
`8.573658094355881` in the assertion. The assertions now read the table.

## Scope of the ISO-transform claims

`DIAMETER_EQUAL_PERIMETER` (perimeter/π) and `GEODETIC_LENGTH` / `THICKNESS` (the DIN ISO 9276-6
rectangle model) are vetted as **transforms**: imea's `macro` functions, fed the Nyxus area and
perimeter, reproduce the Nyxus values exactly. imea's end-to-end values on the same mask do not
agree, because it derives its own perimeter from `cv2.arcLength` (12.657) rather than a chain-code
walk (26.935). That gap is inherited entirely from PERIMETER, which is vetted separately against
scikit-image on the circles benchmark.
