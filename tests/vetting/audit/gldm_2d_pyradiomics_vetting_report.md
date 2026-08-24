# 2D GLDM vs PyRadiomics — vetting report

All 14 2D GLDM rows read `status=vetted, oracle=ibsi`, evidenced by `test_2d_gldm_ibsi.h`. Running
the oracles for the first time turned up three things: the goldens in that file are the IBSI
**NGLDM** consensus values under GLDM names, `GLDM_DE` is computed through an approximate
logarithm and misses the reference by 7.9e-4, and the family's only PyRadiomics-named test compared
the two tools at configurations that do not match.

## Tool and configuration

| | |
|---|---|
| Tool | pyradiomics v3.0.1, SimpleITK 2.3.1, numpy 1.23.5, env `nyxus_oracle` (conda, Python 3.9) |
| Generator | `tests/vetting/oracles/gen_gldm_pyradiomics.py` |
| Recipe | `gldm.ibsi_phantom_2d` |
| Fixture | the four IBSI digital-phantom slices, read out of `tests/test_data.h` by `oracles/ibsi_phantom.py` |
| Nyxus config | `IBSI=true`, `GREYDEPTH=128`, each slice featurised on its own and the four averaged |
| PyRadiomics config | `binWidth=1`, `gldm_a=0`, `distances=[1]`, `force2D`, `force2Ddimension=0` |
| Test | `test_2d_gldm_pyradiomics.h` |
| Tolerance | `rel=1e-9` (SPEC §7 exact tier) for 13 features; `rel=2.5e-3` for `GLDM_DE` (SPEC §7 documented residual — see below) |

```
python tests/vetting/oracles/gen_gldm_pyradiomics.py
```

Verifies every golden pinned in the header — both tables, 14 means and 56 per-slice values — and
exits non-zero on mismatch. Current run: **70 verified, 0 failed, 0 unproducible**, every one at
`rel = 0`. (The generator checks the pinned goldens against PyRadiomics, which is what pins them;
Nyxus' agreement with those goldens is the table below.)

`binWidth=1` is identity binning on this integer phantom, so neither tool discretises and both build
the dependence matrix over the phantom's own levels 1..6. `gldm_a=0` with `distances=[1]` is the
alpha=0, d=1 coarseness Nyxus computes in IBSI mode.

## Results — PyRadiomics reproduces 13 of the 14 to machine precision

| feature | PyRadiomics (= pinned golden) | Nyxus | rel |
|---|---|---|---|
| GLDM_SDE | 0.15807024738501638 | 0.15807024738501638 | 0 |
| GLDM_LDE | 19.173821809425526 | 19.173821809425526 | 0 |
| GLDM_GLN | 10.24637942896457 | 10.24637942896457 | 0 |
| GLDM_DN | 3.9646456828345373 | 3.9646456828345373 | 0 |
| GLDM_DNN | 0.21177218060411693 | 0.21177218060411693 | 0 |
| GLDM_GLV | 2.7037332451477982 | 2.7037332451477987 | 1.6e-16 |
| GLDM_DV | 2.729504577399913 | 2.729504577399913 | 0 |
| GLDM_DE | 2.714292423281547 | 2.7121510523248058 | **7.9e-4** |
| GLDM_LGLE | 0.7017531915300232 | 0.7017531915300232 | 0 |
| GLDM_HGLE | 7.486949604403165 | 7.486949604403165 | 0 |
| GLDM_SDLGLE | 0.047290498640367454 | 0.047290498640367454 | 0 |
| GLDM_SDHGLE | 3.064914180133554 | 3.064914180133554 | 0 |
| GLDM_LDLGLE | 17.59968920804189 | 17.59968920804189 | 0 |
| GLDM_LDHGLE | 49.477721878224976 | 49.477721878224976 | 0 |

Worst residual over the 13 features other than `GLDM_DE` is **1.6e-16** (`GLDM_GLV`, summation
order), with 12 of 14 bit-identical. PyRadiomics is the reference that defines GLDM, and Nyxus
implements the same definition over the same neighbourhood, so those 13 are pinned at the exact tier
rather than as a cross-tool band. `GLDM_DE` is twelve orders of magnitude worse and gets its own
band; the next section measures why.

### Per slice, because a mean cannot vet the four values behind it

The table above compares two averages, and an average is blind to errors in two slices that cancel.
PyRadiomics computes each slice on its own — the averaging is done by this generator, not by the
tool — so intercepting it before the mean yields a per-slice reference with no formula
reimplemented. All 14 features were measured against Nyxus slice by slice:

| | worst per-slice residual |
|---|---|
| `GLDM_DE` | **1.3e-3** (slice z1; see the next section) |
| `GLDM_SDE`, `GLDM_DV` | 2.2e-16 |
| `GLDM_SDLGLE` | 1.9e-16 |
| `GLDM_GLV`, `GLDM_LDLGLE` | 1.3e-16 |
| the other 8 | 0 |

Worst over the 13 exact-tier features x 4 = **2.2e-16**; nothing was cancelling, and `GLDM_DE`'s
per-slice spread is wider than its mean (1.3e-3 vs 7.9e-4), so the mean was flattering it. The
assertions pin all 56 values anyway,
in `gldm_2d_pyradiomics_slice_ref_vals`. For the 13 exact-tier features the measurement says the mean
*was* trustworthy here, not that it *is* a sufficient check; for `GLDM_DE` it says outright that it
was not, which is why the band is sized on the per-slice worst and not on the mean.

**Verified non-vacuous.** Moving `GLDM_LDE_z1` by +0.001 and `GLDM_LDE_z3` by -0.001 leaves the
four-slice mean unchanged to its last digit. The per-slice assertion fails and names `GLDM_LDE_z1`;
`test_2d_gldm_ibsi.h`, which checks only the mean, passes.

## `GLDM_DE` is computed with an approximate logarithm, and is banded for it

`GLDM_DE` is 2.7121510523248058 against the reference 2.7142924232815471 — off by **7.9e-4** on the
mean and by up to **1.3e-3** per slice. Every other feature in the family sits at 1e-16 or below,
so this one residual is twelve orders of magnitude larger than the rest.

`GLDMFeature::calc_DE()` reads its logarithm through `Nyxus::fast_log10` (`src/nyx/helpers/helpers.h`),
a **float** second-order polynomial approximating `log2` over `[0.75, 1.5)`. Measured against
`log2` across that reduction range its worst absolute error is **8.9e-3**, at x = 0.75. That is the
whole of the gap:

| slice | PyRadiomics | Nyxus (`fast_log10`) | rel |
|---|---|---|---|
| z1 | 3.0464393446710125 | 3.0425443887710575 | **1.3e-3** |
| z2 | 2.3789919869000604 | 2.376873304969386 | 8.9e-4 |
| z3 | 2.6098501660289402 | 2.6088697770062619 | 3.8e-4 |
| z4 | 2.8218881955261761 | 2.820316738552517 | 5.6e-4 |
| mean | 2.7142924232815471 | 2.7121510523248058 | 7.9e-4 |

Reproduced independently of Nyxus: re-summing PyRadiomics' own dependence matrix with the
`helpers.h` polynomial transcribed into Python — float arithmetic preserved operation by operation —
lands on 2.7121510523248058, Nyxus' value to the last bit. So the residual is the approximation and
nothing else: the dependence matrix itself is exact, which is why the other 13 features sit at 1e-16.

For reference, `std::log2(p)` in place of `fast_log10(p + EPS) / LOG10_2` reproduces PyRadiomics at
9.8e-16 — the same value `NGLDMfeature`'s dependence-count entropy already produces, and the same
2.7142924232815497 that mirp's `ngl_dc_entr` is pinned at in `test_2d_ngldm_mirp.h`.

**The band, not the source change.** `GLDM_DE` asserts at `rel=2.5e-3` — twice the measured
per-slice worst — under SPEC §7's "documented residual" tier, and the other 13 features keep
`rel=1e-9`. `fast_log10` stays. It is read by `glcm.cpp`, `glrlm.cpp`, `glszm.cpp`, `3d_glcm.cpp`,
`3d_gldm.cpp`, `3d_glrlm.cpp` and `3d_glszm.cpp`; every entropy in the texture set carries the same
~1e-3, and the 3D GLCM re-vet already recorded its share of it — `INFOMEAS1` 1.7e-2, `INFOMEAS2`
7.7e-3, `DIFENTRO` 1.2e-3, `SUMENTROPY` 7.6e-4, `JE` 4.1e-4, each given a feature-specific band at
twice its measured residual. This family follows that precedent rather than diverging from it: one
helper, one convention across every family that reads it, and one change when it changes.

**The band is not a free pass.** Every other GLDM feature holds `rel=1e-9` against the same
dependence matrix, per slice and on the mean, so an error in the matrix — the thing this family
actually implements — still fails, and fails on 13 features at once. What `rel=2.5e-3` absorbs is
one known, measured, single-line arithmetic residual in a shared helper, with the exact value it
would take pinned above.

**Nothing in the tree measured this before.** The only assertion on `GLDM_DE` was the IBSI one at
`rel=1e-2` against a value published to three significant figures (2.71) — 7.9e-4 sits comfortably
inside that band. The drift guard in `test_2d_gldm_regression.h` compares against 5.3430148357241016,
recorded from this same code path, so it cannot see the residual either, by construction. Both
remain correct; neither is evidence, and the number above is.

## The IBSI goldens are the NGLDM table, and that is correct

`gldm_2d_ibsi_ref_vals` holds the IBSI reference manual's **NGLDM** consensus values. IBSI defines no
GLDM: the two are the same statistic. A Nyxus GLDM dependence count is `1 +` the number of
8-neighbours sharing the centre's binned grey level (`gldm.cpp`, `int nd = 1`), which is IBSI's
`j = k + 1` at coarseness alpha=0, distance d=1. The features line up one for one, with the
dependence axis spelled small/large by Nyxus and low/high by IBSI:

| Nyxus | IBSI NGLDM | Nyxus | IBSI NGLDM |
|---|---|---|---|
| `GLDM_SDE` | low dependence emphasis | `GLDM_GLN` | grey level non-uniformity |
| `GLDM_LDE` | high dependence emphasis | `GLDM_DN` | dependence count non-uniformity |
| `GLDM_LGLE` | low grey level count emphasis | `GLDM_DNN` | normalised dependence count non-uniformity |
| `GLDM_HGLE` | high grey level count emphasis | `GLDM_GLV` | grey level variance |
| `GLDM_SDLGLE` | low dependence low grey level emphasis | `GLDM_DV` | dependence count variance |
| `GLDM_SDHGLE` | low dependence high grey level emphasis | `GLDM_DE` | dependence count entropy |
| `GLDM_LDLGLE` | high dependence low grey level emphasis | | |
| `GLDM_LDHGLE` | high dependence high grey level emphasis | | |

The identity is measured, not asserted from the names: on this fixture Nyxus' 14 GLDM values are
**bit-identical to the mirp-vetted NGLDM goldens already pinned in `test_2d_ngldm_mirp.h`**, once
`GLDM_DE` uses an exact logarithm. Both tables can be compared directly — they are two files in this
repository — so no extra generator run is needed to reproduce the check.

## The production config is not comparable to PyRadiomics

`tests/python/test_2d_gldm_pyradiomics.py` claimed a PyRadiomics comparison at the **production**
config (`ibsi=false`, MATLAB grey binning at `coarse_gray_depth=64`) on the canonical 154-px ROI,
pinning three values at `rel=0.20` / `rel=0.15`.

The three pinned numbers do reproduce — they are genuinely PyRadiomics' output on that fixture. What
they are not is agreement:

| feature | PyRadiomics | Nyxus | rel |
|---|---|---|---|
| GLDM_SDE | 0.815295815296 | 0.802534271284 | 0.016 |
| GLDM_LDE | 1.88311688312 | 2.06493506494 | **0.088** |
| GLDM_GLN | 3.8961038961 | 3.94805194805 | 0.013 |
| GLDM_DN | 95.7012987013 | 92.4415584416 | 0.035 |
| GLDM_DNN | 0.621437004554 | 0.60026986001 | 0.035 |
| GLDM_GLV | 312.832349469 | 311.869961208 | 0.003 |
| GLDM_DV | 0.263282172373 | 0.344408837915 | **0.24** |
| GLDM_DE | 5.93712319024 | 5.95011020322 | 0.002 |
| GLDM_LGLE | 0.0160307720963 | 0.0081913915026 | **0.96** |
| GLDM_HGLE | 1012.67532468 | 1064.90909091 | 0.049 |
| GLDM_SDLGLE | 0.0136053972358 | 0.00653737856421 | **1.08** |
| GLDM_SDHGLE | 762.577561328 | 752.621212121 | 0.013 |
| GLDM_LDLGLE | 0.0270097491459 | 0.0158086968728 | **0.71** |
| GLDM_LDHGLE | 2317.33766234 | 3165.14285714 | 0.27 |

The two tools discretise differently here: Nyxus re-bins the ROI with the MATLAB scheme at
`coarse_gray_depth=64`, PyRadiomics bins at `binWidth=1` over the values it is given, so the two
build their dependence matrices over different level assignments. The grey-level-weighted features
diverge most because Nyxus multiplies by the binned intensity while PyRadiomics multiplies by its
own discretised level index. Per SPEC §5 a tolerance cannot absorb a configuration mismatch, so
**this is not an oracle assertion at any band** — the ±15–20% the file used was sized to let a
factor-of-two disagreement pass silently.

The file's real subject is the bug #14b background-pollution guard on the production path, which the
IBSI-mode oracle tests cannot reach (they run on fully-masked ROIs). It is renamed to
`tests/python/test_2d_gldm_mechanics.py` — its actual SPEC §2 kind — and re-pinned against Nyxus'
own values at `rel=1e-9`, keeping the `GLDM_LDE < 5.0` bound that fails on the pre-fix inflated
~16.6. Its C++ counterpart, `test_2d_gldm_mechanics.h`, already had the right name and pointed at
the old one.

## Registry corrections

All 14 rows: `oracle` `ibsi` → `pyradiomics`, `config_recipe` → `gldm.ibsi_phantom_2d`,
`current_test` → `test_2d_gldm_ibsi.h;test_2d_gldm_pyradiomics.h`, and `tolerance` → `rel=1e-9`
except `GLDM_DE`, which reads `rel=2.5e-3`. No row changes
`status`; the family was vetted and remains so, now against the tool that defines the family and at
up to 15 more digits — 15 for the 13 exact-tier features, 6 for `GLDM_DE`.

Every row's `current_test` had listed **`test_3d_gldm_regression.h` and `test_3d_gldm_pyradiomics.h`**
— 3D files, for 2D rows, naming a different implementation (`3d_gldm.cpp`). `test_3d_gldm_regression.h`
is additionally one of the headers `test_all.cc` never includes, so it runs nothing at all. Those
citations are dropped.
