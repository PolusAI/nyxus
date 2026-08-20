# 2D GLSZM vs mirp — vetting report

Ten of the family's sixteen rows have claimed `oracle=pyradiomics` with
`target_test=test_2d_glszm_pyradiomics.h` since the tracker was imported — **a file that does not
exist**. The other six claimed `oracle=ibsi`, backed by a file that does exist and asserts all
sixteen features at ±1%.

Running independent oracles for the first time found Nyxus reproducing **both** mirp and PyRadiomics
exactly on fifteen of the sixteen features, and missing the sixteenth, `GLSZM_ZE`, by 2.5e-3 against
each of them — the `fast_log10` approximation, not a GLSZM defect. The family is now vetted against
mirp per slice and on the four-slice mean, with PyRadiomics corroborating every value.

## Tool and configuration

| | |
|---|---|
| Tool | mirp 2.6.0, numpy 2.4.6, env `nyxus_mirp` (conda) |
| Generator | `tests/vetting/oracles/gen_glszm_mirp.py` |
| Recipe | `glszm.ibsi_phantom_2d` |
| Fixture | the four IBSI digital-phantom slices, read out of `tests/test_data.h` by `oracles/ibsi_phantom.py` |
| Nyxus config | `IBSI=true`, `GREYDEPTH=128`, each slice featurised on its own |
| mirp config | `by_slice=True`, `base_discretisation_method="none"` |
| Test | `test_2d_glszm_mirp.h` |
| Tolerance | `rel=1e-9` (SPEC §7 exact tier); `rel=4e-3` for `GLSZM_ZE` alone |
| Corroborating run | PyRadiomics 3.0.1, env `nyxus_oracle` (conda) — see below |

```
python tests/vetting/oracles/gen_glszm_mirp.py
```

Verifies every golden pinned in the header — both tables, 16 means and 64 per-slice values — and
exits non-zero on mismatch. Current run: **80 verified, 0 failed, 0 unproducible**, every one at
`rel = 0`.

GLSZM takes no distance or coarseness parameter on either side: a zone is a maximal connected set of
equal-valued voxels, so there is nothing to configure beyond the discretisation.

## Result

Nyxus against mirp, four-slice mean, and the worst of the four per-slice residuals:

| feature | Nyxus (4-slice mean) | mirp | rel (mean) | worst rel (slice) |
|---|---|---|---|---|
| GLSZM_SAE | 0.36330794123204835 | 0.36330794123204835 | 0 | 0 |
| GLSZM_LAE | 43.86666666666667 | 43.86666666666667 | 0 | 0 |
| GLSZM_LGLZE | 0.3711970899470899 | 0.3711970899470899 | 0 | 0 |
| GLSZM_HGLZE | 16.44047619047619 | 16.44047619047619 | 0 | 0 |
| GLSZM_SALGLE | 0.025854788674729148 | 0.025854788674729148 | 0 | 0 |
| GLSZM_SAHGLE | 10.277990480914587 | 10.277990480914587 | 0 | 0 |
| GLSZM_LALGLE | 40.398082010582016 | 40.398082010582016 | 0 | 0 |
| GLSZM_LAHGLE | 112.52142857142857 | 112.52142857142857 | 0 | 0 |
| GLSZM_GLN | 1.4142857142857144 | 1.4142857142857144 | 0 | 0 |
| GLSZM_GLNN | 0.3229931972789115 | 0.3229931972789115 | 0 | 0 |
| GLSZM_SZN | 1.4857142857142858 | 1.4857142857142858 | 0 | 0 |
| GLSZM_SZNN | 0.3331972789115646 | 0.3331972789115646 | 0 | 0 |
| GLSZM_ZP | 0.24038957688338494 | 0.24038957688338494 | 0 | 0 |
| GLSZM_GLV | 3.9694784580498865 | 3.9694784580498865 | 0 | 1.2e-16 |
| GLSZM_ZV | 20.997052154195014 | 20.99705215419501 | 1.7e-16 | 2.0e-16 |
| **GLSZM_ZE** | **1.9280961666788374** | **1.9319448617396766** | **2.0e-3** | **2.5e-3** |

Fifteen features are bit-identical. Nyxus and mirp build the same zones over the same
8-connected neighbourhood and discretise the phantom the same way, so this is the exact tier rather
than a cross-tool band.

## PyRadiomics corroborates, feature for feature

mirp is not the oracle the registry named — ten of the sixteen rows claimed `pyradiomics`. Replacing
a claimed oracle without running it would leave the question of whether the two agree unanswered, so
PyRadiomics 3.0.1 was run on the same four slices, at the same recipe (`binWidth=1`, which is
identity binning on this integer image, `force2D=True`, `force2Ddimension=0`, one slice per call):

| | worst residual |
|---|---|
| PyRadiomics vs the mirp goldens pinned here (16 means + 64 per-slice) | **7.0e-16** |
| PyRadiomics vs Nyxus, fifteen features | **2.6e-16** |
| PyRadiomics vs Nyxus, `GLSZM_ZE` | **2.5e-3** |

The two reference implementations are the same to floating-point summation order. That does three
things: it confirms the registry's original `pyradiomics` claim was correct in substance and only
ever lacked a file; it means changing the pinned oracle to mirp loses nothing; and it rules out a
mirp convention artifact as the explanation for the `GLSZM_ZE` gap below — **two independent tools
and the published consensus all agree with each other and disagree with Nyxus by the same amount in
the same direction.**

Only mirp's values are pinned. A second pinned table of numbers identical to the first to 7e-16
would be redundancy, not coverage; the corroboration is recorded here and reproducible from
`glszm_2d_golden_regen.md`.

## The one residual is `fast_log10`, not GLSZM

`glszm.cpp` computes zone entropy as

```cpp
double entrTerm = fast_log10(p / sum_p + EPS) / LOG10_2;
f_ZE += p / sum_p * entrTerm;
```

`Nyxus::fast_log10` (`src/nyx/helpers/helpers.h`) is a **float-precision quadratic approximation** of
the logarithm, not `std::log10`. That is the whole of the residual:

| slice | Nyxus | mirp | PyRadiomics | rel |
|---|---|---|---|---|
| z1 | 2.3185803890228271 | 2.321928094887362 | 2.321928094887361 | 1.4e-3 |
| z2 | 2.2318551199776784 | 2.2359263506290326 | 2.2359263506290312 | 1.8e-3 |
| z3 | 1.5809745788574219 | 1.5849625007211563 | 1.5849625007211552 | 2.5e-3 |
| z4 | 1.5809745788574219 | 1.5849625007211563 | 1.5849625007211552 | 2.5e-3 |

Slices z3 and z4 are the tell: both references put zone entropy at exactly `log2(3) = 1.5849625007`,
which is what a three-zone ROI with equal probabilities must give. Nyxus returns 1.5809745789. The
exact value being missed is a plain logarithm of a small integer, which is as clean a fingerprint of
an approximate `log` as this fixture can produce.

The normalisation is correct here — the probability is `p / sum_p`, not a raw count — so the only
difference from the reference is the logarithm itself. `GLSZM_ZE` therefore asserts at `rel=4e-3`
with that stated cause, and the other fifteen features stay at `rel=1e-9`. The band is a statement
about the approximation and nothing else; it is deliberately too tight to absorb anything larger.

The same approximation is live in `glcm.cpp`, `glrlm.cpp`, `3d_glcm.cpp`, `3d_gldm.cpp`,
`3d_glrlm.cpp` and `3d_glszm.cpp`. `gldm.cpp` was switched to `std::log` when the 2D GLDM family was
re-vetted, so the codebase is currently mixed. Changing it here would change a public feature value,
which the project handles on its own branch; this PR measures it and states the band.

## Per slice, not just the mean

The mean over the four phantom slices is what IBSI publishes, but it cannot vet the four values
behind it: two slice errors that cancel leave it unmoved, and a defect confined to one slice reaches
it quartered. `test_2d_glszm_mirp.h` therefore pins both tables and asserts on both.

**Negative control.** Adding +0.02 to the pinned `GLSZM_SAE_z1` and −0.02 to `GLSZM_SAE_z2` leaves
the mean untouched. `TEST_2D_GLSZM_SAE_MIRP` fails and names the element —

```
abs of (actual=0.105556 - groundtruth=0.125556)=0.02 > tolerance=1.25556e-10
  ... GLSZM_SAE_z1
```

— while `TEST_2D_GLSZM_SAE_IBSI`, which checks only the mean, passes. That is the assertion shape
working, demonstrated rather than asserted.

## Range, identity and cross-table checks on the pinned goldens

Run mechanically over **all 112 pins in the family** — the 16 mirp means, the 64 mirp per-slice
values, the 16 IBSI values and the 16 default-mode regression pins — not spot-checked:

- **Range.** `GLSZM_SAE`, `GLSZM_ZP`, `GLSZM_GLNN`, `GLSZM_SZNN` in (0, 1]; `GLSZM_LAE` ≥ 1;
  `GLSZM_GLV`, `GLSZM_ZV`, `GLSZM_ZE` ≥ 0.
- **Identity.** `GLSZM_GLNN` ≤ `GLSZM_GLN`, `GLSZM_SZNN` ≤ `GLSZM_SZN`, `GLSZM_LGLZE` ≤
  `GLSZM_HGLZE`, `GLSZM_SAE` ≤ `GLSZM_LAE`.
- **Digit count.** No entry in the `_ibsi` table carries more than three significant figures.
- **Cross-table.** Each IBSI pin agrees with the corresponding mirp mean within `rel=1e-2`, the
  precision IBSI publishes at; and each mirp mean equals the arithmetic mean of its own four
  per-slice pins within `rel=1e-12`. That second one is the check that a per-slice table and a mean
  table cannot drift apart in a copy-paste.

All pass. These catch a rotted pin instantly; they cannot catch a wrong definition — the pre-fix 2D
GLDZM values passed every check of this kind — so they are a floor, not the vetting.

## What the registry rows themselves got wrong

Beyond the dangling `target_test`, the sixteen 2D rows carried two errors worth naming, both now
corrected:

- **`current_test` listed two 3D files on every 2D row** — `test_3d_glszm_regression.h` and
  `test_3d_glszm_pyradiomics.h` — neither of which asserts a single 2D feature. A 2D row pointing at
  a 3D file is not a near-miss; it means the row's evidence was never checked against the file
  claimed to hold it.
- **`test_3d_glszm_regression.h` is orphaned** — no `#include` in `test_all.cc`, so none of it ever
  runs. That was recorded in the 2D rows' notes, where it did not belong and where correcting the
  rows would have silently erased it. It is a **3D** GLSZM problem; the 3D rows still carry the note
  and it is called out here so the 3D family re-vet inherits it rather than rediscovering it.

## Also corrected: the default-mode regression pins

`test_2d_glszm_regression.h` guards a different config point — Nyxus' default mode, `IBSI=false` at
64 grey levels, which weights the grey-level-dependent features by raw intensity and matches no
reference (`GLSZM_HGLZE` is 16.44 in IBSI mode and 1497.57 here on the same fixture). Nothing vets
it, so a drift guard is the right kind of test for it.

It was asserting at ±1%, and one pin had drifted underneath that band:

| feature | pinned | computed now | rel |
|---|---|---|---|
| GLSZM_SALGLE | 0.109386 | 0.10956092298348039 | **1.6e-3** |

Every other entry was a truncation of the current value to 1e-5 or better. A ±1% band on a
self-recorded snapshot is wide enough to stop being a drift guard, which is what happened here. The
pins are re-recorded at full precision and the band tightened to `agrees_gt`'s `rel=1e-3` default.
Which of the two `GLSZM_SALGLE` values is "right" is not a question this recipe can answer — nothing
vets default mode — so the fact recorded is only that the value moved and the test did not notice.

## Reproduction

```
conda activate nyxus_mirp          # mirp 2.6.0
python tests/vetting/oracles/gen_glszm_mirp.py
```

Nyxus side: build `runAllTests` with `-DRUN_GTEST=ON` and run
`--gtest_filter=*GLSZM*MIRP*`. Regenerating the goldens from scratch:
`glszm_2d_golden_regen.md`.
