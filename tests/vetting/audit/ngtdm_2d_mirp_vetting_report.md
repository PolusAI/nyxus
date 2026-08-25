# 2D NGTDM vs mirp — vetting report

All five 2D NGTDM rows have claimed `oracle=pyradiomics` with
`target_test=test_2d_ngtdm_pyradiomics.h` since the tracker was imported — **a file that does not
exist** — with an **empty `tolerance` column** and a prose sentence where the `config_recipe` id
belongs. The only test covering them asserted the published consensus at ±1%.

Running independent oracles for the first time found Nyxus reproducing **both** mirp and PyRadiomics
on every feature and every slice. Nothing is promoted or demoted; what changes is that the claim is
now backed, measured, and pinned at the precision the agreement actually supports.

## Tool and configuration

| | |
|---|---|
| Tool | mirp 2.6.0, numpy 2.4.6, env `nyxus_mirp` (conda) |
| Corroborating run | PyRadiomics 3.0.1, env `nyxus_oracle` (conda) |
| Generator | `tests/vetting/oracles/gen_ngtdm_mirp.py` |
| Recipe | `ngtdm.ibsi_phantom_2d` |
| Fixture | the four IBSI digital-phantom slices, read out of `tests/test_data.h` by `oracles/ibsi_phantom.py` |
| Nyxus config | `IBSI=true`, `GREYDEPTH=128`, each slice featurised on its own |
| mirp config | `by_slice=True`, `base_discretisation_method="none"` |
| PyRadiomics config | `binWidth=1` (identity binning on an integer image), `force2D=True`, `force2Ddimension=0` |
| Test | `test_2d_ngtdm_mirp.h` |
| Tolerance | `abs=1e-9` (SPEC §7 exact tier, which is an absolute band) |

```
python tests/vetting/oracles/gen_ngtdm_mirp.py
```

Verifies every golden pinned in the header — both tables, 5 means and 20 per-slice values — reports
any value it produces that the header pins nothing for, and exits non-zero on failure. Current run:
**25 verified, 0 failed, 0 unproducible, 0 unpinned**, every one at `rel = 0`.

## Result

Three implementations, worst residual across all five features and all four slices:

| comparison | worst residual |
|---|---|
| Nyxus vs mirp | **3.2e-16** relative, **3.6e-15** absolute |
| Nyxus vs PyRadiomics | **2.4e-16** relative |
| mirp vs PyRadiomics | **1.6e-16** relative |
| Nyxus vs the published IBSI consensus (3 s.f.) | 4.1e-3, on `NGTDM_COARSENESS` |

The values **agree**; they are not bit-identical. Eight of the 25 comparisons carry a non-zero
residual, the largest of them 3.6e-15 absolute on `NGTDM_COMPLEXITY` slice 2 and 3.2e-16 relative on
`NGTDM_CONTRAST` slice 2. SPEC §7's exact tier is an **absolute** 1e-9 band, so that is what
`test_2d_ngtdm_mirp.h` asserts (`ASSERT_NEAR`) — six orders of magnitude of headroom over the worst
measured residual, and the same tier the 2D GLSZM and GLDZM families were pinned at.

Every value, all three implementations — 5 features x (4 slices + the mean):

| feature | slice | Nyxus | mirp | PyRadiomics | rel (Nyxus vs mirp) |
|---|---|---|---|---|---|
| `NGTDM_COARSENESS` | z1 | 0.1003260596940055 | 0.1003260596940055 | 0.1003260596940055 | 0 |
| `NGTDM_COARSENESS` | z2 | 0.1063128234847425 | 0.1063128234847425 | 0.1063128234847425 | 0 |
| `NGTDM_COARSENESS` | z3 | 0.13481363996827916 | 0.13481363996827916 | 0.13481363996827916 | 0 |
| `NGTDM_COARSENESS` | z4 | 0.14058969566794052 | 0.14058969566794052 | 0.14058969566794052 | 0 |
| `NGTDM_COARSENESS` | **mean** | 0.12051055470374192 | 0.12051055470374192 | 0.12051055470374192 | 0 |
| `NGTDM_CONTRAST` | z1 | 1.6362843750000002 | 1.6362843750000002 | 1.636284375 | 0 |
| `NGTDM_CONTRAST` | z2 | **0.6917298440005832** | **0.691729844000583** | 0.6917298440005831 | **3.2e-16** |
| `NGTDM_CONTRAST` | z3 | **0.7298857259166642** | **0.7298857259166641** | 0.7298857259166641 | **1.5e-16** |
| `NGTDM_CONTRAST` | z4 | 0.6431521082369848 | 0.6431521082369848 | 0.6431521082369848 | 0 |
| `NGTDM_CONTRAST` | **mean** | 0.9252630132885581 | 0.9252630132885581 | 0.925263013288558 | 0 |
| `NGTDM_BUSYNESS` | z1 | **2.166847826086957** | **2.1668478260869564** | 2.1668478260869564 | **2e-16** |
| `NGTDM_BUSYNESS` | z2 | 2.2912545787545793 | 2.2912545787545793 | 2.2912545787545793 | 0 |
| `NGTDM_BUSYNESS` | z3 | 3.940625 | 3.940625 | 3.940625 | 0 |
| `NGTDM_BUSYNESS` | z4 | 3.556448412698413 | 3.556448412698413 | 3.556448412698413 | 0 |
| `NGTDM_BUSYNESS` | **mean** | 2.9887939543849873 | 2.9887939543849873 | 2.9887939543849873 | 0 |
| `NGTDM_COMPLEXITY` | z1 | 10.750942513368985 | 10.750942513368985 | 10.750942513368985 | 0 |
| `NGTDM_COMPLEXITY` | z2 | **14.882002533807045** | **14.882002533807048** | 14.882002533807048 | **2.4e-16** |
| `NGTDM_COMPLEXITY` | z3 | 8.37328431372549 | 8.37328431372549 | 8.37328431372549 | 0 |
| `NGTDM_COMPLEXITY` | z4 | **7.594298066448802** | **7.594298066448804** | 7.594298066448804 | **2.3e-16** |
| `NGTDM_COMPLEXITY` | **mean** | **10.40013185683758** | **10.400131856837582** | 10.400131856837582 | **1.7e-16** |
| `NGTDM_STRENGTH` | z1 | **1.7958446251129179** | **1.7958446251129176** | 1.7958446251129176 | **1.2e-16** |
| `NGTDM_STRENGTH` | z2 | 2.5242791143458305 | 2.5242791143458305 | 2.5242791143458305 | 0 |
| `NGTDM_STRENGTH` | z3 | 3.5422771781859765 | 3.5422771781859765 | 3.5422771781859765 | 0 |
| `NGTDM_STRENGTH` | z4 | **3.6430627518710414** | **3.6430627518710423** | 3.6430627518710423 | **2.4e-16** |
| `NGTDM_STRENGTH` | **mean** | 2.8763659173789415 | 2.8763659173789415 | 2.8763659173789415 | 0 |

Everything is floating-point summation order. This family has no entropy term, so the `fast_log10`
approximation that costs 2D GLDM and 2D GLSZM their exact tier does not arise here.

**PyRadiomics — the oracle the rows actually named — was run rather than quietly replaced.** It
agrees with mirp to 1.6e-16, so the original registry claim was correct in substance and only ever
lacked a file, and swapping the pinned oracle to mirp loses nothing. Only mirp's values are pinned: a
second table identical to the first to 1.6e-16 is redundancy, not coverage.

## Per slice, not just the mean

The mean over the four phantom slices is what IBSI publishes, but it cannot vet the four values
behind it. `test_2d_ngtdm_mirp.h` pins both tables and asserts on both, and the mean it checks is
averaged from the per-slice vector it has just asserted rather than from a second featurisation of
the same four slices — so the two assertions cover one computation, not two that happen to agree.

**Negative control.** Adding +0.01 to the pinned `NGTDM_CONTRAST_z1` and −0.01 to `_z2` leaves the
mean untouched. `TEST_2D_NGTDM_CONTRAST_MIRP` fails and names the element —

```
The difference between per_slice[z] and ngtdm_2d_mirp_slice_ref_vals.at(key) is 0.01,
which exceeds ngtdm_2d_mirp_abs_tolerance, where ... ngtdm_2d_mirp_abs_tolerance evaluates to 1e-09.
  ... NGTDM_CONTRAST_z1
```

— while `TEST_2D_NGTDM_CONTRAST_IBSI`, which checks only the mean, passes. The generator also fails,
reporting `23 verified, 2 failed`.

## The static nobody was tracking

`NGTDMFeature::n_levels` is a `static int` on the feature class, shared by every test in the binary
and by every caller in a process. `test_2d_ngtdm_regression.h` set it to **100** and never restored
it; no other NGTDM test set it at all.

That is **not** a live defect, and the reason is worth stating precisely: `ngtdm.cpp` forces the
grey-binning info to 0 whenever IBSI compliance is on, so in IBSI mode the static is ignored
entirely. Measured rather than assumed — the IBSI-mode values are **bit-identical** with the static
at 0 and at 100.

Outside IBSI mode it is decisive:

| config | `NGTDM_CONTRAST`, four-slice mean |
|---|---|
| IBSI mode | 0.9252630 |
| default mode, `n_levels = 100` | 3169.9291 |
| default mode, `n_levels = 0` | 6634.5048 |

So a future non-IBSI NGTDM test placed after the regression one would silently inherit a grey count
nobody chose for it, and the values would be off by a factor of two with nothing to say why.

Two changes close it. The shared fixture takes `n_levels` as a **parameter** and restores whatever it
found, so no test can leak it; and `test_2d_ngtdm_mechanics.h` asserts both properties — that IBSI
mode ignores the static, bit-exactly, and that the fixture restores it.

`GLRLMFeature::n_levels` is assigned exactly the same way in `test_2d_glrlm_regression.h:62` and
should get the same treatment when that family is next touched.

## What else the registry rows got wrong

- **`tolerance` was empty on all five rows.** Nothing recorded what was being claimed. Now
  `abs=1e-9`, SPEC §7's exact tier, on the vetted rows and `rel=1e-3` on the regression ones.
- **`config_recipe` held a prose sentence**, not a recipe id. Now `ngtdm.ibsi_phantom_2d`, defined in
  `config_recipes.md`.
- **One row described two configurations.** The five rows named the IBSI recipe while their
  `current_test` also listed `test_2d_ngtdm_regression.h`, which runs `IBSI=false` at
  `NGTDMFeature::n_levels=100` — a different quantity on the same fixture — and
  `test_2d_ngtdm_mechanics.h`, which checks setting behaviour rather than a reference value. The
  registry is one row per assertion (SPEC §3), so the default-mode snapshots now have their own
  recipe, `ngtdm.default_fbn100`, and their own five `status=regression` rows; the mechanics file is
  cited by neither, because a mechanics test establishes no vetting and pins no reference value.
- **`test_name` and `benchmark` were empty.** Every row now names the exact gtest cases that back it
  and the `ibsi_digital_phantom` benchmark it runs on. The family's config matrix is
  `matrix/ngtdm.md`.
- **`current_test` listed two 3D files on every 2D row** — `test_3d_ngtdm_regression.h` and
  `test_3d_ngtdm_pyradiomics.h` — neither of which asserts a single 2D feature.
- **`test_3d_ngtdm_regression.h` is orphaned** (no `#include` in `test_all.cc`, so none of it runs).
  That was recorded in the **2D** rows' notes, where correcting the rows would have erased it. It is
  a 3D NGTDM problem; the 3D rows still carry it and it is restated here so the 3D re-vet inherits it.

## A setting that looks meaningful and is not

The previous tests set `PIXELDISTANCE=5`. `ngtdm.cpp` never reads it: the neighbourhood is fixed at
the d=1 8-neighbourhood the IBSI definition uses, which is why the values match the published
consensus at all. The recipe therefore does not set a pixel distance, and says why.

## Reproduction

```
conda activate nyxus_mirp          # mirp 2.6.0
python tests/vetting/oracles/gen_ngtdm_mirp.py
```

Nyxus side: build `runAllTests` with `-DRUN_GTEST=ON` and run `--gtest_filter=*NGTDM*`. Regenerating
the goldens from scratch, including the PyRadiomics corroboration: `ngtdm_2d_golden_regen.md`.
