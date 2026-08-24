# 2D GLDZM vs mirp — vetting report

Seventeen of the family's eighteen rows have claimed `oracle=mirp` with
`target_test=test_2d_gldzm_mirp.h` since the tracker was imported — **a file that did not exist**.
Running that oracle for the first time found Nyxus disagreeing with it on **every one of the 18
features**, by 1.3% to 60%, and the only test covering them passing anyway because its tolerance was
±50%.

## Tool and configuration

| | |
|---|---|
| Tool | mirp 2.6.0, numpy 2.4.6, env `nyxus_mirp` (conda) |
| Generator | `tests/vetting/oracles/gen_gldzm_mirp.py` |
| Recipe | `gldzm.ibsi_phantom_2d` |
| Fixture | the four IBSI digital-phantom slices, read out of `tests/test_data.h` by `oracles/ibsi_phantom.py` |
| Nyxus config | `IBSI=true`, `GREYDEPTH=128`, each slice featurised on its own |
| mirp config | `by_slice=True`, `base_discretisation_method="none"` |
| Test | `test_2d_gldzm_mirp.h` |
| Tolerance | SPEC §7 exact tier, i.e. an **absolute** 1e-9 band (`ASSERT_NEAR`) |

```
python tests/vetting/oracles/gen_gldzm_mirp.py
```

Verifies every golden pinned in the header — both tables, 16 means and 64 per-slice values — and
exits non-zero on mismatch. Current run: **80 verified, 0 failed, 0 unproducible**, every one at
`rel = 0`.

GLDZM takes no distance or coarseness parameter on either side: the zone distance is a property of
the ROI mask rather than a setting, which is why this recipe is shorter than the NGLDM one.

## What the first run showed

Nyxus against mirp, before the fix below. Every feature disagreed:

| feature | Nyxus (before) | mirp | rel |
|---|---|---|---|
| GLDZM_ZDV | 0.081632653061224497 | 0.05102040816326531 | **0.60** |
| GLDZM_LDLGLE | 0.50041335978835977 | 0.38607804232804227 | **0.30** |
| GLDZM_GLNU | 1.6428571428571428 | 1.4142857142857144 | **0.16** |
| GLDZM_ZP | 0.26538957688338494 | 0.24038957688338494 | **0.10** |
| GLDZM_LDE | 1.3214285714285714 | 1.2142857142857144 | 0.088 |
| GLDZM_ZDNUN | 0.83673469387755106 | 0.8979591836734694 | 0.068 |
| GLDZM_SDLGLE | 0.34788359788359785 | 0.3674768518518518 | 0.053 |
| GLDZM_ZDE | 1.8120555863121122 | 1.7319448617396767 | 0.046 |
| GLDZM_SDE | 0.91964285714285721 | 0.9464285714285714 | 0.028 |
| GLDZM_SDHGLE | 14.815476190476193 | 15.235119047619047 | 0.028 |
| GLDZM_HGLZE | 16.047619047619047 | 16.44047619047619 | 0.024 |
| GLDZM_GLNUN | 0.32993197278911562 | 0.3229931972789115 | 0.021 |
| GLDZM_GLV | 3.8866213151927438 | 3.9694784580498865 | 0.021 |
| GLDZM_LGLZE | 0.37838955026455023 | 0.3711970899470899 | 0.019 |
| GLDZM_ZDNU | 3.8571428571428572 | 3.7857142857142856 | 0.019 |
| GLDZM_LDHGLE | 20.976190476190478 | 21.261904761904763 | 0.013 |

**The published IBSI values side with mirp.** `test_2d_gldzm_ibsi.h` pins `GLDZM_SDE` at the IBSI
reference manual's `0.946`; mirp gives 0.94643 and Nyxus gave 0.91964, 2.8% away. That holds for all
fourteen published entries, so this is not a mirp configuration artifact: two independent references
agreed with each other and disagreed with Nyxus.

## Every disagreement lived in one slice

Per slice, the picture was sharper than the averages: **slices z2, z3 and z4 reproduced mirp exactly,
digit for digit, and every discrepancy was in z1.** z1 is the only phantom slice whose ROI is a solid
rectangle with no holes — and the only one containing two same-level pixels that touch at a corner
and nowhere else.

That is what pointed at the cause, and it is the reason this file pins per-slice values as well as
means: the four-slice mean *diluted* a defect that lived entirely in one quarter of the fixture.

**Verified non-vacuous.** Moving `GLDZM_HGLZE_z1` by +0.002 and `GLDZM_HGLZE_z3` by -0.002 leaves the
four-slice mean unchanged to its last digit. `TEST_2D_GLDZM_HGLZE_MIRP` fails and names
`GLDZM_HGLZE_z1`; `TEST_2D_GLDZM_HGLZE_IBSI`, which checks only the mean, passes. The per-slice
table is therefore doing work no averaged assertion in this family can do.

## Cause — GLDZM zones were 4-connected

`GLDZMFeature::calculate` grew a zone by testing four neighbours — East, South, West, North. A GLDZM
zone is 8-connected: two same-level pixels touching only at a corner are one zone. That is the
connectivity IBSI defines, the one mirp implements, and the one **Nyxus' own GLSZM already used**
(`glszm.cpp` tests E/SE/S/SW, the forward half of the 8-neighbourhood). The two families disagreed
with each other inside the same codebase about what a zone is.

On phantom slice z1 the difference is two diagonal links, which split five true zones into seven:

| | zones on z1 | GLDZM_ZP (Ns/Nv) | GLDZM_GLNU |
|---|---|---|---|
| 4-connected (before) | 7 | 0.35 | 2.714 |
| 8-connected (after) | 5 | 0.25 | 1.8 |
| mirp | 5 | 0.25 | 1.8 |

Adding the four diagonal directions makes Nyxus reproduce mirp on all 16 mirp-exposed features, on
every slice, to a worst **absolute** residual of **1.3e-15** (`GLDZM_ZDE`, slice 2; worst relative
7.0e-16 on the same feature, slices 3 and 4) — and match all fourteen published IBSI values.

`3d_gldzm.cpp` carries the identical four-direction loop, so **3D GLDZM has the same defect**. It is
a separate family in the re-vetting order and its own PR's business; changing it here would move
pinned goldens this PR does not own.

## The matrix test was encoding the bug

Fixing the source broke `TEST_2D_GLDZM_MATRIX_CORRECTNESS_IBSI`, which compares Nyxus' GLDZ-matrix
against `ibsi_fig3_17c_gldzm_reference_matrix` in `test_data.h`. That reference was itself wrong,
independently of the fix: it summed **9 zones over a 16-pixel ROI**, and mirp finds 8.

mirp on that exact fixture pins the matrix three ways over, and all three agree with the 8-connected
reading and none with the pinned one:

| mirp on fig. 3.17a | value | implies |
|---|---|---|
| `dzm_z_perc_2d` | 0.5 | 8 zones over 16 pixels |
| `dzm_glnu_2d` | 2.25 | per-grey zone counts 3, 1, 2, 2 (sum of squares 18) |
| `dzm_zdnu_2d` | 6.25 | 7 zones at distance 1, one at distance 2 (sum of squares 50) |

The grey-level-2 row read `2, 0` and is `1, 0`: its five pixels are one zone, linked through the
(2,1)–(3,2) diagonal. Corrected at the definition site with that arithmetic recorded beside it.

So the family's only assertion against a published *matrix* had the bug as its expected answer, and
its only assertion against published *values* could not see a 30% error through its band. Neither
could fail.

## The ±50% tolerance

`assert_gldzm_feature_against_golden_values_ibsi()` ended in `agrees_gt(aveTotal, reference, 2.)`.
`agrees_gt` computes `tolerance = ground_truth / frac_tolerance`, so the argument is a divisor and
`2.` is a band of **±50%** — about 40x looser than the worst error it was covering, and 5·10⁷ times
looser than the agreement now measured. It is the same defect the 2D NGLDM re-vet found and fixed;
the sibling family was never swept for it.

The IBSI file now asserts at `rel=1e-2`, set by the three significant figures its values are
published to, and the mirp file at SPEC §7's exact tier — an **absolute** 1e-9 band, which is what
that row of the tolerance table specifies, rather than a relative one of the same magnitude.

## Range and identity checks on the pinned goldens

All 82 pinned values — 16 means, 64 per-slice, 2 regression — were checked against the bounds the
features hold by construction, since a golden outside its own range is the cheapest kind of wrong:

| check | features | result |
|---|---|---|
| in [0, 1] | `SDE`, `ZP`, `GLNUN`, `ZDNUN`, `LGLZE`, `SDLGLE` | pass |
| ≥ 1 | `LDE`, `ZDM` | pass |
| ≥ 0 | `ZDE`, `ZDV`, `GLV`, `GLNU`, `ZDNU` | pass |
| normalised ≤ raw (`GLNUN ≤ GLNU`, `ZDNUN ≤ ZDNU`) | — | pass |
| `LDE ≥ SDE` per slice | — | pass |

No violations. Worth noting the pre-fix values passed these too — a range check catches a rotted
pin, not a wrong definition, which is why it is a complement to running the oracle and not a
substitute.

## Registry corrections

- The 16 mirp-covered rows: `status=vetted`, `oracle=mirp`, `config_recipe=gldzm.ibsi_phantom_2d`,
  `tolerance=abs=1e-9`, `current_test=test_2d_gldzm_ibsi.h;test_2d_gldzm_mirp.h`. Each row's notes
  say which quantity backs it — per slice and the four-slice mean.
- `GLDZM_GLM` and `GLDZM_ZDM` → `status=regression`. Neither is an IBSI GLDZM feature and mirp
  exposes no column for either, so nothing independent covers them; they are drift guards in
  `test_2d_gldzm_regression.h`. Their previous `vetted` verdict rested on a pinned Nyxus value in a
  table named `_ibsi`.
- Every row's `current_test` had named `test_3d_gldzm_regression.h` — a 3D file, on 2D rows, for a
  different implementation. Dropped.
- `GLDZM_ZDV` was the one row reading `oracle=ibsi`, citing the value 0.0816326530612245 as its
  evidence. That number is Nyxus' own pre-fix output; the correct value is 0.05102040816326531. It
  now reads `oracle=mirp`.
