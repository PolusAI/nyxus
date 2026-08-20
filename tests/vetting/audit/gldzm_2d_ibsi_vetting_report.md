# Audit: `test_2d_gldzm_ibsi.h` goldens vs the IBSI digital phantom

Companion to `gldzm_2d_mirp_vetting_report.md`, which carries the family's exact-tier measurement
and the zone-connectivity defect it exposed. This one records what the IBSI file's table actually
held, which entries were never IBSI values, and what the published precision buys.

## Method

The goldens are transcribed from the IBSI reference manual (IBSI Documentation, Release 0.0.1dev,
Dec 13 2021, dataset "dig phantom", aggregation "2D, averaged") and are not produced by a tool, so
there is no generator for this file. Verifying them is a two-step check:

1. **Is each entry actually a published value?** A published consensus value carries three
   significant figures. An entry with sixteen is not a transcription of anything IBSI printed.
2. **What residual does it leave?** Computed against the full-precision values from the mirp run,
   which agree with Nyxus to 8.9e-16 after the connectivity fix.

## Four entries were Nyxus output, not IBSI values

The table mixed fourteen 3-significant-figure published values with four full-precision numbers:

| entry | pinned value | what it is |
|---|---|---|
| `GLDZM_SDLGLE` | 0.34788359788359785 | Nyxus' own pre-fix output, to 17 digits |
| `GLDZM_GLM` | 3.476190476190476 | ditto |
| `GLDZM_ZDM` | 1.1071428571428572 | ditto |
| `GLDZM_ZDV` | 0.0816326530612245 | ditto |

Each matched Nyxus digit for digit before the fix, which is what identifies them: the IBSI manual
does not publish at that precision, and no independent run produces those numbers. Two of the four
are demonstrably wrong — mirp gives `GLDZM_SDLGLE` = 0.3674768518518518 (5.3% away) and `GLDZM_ZDV`
= 0.05102040816326531 (60% away).

This is the SPEC §6.3.1 rule about a table holding one oracle, failing in the way it warns of: a
table named `_ibsi` whose keys are partly snapshots has no honest `<oracle>` to put in its name, and
the fix is to split it rather than to pick the majority.

- `GLDZM_SDLGLE` and `GLDZM_ZDV` move to `test_2d_gldzm_mirp.h`, where they are vetted at the exact
  tier. The IBSI file never held a published value for either.
- `GLDZM_GLM` and `GLDZM_ZDM` move to `test_2d_gldzm_regression.h` as drift guards. They are not
  IBSI GLDZM features and mirp exposes no column for either, so nothing independent covers them; the
  registry rows are demoted to `status=regression` accordingly. Their values are re-recorded after
  the connectivity fix (`GLDZM_GLM` 3.4761904761904758 → 3.5261904761904761, `GLDZM_ZDM`
  1.1071428571428572 → 1.0714285714285714).

## Result table — the fourteen published values

`rel` is the published 3-significant-figure value against the full-precision one.

| feature | IBSI published | full precision | rel |
|---|---|---|---|
| GLDZM_SDE | 0.946 | 0.9464285714285714 | 4.5e-4 |
| GLDZM_LDE | 1.21 | 1.2142857142857144 | **3.5e-3** |
| GLDZM_LGLZE | 0.371 | 0.3711970899470899 | 5.3e-4 |
| GLDZM_HGLZE | 16.4 | 16.44047619047619 | 2.5e-3 |
| GLDZM_SDHGLE | 15.2 | 15.235119047619047 | 2.3e-3 |
| GLDZM_LDLGLE | 0.386 | 0.38607804232804227 | 2.0e-4 |
| GLDZM_LDHGLE | 21.3 | 21.261904761904763 | 1.8e-3 |
| GLDZM_GLNU | 1.41 | 1.4142857142857144 | 3.0e-3 |
| GLDZM_GLNUN | 0.323 | 0.3229931972789115 | 2.1e-5 |
| GLDZM_ZDNU | 3.79 | 3.7857142857142856 | 1.1e-3 |
| GLDZM_ZDNUN | 0.898 | 0.8979591836734694 | 4.5e-5 |
| GLDZM_ZP | 0.24 | 0.24038957688338494 | 1.6e-3 |
| GLDZM_GLV | 3.97 | 3.9694784580498865 | 1.3e-4 |
| GLDZM_ZDE | 1.73 | 1.7319448617396767 | 1.1e-3 |

Worst residual **0.35%** on `GLDZM_LDE` (1.21 published against 1.2142857 computed), which is what
sets the file's `rel=1e-2` tolerance. Every residual here is rounding in the published value, not
disagreement.

**Before the connectivity fix, none of this held.** Nyxus was 0.4% to 30% away from these same
published values — 2.8% on `GLDZM_SDE`, 8.8% on `GLDZM_LDE`, 16% on `GLDZM_GLNU`. The file passed
regardless, because its tolerance was ±50%.

## What the two files are for

This file fixes the **definition** — it is the only assertion in the family tied to a published
consensus rather than to a tool run. `test_2d_gldzm_mirp.h` fixes the **digits**, per slice and
averaged, at `rel=1e-9`. Keeping both is deliberate: the published values are the reason the mirp
comparison can be trusted as more than agreement between two implementations of the same mistake.

## What changed in the file

- The tolerance is `rel=1e-2`, set by the published precision. It was `frac_tolerance=2.`, i.e.
  **±50%** — 40x looser than the worst error it was covering.
- The table is `const`. The `.at()` read behind a `count()` guard is kept from the original.
- The settings block and the four-slice averaging move to `test_2d_gldzm_common.h`, shared with the
  mirp and regression files. The copy here sized every slice's mask with `sizeof` of the intensity
  array; the arrays are all 20 entries, so the counts happened to agree.
- `<unordered_map>` is gone: `test_ref_vals.h`, included two lines below it, already supplies it.
- The `VERIFIABLE_WITH_3P_BUILTIN_ORACLE__` trace prefix is now the oracle token the assertion
  actually compares against.
- `assert_gldzm_matrix_ibsi()` keeps its own fixture and stays in this file: it checks the
  GLDZ-matrix against the manual's figure 3.17 worked example, which is a published reference of a
  different kind. That reference needed correcting too — see the mirp report.
