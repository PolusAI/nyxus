# 3D GLDM vs PyRadiomics — vetting report

Recipe `gldm3d.pyradiomics_bincount20` on `bench_compat_liver_3d`
(`tests/data/nifti/compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, label 1).

| | |
|---|---|
| tool | pyradiomics **v3.0.1**, SimpleITK 2.3.1, numpy 1.23.5, Python 3.8.20 (conda env `nyxus_oracle`) |
| tool settings | `binCount: 20`, `interpolator: sitkBSpline`, `resampledPixelSpacing:` empty, `weightingNorm:` empty, `imageType: Original`; class defaults `distances=[1]`, `gldm_a=0` |
| Nyxus settings | `GREYDEPTH=100` (inert), `IBSI=false`, `GLDM_GREYDEPTH=-20` |
| generator | `tests/vetting/oracles/gen_gldm3d_pyradiomics.py` |
| assertions | `tests/test_3d_gldm_pyradiomics.h`, `tests/python/test_nyxus.py::test_3d_gldm_compatibility` |
| ROI as extracted | cube 30×87×4, intensity range [212, 653] |

## Verdict: all fourteen vetted, at SPEC §7's exact tier

Nothing was demoted. Every pinned golden reproduces from a fresh PyRadiomics run at `rel = 0`, so the
goldens were genuinely that tool's output rather than a snapshot under an oracle name. What did not
hold up was the **band** (`±10%`), the **orphaned snapshot file**, and two defects inside it.

## The measurement

Nyxus against PyRadiomics, both at the recipe above. Nyxus values printed by
`TEST_3D_GLDM_DUMP_PYRADIOMICS` at `setprecision(17)`.

| feature | Nyxus | PyRadiomics | abs | rel |
|---|---|---|---|---|
| `3GLDM_DE` | 6.6031219389041276 | 6.6048731874541904 | **1.7512e-3** | **2.65e-4** |
| `3GLDM_DN` | 620.28166666666664 | 620.28166666666664 | 0 | 0 |
| `3GLDM_DNN` | 0.12922534722222223 | 0.12922534722222223 | 0 | 0 |
| `3GLDM_DV` | 5.4254789930555578 | 5.4254789930555560 | 1.78e-15 | 3.27e-16 |
| `3GLDM_GLN` | 481.78125 | 481.78125 | 0 | 0 |
| `3GLDM_GLV` | 8.7284944010416616 | 8.7284944010416670 | 5.33e-15 | 6.11e-16 |
| `3GLDM_HGLE` | 129.87979166666668 | 129.87979166666668 | 0 | 0 |
| `3GLDM_LDE` | 24.279166666666665 | 24.279166666666665 | 0 | 0 |
| `3GLDM_LDHGLE` | 3061.1764583333334 | 3061.1764583333334 | 0 | 0 |
| `3GLDM_LDLGLE` | 0.25264958487679379 | 0.25264958487679401 | 2.22e-16 | **8.79e-16** |
| `3GLDM_LGLE` | 0.012371308742463947 | 0.012371308742463947 | 0 | 0 |
| `3GLDM_SDE` | 0.163503551425667 | 0.16350355142566711 | 1.11e-16 | 6.79e-16 |
| `3GLDM_SDHGLE` | 21.958648461266709 | 21.958648461266701 | **7.11e-15** | 3.24e-16 |
| `3GLDM_SDLGLE` | 0.0024445083605478196 | 0.0024445083605478196 | 0 | 0 |

**Eight of the fourteen agree to the last bit.** Across the thirteen that take no logarithm the worst
residual is **7.11e-15 absolute** and **8.79e-16 relative** — float summation order and nothing else.

## Why the exact tier is honest here

SPEC §7 gives the exact tier for "no estimator disagreement", and there genuinely is none. Three
things could have separated the two implementations, and all three are asserted rather than assumed:

- **Binning.** `GLDM_GREYDEPTH=-20` selects radiomics bin-count binning; PyRadiomics' `binCount=20`
  is the same scheme. Both produce twenty levels, of which the ROI occupies twenty (`Ng = 20`).
- **Neighbourhood and cutoff.** PyRadiomics defaults to `distances=[1]`, `gldm_a=0`. Nyxus fixes the
  same thing in code: a 26-entry `shifts[]` table and `pi == neig_pi`. Neither side exposes a knob
  the other lacks, which is also why `matrix/gldm3d.md` records no distance or alpha axis.
- **The dependence offset.** PyRadiomics places a voxel with no dependent neighbour in column `j=1`;
  Nyxus starts its counter at `nd = 1`. The two therefore index the same column, with no shift.

The band is `abs=1e-9`, asserted with `ASSERT_NEAR` — six orders inside the worst measurement.

## `3GLDM_DE` is banded, not reported

The family's only sum over logarithms. Nyxus takes it through `fast_log10()`, a float-precision
`log10` approximation divided by a ten-digit `LOG10_2`; PyRadiomics uses `numpy.log2`. Measured
residual **1.7512e-3 absolute / 2.65e-4 relative**. Band `abs=4e-3` — the measurement doubled and
rounded up to one significant figure, i.e. 2.3× the residual.

This is this family's own figure. The 2D twin's identical `calc_DE()` measures **1.3e-3 relative**,
roughly five times wider; carrying that number across would have produced a band far looser than
anything measured here. `fast_log10` is a deliberate fast path and its error is a convention of this
codebase, so it belongs in the band rather than in a defect report.

## The matrix under the scalars

All fourteen features are contractions of one dependence matrix `P(i, j)`, so a compensating pair of
errors inside it survives every scalar assertion. The matrix is therefore pinned as well.

| | value |
|---|---|
| non-empty cells | **163** |
| `Ng` (grey levels present) | 20 |
| `Nd` (largest dependence) | 15 |
| `Nz` (dependence zones) | 4800 |
| `Np` (ROI voxels) | 4800 |

**`Nz == Np` is asserted, not noted.** PyRadiomics allows incomplete zones, so every ROI voxel owns
exactly one dependence zone. Nyxus reaches the same number by a different route — it skips
background-valued voxels of the ROI cube and counts the rest — and `Np` is counted off the cube
rather than read out of the matrix, so the two agreeing is evidence that Nyxus pulls no voxel from
outside the ROI into the matrix.

**Both PyRadiomics axes are compacted.** `_calculateMatrix()` deletes the rows of grey levels absent
from the ROI *and* the columns of dependences no voxel has, keeping the survivors in `ivector` and
`jvector`. Reading either axis off its index would relabel cells silently. The pinned cells are keyed
by the values themselves, so the two representations line up however either side indexes.

**Three independent constructions must agree.** The generator compares (a) PyRadiomics' own `P_gldm`
from its C extension, (b) the same matrix rebuilt from the definition in plain numpy — 26 offsets,
in-mask, equal-to-centre, count starting at 1 — and (c) the fourteen published scalars recomputed
from the pinned cells. All 163 cells and all 14 features agree. (b) is what turns the neighbourhood,
the cutoff and the offset into checked facts rather than comments; (c) is what makes it impossible to
edit one table and not the other.

### The small volume

`TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS` runs the same assertion on a 4×4×3 volume with two
**identical** populated slices above an empty one, `Ng=4`, `Nd=6`, `Nz=32`, 9 non-empty cells. Because
the slices are identical, every voxel sees its in-slice matches twice — once in its own slice, once
among the eight diagonal neighbours of the other — plus itself and its vertical partner. Every
dependence is therefore `2·(in-slice matches) + 2`, and **every observed value is even**. A
neighbourhood missing the z offsets would halve them; one missing the two strictly-vertical offsets
would make them odd. That is the direct check on the vertical half of the 26-neighbourhood.

## Range and identity checks

Run mechanically over the whole family by the generator, all gating its exit code.

- **12 bounds, all in range.** Ten features are a weighted mean of a quantity bounded on one side
  (`i, j ≥ 1`, so a mean of `1/i²`, `1/j²` or `1/(i²j²)` cannot exceed 1 and a mean of their
  reciprocals cannot fall below it). `GLN` and `DN` are sums of squared marginals over `Nz` and so
  cannot exceed `Nz`; `DE` cannot exceed `log₂` of the number of non-empty cells (7.35 against a
  measured 6.605). Three of the twelve are computed from the run rather than typed.
- **8 identities, all hold.** `DNN == DN / Nz`; `Nz == Np`; and six inequalities over the *joint*
  distribution — `SDE ≤ LDE`, `LGLE ≤ HGLE`, `SDLGLE ≤ SDE`, `SDLGLE ≤ LGLE`, `LDHGLE ≥ LDE`,
  `LDHGLE ≥ HGLE`. The joint ones are the ones that can catch a transposed or mis-weighted axis,
  which a per-marginal check cannot.

## Negative controls

Every new assertion shape, both halves — a perturbed **golden** proves an assertion compares; only a
perturbed **input** proves it compares the run under test. Each control was planted, built and run in
isolation, then reverted.

| # | perturbation | target | result |
|---|---|---|---|
| A | one scalar golden moved by 1e-4 (`3GLDM_SDE`) | `TEST_3D_GLDM_SDE_PYRADIOMICS` | fails |
| B | one matrix cell count 2 → 3 | `TEST_3D_GLDM_MATRIX_PYRADIOMICS` | fails |
| C | **one voxel of the binned cube overwritten** | `TEST_3D_GLDM_MATRIX_PYRADIOMICS` | fails |
| D | **binning changed to binCount 19** | `TEST_3D_GLDM_MATRIX_PYRADIOMICS` | fails |
| E | **one voxel of the 4×4×3 volume 1 → 2** | `TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS` | fails |
| F | `3GLDM_DE` band tightened to the exact tier | `TEST_3D_GLDM_DE_PYRADIOMICS` | fails |
| G | one snapshot pin moved by 1e-6 relative | `TEST_3D_GLDM_LGLE_REGRESSION` | fails |

C, D and E are the input half: they demonstrate the matrix assertions read the cube the featurisation
produced rather than a hand-written copy of it. F demonstrates the `3GLDM_DE` band is load-bearing —
at `abs=1e-9` the `fast_log10` residual fails, so the band is neither decorative nor over-wide.

Generator-side controls: removing the header's matrix table makes `gen_gldm3d_pyradiomics.py` exit 1;
so does any bound violation, identity failure, cross-table mismatch, definition mismatch or unpinned
feature, each through its own counter.

## What this report does not cover

- **One oracle.** IBSI calls this family NGLDM and `mirp` exposes one, but Nyxus ships 3D GLDM and 3D
  NGLDM as *separate* families with different conventions, so a `mirp` run would need its own
  reconciliation before it could corroborate anything here.
- **`GLDM_GREYDEPTH = 0` / `IBSI=true` are unmeasured.** They are the same code path reached two
  ways, reachable in production, and pinned nowhere. Recorded in `matrix/gldm3d.md`.
- **The matrix assertion re-derives the matrix rather than reading `calculate()`'s.** `P`, `I`, `Ng`,
  `Nd` and `Nz` are private and exposing them would be a source change. So the assertion establishes
  that *the cube Nyxus loaded produces PyRadiomics' matrix*; what ties that to `calculate()`'s own
  matrix is the fourteen scalars agreeing to 7.11e-15.
