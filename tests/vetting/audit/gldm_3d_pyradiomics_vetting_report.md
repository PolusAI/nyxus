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

### The same helper costs `3GLDM_DE` a second carve-out, on the snapshot side

Found by CI, not by this pass, and worth separating from the paragraph above: that band is against
*PyRadiomics*. The two **snapshot** tables compare Nyxus to itself, so they were pinned at the
family's `rel=1e-9` drift band like the other thirteen features — and both failed on macOS-arm64:

```
[  FAILED  ] TEST_NYXUS.TEST_3D_GLDM_DE_REGRESSION
[  FAILED  ] TEST_NYXUS.TEST_3D_GLDM_DE_NOBINNING_REGRESSION
```

885 of 888 passed there; nothing else in the family moved. `calc_DE()` is the family's **single**
`fast_log10()` call site (`3d_gldm.cpp`), and that helper evaluates
`fexp + a*signif*signif + b*signif` in `float`, which clang contracts into fused multiply-adds on
arm64 where MSVC does not on x86-64. A pin is one platform's output, so the contraction lands in the
golden.

Both are now banded at `rel=1e-6` through `gldm_3d_regression_band()`, which is the carve-out
`test_3d_glrlm_regression.h` and `test_3d_glszm_regression.h` already carry for `3GLRLM_RE` and
`3GLSZM_ZE`, at the same `1e6` and for the same helper. **Only this one feature is widened** — the
other thirteen of each table stay at the exact drift tier, and the macOS run is the evidence that
they can: all twenty-six of them passed there unchanged.

**Which features are exposed is a structural property, not a judgement.** Counting `fast_log10` call
sites per 3D family: GLCM 13, GLRLM 1, GLSZM 1, GLDM 1, and **GLDZM, NGLDM and NGTDM 0** — those
three cannot drift this way and need no carve-out. That count is what a check should key on rather
than a feature name.

## The matrix under the scalars

All fourteen features are contractions of one dependence matrix `P(i, j)`, so a compensating pair of
errors inside it survives every scalar assertion. The matrix is therefore pinned as well.

**The pinned table is held to two matrices, one of them the run's own.** `TEST_3D_GLDM_MATRIX_PYRADIOMICS`
asserts the cells twice. The first arm reads the table `D3_GLDM_feature::calculate()` left behind —
`P`, the grey-level list `I` its rows are indexed through, and `Ng` / `Nd` / `Nz`, exposed read-only
on the feature object — off the same run the fourteen scalar assertions go through. That puts
production's allocation, its row mapping, its fill loop and its `Nd` trim under the assertion, not
just the neighbourhood walk that feeds them. The second arm comes from a 26-offset walk written out
in the test file, sharing no code with the feature class, which is what keeps the pins from being
whatever production happens to say. Both are compared cell for cell and in both directions — a
populated cell the table does not carry fails as loudly as a pinned cell the run does not produce.

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
| H | **production's `shifts` table loses the `-z` offset** | `TEST_3D_GLDM_MATRIX_PYRADIOMICS`, `TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS` | fail |

C, D and E are the input half: they demonstrate the matrix assertions read the cube the featurisation
produced rather than a hand-written copy of it. F demonstrates the `3GLDM_DE` band is load-bearing —
at `abs=1e-9` the `fast_log10` residual fails, so the band is neither decorative nor over-wide.
H is the production half: `{0,0,-1}` was made a duplicate of `{0,0,+1}`, which is a change no scalar
golden was moved for. Both matrix assertions failed on the arm that reads the run, and the definition
arm — which ran, since the two are separated by `EXPECT` rather than `ASSERT` — still matched the
pins, so the failure named production rather than the goldens. Before the review round this
perturbation failed no matrix assertion at all.

Generator-side controls: removing the header's matrix table makes `gen_gldm3d_pyradiomics.py` exit 1;
so does any bound violation, identity failure, cross-table mismatch, definition mismatch or unpinned
feature, each through its own counter.

### Review round: the default configuration and the degenerate ROI

| # | perturbation | caught by | result |
|---|---|---|---|
| I | **`I[i] = i + 1` → `i + 2` in the no-binning branch** | `TEST_3D_GLDM_LGLE_NOBINNING_REGRESSION` | fails |
| J | **the `aux_min == aux_max` intercept restricted to empty cubes** | `TEST_3D_GLDM_CONSTANT_ROI_REGRESSION` | fails |
| K | a registry row's `test_name` set to a matrix case | `scan_gldm3d_coverage.py --check` | fails |
| L | a registry row's `test_name` set to another family's case | `scan_gldm3d_coverage.py --check` | fails |
| M | a defaults row's `test_name` swapped to the `64`-config case | `scan_gldm3d_coverage.py --check` | fails |
| N | a registry row's `test_name` emptied | `scan_gldm3d_coverage.py --check` | fails |

I is the point of the default-configuration table: the perturbation is confined to the
`ibsi_grey_binning()` branch, so **every assertion that existed before it stayed green** — the
`bincount20` oracle cases, the `64` snapshots and both matrix cases — and only the new pins moved.
That is what "the compiled default was unguarded" means in a failing test rather than in prose.

J is the control for the degenerate cell, and it carried a finding. At the cell's own configuration
(`GLDM_GREYDEPTH=0`) the bypass produces real values and the guard fails as it should — and those
values reproduce PyRadiomics **to the last bit** on all seven dependence-axis features
(`SDE` 0.0066408036122542298, `LDE` 239.41666666666666, `DN` 15.333333333333334, `DNN`
0.31944444444444442, `DV` 26.743055555555554, `GLN` 48, `GLV` 0), with `3GLDM_DE` at the family's
documented `fast_log10` residual and the grey-level features apart only by the level convention.
So nothing under the intercept is wrong; the intercept is the whole defect. Run at *radiomics*
binning instead, the same bypass leaves the case green: `to_grayscale_radiomix()`'s
`binW = (max - min) / binCount` is zero at zero range, every voxel bins to background and the
`Nz == 0` branch emits the same sentinel by a second route — which is why the cell is asserted at the
configuration where the intercept stands alone.

K–N are the registry-side controls. K is the defect the review found: before this round the
per-feature rows named `TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS`, a case that asserts dependence cells
on a hand-written unbinned volume, and no check looked at the `test_name` column closely enough to
object. `check_coverage.py` asked only whether the name was *a* gtest case; the scanner's other two
checks asked which files cover a feature and which functions read a table. The name is now resolved
through `test_all.cc` to the function it runs, which must carry an assertion of the row's own
feature, at its kind, in a file `current_test` names, and at its `config_recipe` — M is the control
for that last clause, since the family's two regression recipes share a feature set, a kind and a
file.

### Second review round: what reading `P` catches that nothing else does

The arm above reads the table `calculate()` left behind. These four controls measure the difference
between that and the previous arm, which reproduced the zone list beside `calculate()` and so could
only ever see the traversal.

| # | perturbation | caught by | everything else |
|---|---|---|---|
| O | **`P.allocate(Nd + 1, Ng + 1)` → `Nd + 3`** | both matrix cases | **46 of 48 GLDM cases stay green** |
| P | **the `if (greyInfo) Nd = max_Nd` trim disabled** | `TEST_3D_GLDM_MATRIX_PYRADIOMICS` | **47 of 48 stay green** |
| Q | **the fill loop skips dependence 6 (`col != 5`)** | both matrix cases | 14 oracle + 28 snapshot cases also fail |
| R | **`lower_bound` row lookup → `inten - 1`** | the 14 `_REGRESSION` cases | **both matrix cases stay green** |

O and P are the point. Neither perturbation changes a single feature value — `calc_*()` reads rows
`1..Ng` and columns `1..Nd`, so extra allocated columns are never summed, and the untrimmed `Nd`
only widens a loop over cells that are zero. Both were invisible to all 48 GLDM assertions before
this round, and each is now caught by the matrix assertion alone: O on `P.width()`, P on
`f.get_Nd()` reading 27 where 15 is pinned.

Q is the fill loop, which the scalars do see — it is here to show the matrix arm names the cell
(`level 4, dependence 6`) where a scalar assertion reports only a moved number.

R is the row mapping, and it is the one this arm does **not** catch; see the limitation at the end of
this report for why and for what fixture would.

## What this report does not cover

- **One oracle.** IBSI calls this family NGLDM and `mirp` exposes one, but Nyxus ships 3D GLDM and 3D
  NGLDM as *separate* families with different conventions, so a `mirp` run would need its own
  reconciliation before it could corroborate anything here.
- **`GLDM_GREYDEPTH = 0` / `IBSI=true` are pinned but unvetted.** They are the same code path reached
  two ways, and it is the **compiled default** — `compile_feature_settings()` names no default for
  `GLDM_GREYDEPTH`, so the zero-fill stands and only `--3gldm/greydepth=N` moves it. All fourteen are
  now snapshotted at that configuration (`gldm3d.regression_ut_phantom_nobinning`), which claims no
  vetting: the oracle question, whether PyRadiomics can be made to skip discretisation on a
  non-integer intensity volume, is still open. See `matrix/gldm3d.md`.
- **A nonempty constant-intensity ROI is a measured defect, guarded rather than fixed.** PyRadiomics
  3.0.1 computes a full GLDM on such an ROI; Nyxus emits the no-value sentinel for all fourteen,
  through an intercept whose comment says "blank ROIs". `test_3d_gldm_constant_roi_regression()` pins
  the emitted value. The fix is a production change on a reachable input, and the same intercept sits
  in five sibling families, so it is tracked as its own change — see `matrix/gldm3d.md`.
- **The two matrix fixtures cannot tell the two row-mapping branches apart.** `calculate()` indexes a
  zone's row as `inten - 1` under `IBSI` and as the position of that level in `I` otherwise. Both
  matrix fixtures carry contiguous grey levels — the phantom's `binCount` binning gives 1..20, the
  small volume's raw levels are 1..4 — so on either one the two expressions are the same number.
  Measured: replacing the `lower_bound` lookup with `inten - 1`, with no golden touched, leaves both
  matrix cases green. What catches it is the MATLAB-binning snapshot, whose levels are gapped: all
  fourteen `_REGRESSION` assertions fail. So the branch is under assertion in this suite, but not by
  the assertion that reads `P`. A fixture that would put it there is a gapped-level volume with a
  `binWidth=1` oracle — the same measurement `matrix/gldm3d.md` lists as what would move the
  no-binning cell to VALID, and the shape `glszm3d.pyradiomics_ibsi_gapped` already uses next door.
