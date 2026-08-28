# 3D GLSZM vs PyRadiomics — vetting report

Family: 3D GLSZM, 16 public features. Oracle: **pyradiomics 3.0.1** (SimpleITK 2.3.1, Python 3.8, in
the `nyxus_oracle` conda env). Generator: `tests/vetting/oracles/gen_glszm3d_pyradiomics.py`.

Two oracle recipes:

- `glszm3d.pyradiomics_bincount20` on `bench_compat_liver_3d` (`compat_int/compat_int_mri.nii` +
  `compat_seg/compat_seg_liver.nii`, label 1) — the family at radiomics binning.
- `glszm3d.pyradiomics_ibsi_gapped` on `bench_cube3_gapped_levels` — the family at `IBSI=true`, added
  in the review pass below, which is the only configuration reaching `calculate()`'s IBSI-only row
  index and `Ng`.

Two drift guards with no oracle: `glszm3d.regression_ut_phantom` (MATLAB binning) and
`glszm3d.regression_ut_phantom_nobinning` (the default a flagless run gets).

Reproduce with `tests/vetting/audit/glszm_3d_golden_regen.md`.

---

## Verdict

**All 16 features reproduce PyRadiomics.** Fifteen agree to within a few ULP -- worst relative
residual 1.19e-15, which is float summation order and nothing else. `3GLSZM_ZE` differs by 1.9e-4,
which is Nyxus' `fast_log10` approximation. No feature changed status and
none was demoted.

What this pass changed is what surrounded that claim:

| | before | after |
|---|---|---|
| oracle band | `agrees_gt(..., 10.)` = ±10% on all 16 | `rel=1e-9` on 15, `rel=1e-3` on `3GLSZM_ZE` |
| pytest band | `rtol=1e-1, atol=1e-2` | `rtol=1e-9, atol=0` (`3GLSZM_ZE` `rtol=1e-3`) |
| the size-zone matrix on the phantom | not asserted at all | 186 cells, `Ng`/`Ns`/`Nz`/`Np`, and a no-other-cell-is-populated count |
| the 4×4×3 matrix check | hand-computed expectations under a `_pyradiomics` name | the same cells, produced by a PyRadiomics run the generator repeats |
| `test_3d_glszm_regression.h` | 16 functions, 0 registrations, never `#include`d, pins reproducible at no configuration | wired in, re-pinned at `%.17g` from `glszm3d.regression_ut_phantom` |
| registry `config_recipe` / `tolerance` | empty on all 16 rows | filled on all 16 |
| the family's settings at a default run | never exercised | `test_3d_glszm_default_greydepth_mechanics` |

A second pass, answering review of PR #454, closed four more gaps. Each of them is a case of an
assertion that named something stronger than what it was checking:

| | before | after |
|---|---|---|
| the matrix pin | rebuilt `P` beside `calculate()` from `gather_size_zones()`, so a defect in the row mapping, `Ng`/`Ns`, the allocation or the fill loop left it green | reads `P`, `I` and the four dimensions off the feature object, through read-only accessors added to `3d_glszm.h` |
| the 4×4×3 connectivity fixture | one populated slice between two empty ones — 26-, 18- and 2D 8-connectivity all produce its nine zones | three populated slices; the four readings give 9, 10, 13 and 13 |
| the `IBSI=true` cell | `NOT MEASURED` in `matrix/glszm3d.md` | vetted, `glszm3d.pyradiomics_ibsi_gapped`, 16/16 + the matrix; and measured identical to `GLSZM_GREYDEPTH=0` |
| the `GLSZM_GREYDEPTH=0` cell | asserted finite only | 16 numeric pins, `glszm3d.regression_ut_phantom_nobinning` |
| `aux_min == aux_max` | `INVALID` — "a blank ROI has no zones" | `DIVERGENCE`: a constant-intensity ROI has a valid one-zone GLSZM; pinned by `test_3d_glszm_constant_roi_regression` and filed in `PR/todo.md` |
| registry rows | 16, each conflating the `-20` oracle with the `+64` regression | 64, one per assertion (SPEC §3) |
| `scan_glszm3d_coverage.py --check` | one-way: passed when `current_test` held the right file plus others | exact file match, kind match, one-recipe-per-case, and (feature, recipe, oracle) uniqueness |

`agrees_gt`'s third argument is a **divisor**: `10.` is ±10%, `1e9` is `rel=1e-9`. The old band
accepted anything within a tenth of the golden on every feature of the family.

---

## Measured residuals, Nyxus against PyRadiomics

Nyxus values are `test_3d_glszm_dump_pyradiomics()` at `%.17g`; PyRadiomics values are a fresh
`RadiomicsFeatureExtractor` run at the recipe.

| feature | PyRadiomics | Nyxus | rel |
|---|---|---|---|
| `3GLSZM_GLN` | 61.77441860465116 | 61.77441860465116 | 0 |
| `3GLSZM_GLNN` | 0.071830719307733909 | 0.071830719307733909 | 0 |
| `3GLSZM_GLV` | 14.965087885343427 | 14.965087885343445 | 1.2e-15 |
| `3GLSZM_HGLZE` | 134.6639534883721 | 134.6639534883721 | 0 |
| `3GLSZM_LAE` | 723.70930232558135 | 723.70930232558135 | 0 |
| `3GLSZM_LAHGLE` | 87509.952325581398 | 87509.952325581398 | 0 |
| `3GLSZM_LALGLE` | 6.2806536910163127 | 6.2806536910163109 | 2.8e-16 |
| `3GLSZM_LGLZE` | 0.016482439101794737 | 0.016482439101794737 | 0 |
| `3GLSZM_SAE` | 0.53068400855035069 | 0.53068400855035058 | 2.1e-16 |
| `3GLSZM_SAHGLE` | 72.656400402294139 | 72.656400402294139 | 0 |
| `3GLSZM_SALGLE` | 0.008788101239865679 | 0.0087881012398656824 | 3.9e-16 |
| `3GLSZM_SZN` | 231.4279069767442 | 231.4279069767442 | 0 |
| `3GLSZM_SZNN` | 0.26910221741481882 | 0.26910221741481882 | 0 |
| `3GLSZM_ZE` | 6.4264170267860647 | 6.4252043186232015 | **1.9e-4** |
| `3GLSZM_ZP` | 0.17916666666666667 | 0.17916666666666667 | 0 |
| `3GLSZM_ZV` | 692.55732828555983 | 692.55732828556063 | 1.1e-15 |

Worst residual outside `3GLSZM_ZE`: **1.19e-15** (`3GLSZM_GLV`). The band is `rel=1e-9`, six orders
of magnitude above the measurement and nine below the previous one.

### `3GLSZM_ZE` and `fast_log10`

`calc_sums_of_P()` accumulates the zone entropy as `fast_log10(p/sum_p + EPS) / LOG10_2`, where
`fast_log10` is a float second-order polynomial approximation and `LOG10_2` is a ten-digit literal;
PyRadiomics uses `numpy.log2`. The residual is 1.9e-4, and the band is `rel=1e-3`, the
documented-residual tier of SPEC §7. This is the same fast path 2D GLCM (`rel=5e-3`) and 2D GLDM
(`rel=2.5e-3`) already band for, and it is a deliberate convention of this codebase rather than a
defect of this family — open question 4 was answered "no" on 2026-08-24. The band is not a free pass:
the other fifteen features are contractions of the same matrix at `rel=1e-9`, so an error in the
matrix fails fifteen assertions at once, plus the 186 matrix cells.

---

## The matrix, and why it is pinned

All sixteen features are sums over one `P(i, j)` size-zone matrix — the count of connected
components of grey level `i` and voxel size `j`. Sixteen scalar assertions cannot separate a correct
matrix from two errors inside it that cancel, so `test_3d_glszm_matrix_pyradiomics` pins the matrix
itself against PyRadiomics' `P_glszm`, the table it builds before any feature formula runs.

On this phantom the matrix is **20 × 634** with **186 non-empty cells** holding **860 zones** over
**4 800** voxels. The assertion pins every one of those cells, the four dimensions, and — the part
that makes the table complete rather than a sample — a count that no cell outside the 186 is
populated.

### A column index is not a zone size

PyRadiomics' `_calculateCoefficients()` **deletes every column whose zone size no zone occupies** and
carries the surviving sizes in `jvector`. On this phantom that turns 634 columns into 46, whose sizes
run 1..16, 18..29, 31, 32, 34, 36, 44, 51, 59, 61, 65, 80, 88, 96, 123, 131, 137, 199, 249, 634.
Nyxus keeps the matrix dense — column index = zone size − 1, width = the largest zone — so a
comparison that reads the size off the column index relabels almost every large zone.

The first draft of the generator did exactly that, and the **cross-table check caught it before
anything was pinned**: recomputing the sixteen IBSI definitions from the mislabelled matrix
reproduced the nine features that do not weight by `j` and missed the seven that do, by up to 94%
(`3GLSZM_ZV` 41.17 against 692.56). That is the check revet.txt §9 asks for, doing the job it exists
for on the generator itself rather than on the tree.

With the sizes taken from `jvector`, all sixteen recompute from the matrix at `rel < 1e-9`.

### It is the production matrix, not a copy of it

The assertion reads `P` and `I` off the `D3_GLSZM_feature` that the sixteen scalar assertions ran,
through read-only accessors `3d_glszm.h` now carries, and checks `Ng`, `Ns`, `Nz`, `Np` and the
width and height the table was allocated at as well as its cells.

It did not, until the review pass. It called the static `gather_size_zones()` and filled a local
`SimpleMatrix<int>` from the result, which reproduces the zone list but not the four places a defect
can live in `calculate()`: the row mapping, `Ng`/`Ns`, the allocation, and the fill loop. A pin that
rebuilds the table it is pinning cannot see any of them. Negative control J below plants a defect in
the fill loop alone — `gather_size_zones()` stays correct — and the assertion now fails naming the
level and the size, where before it would have passed.

### The 4×4×3 volume

The phantom's matrix has 186 cells nobody can check by hand, so the connectivity itself is checked on
a fixture that can be: `bench_cube4x4x3_zcross`, three populated 4×4 slices, four grey levels,
**nine zones**. Its matrix is

|            | size=1 | size=2 | size=3 |
|---|---|---|---|
| level=1 | 0 | 1 | 1 |
| level=2 | 1 | 1 | 0 |
| level=3 | 0 | 1 | 1 |
| level=4 | 2 | 1 | 0 |

This check existed before the first pass but was named `_pyradiomics` while asserting hand-computed
expectations that no PyRadiomics run had ever produced — the function name is the oracle claim
(revet.txt §9), so the name was crediting an oracle the assertion did not use. The fix was to make
the claim true rather than to rename: `gen_glszm3d_pyradiomics.py` runs `RadiomicsGLSZM` on the same
literal at `binWidth=1`.

**The volume it ran on could not see 3D connectivity at all.** It had one populated slice between two
empty ones, so for every foreground voxel all 18 neighbours with `dz != 0` were background. Counting
its zones under the readings it was supposed to separate:

| fixture | 26-conn | 18-conn | 6-conn | 2D 8-conn |
|---|---|---|---|---|
| the old volume, one populated slice | 9 zones | **9, same table** | 11 | **9, same table** |
| `bench_cube4x4x3_zcross` | 9 zones | 10 | 13 | 13 |
| `bench_cube3_gapped_levels` | 6 zones | 8 | 8 | 9 |

The replacement carries a strictly vertical run (`dz=±1`, `dy=dx=0`), an in-slice diagonal, a z-edge
join (`dz=1, dy=0, dx=1`), a z-corner join (`dz=1, dy=1, dx=1`) that no 18-neighbourhood makes, and
two same-level voxels two slices apart that must stay two zones — the control in the other direction,
against an implementation that joined a whole column. PyRadiomics reproduces all eight of its cells.

**This is not SPEC §5.2 self-consistency.** The volumes are the fixtures, not copies of ones — there
is no file to read them from — and their expected values come from an independent tool, not from the
model that wrote the volumes.

### Connectivity

Both sides walk the full 26-voxel neighbourhood. `gather_size_zones()` spells out 8 in-slice, 8
upper, 8 lower and 2 strictly-vertical offsets; PyRadiomics' C extension uses full 3D connectivity by
construction. The two literal fixtures are the direct check, and negative control K removes the
upper-Z offset block: both fail, 11 zones against 9 and 8 against 6. No repeat here of the 3D GLDZM
4-vs-8 defect.

---

## The `IBSI=true` branch, measured

`calculate()` overwrites `GLSZM_GREYDEPTH` with 0 whenever `IBSI` is on, and two of its expressions
then read the flag directly: the row a zone lands in is `zone.first - 1` rather than the position of
that level in `I`, and `Ng` is `max_element(I)` rather than `I.size()`. Nothing in the tree ran on
that branch, and `matrix/glszm3d.md` recorded it as NOT MEASURED with the note that a fixture whose
levels are contiguous from 1 cannot tell the two apart.

`bench_cube3_gapped_levels` is that fixture: a 3×3×3 literal carrying levels 1, 3 and 5, with 2 and 4
absent, so a level's position in the occupied-level list and the level itself are different numbers.

**PyRadiomics is a valid oracle at this point.** At `binWidth=1` it leaves integer levels where they
are — which is exactly what Nyxus' no-binning path reads off the volume — and it reports `Ng = 5` for
three occupied levels, leaving rows 2 and 4 empty. So does Nyxus. Measured:

| | oracle | nyxus | rel |
|---|---|---|---|
| 15 features | — | — | exact |
| `3GLSZM_ZE` | 1.7924812503605767 | 1.7904872894287109 | 1.11e-3 |
| matrix | 4 cells, `Ng`=5 `Ns`=3 `Nz`=6 `Np`=9 | identical | — |

`3GLSZM_ZE` bands at `rel=2e-3` here against `rel=1e-3` on the phantom. It is the same `fast_log10`
path; zone entropy sums `-p·log2(p)` over the matrix, so on six zones each term carries the
approximation's error at full weight, while on the phantom's 860 zones it is spread over 186 terms
and partly cancels. Measured, banded at the measurement rounded up, not assumed.

**And the branch turns out not to be a branch.** At `greyInfo == 0` the `ibsi_grey_binning` path
builds `I` as the contiguous run `1..max(D)` whatever the volume holds — absent levels get empty rows
rather than being packed out — so the position of a level in `I` is always `level - 1` and `I.size()`
is always `max(I)`. The two forms compute the same numbers on every input.
`test_3d_glszm_ibsi_equals_no_binning_mechanics` asserts that on the gapped fixture: sixteen values
and the whole `SimpleMatrix<int>`, bit for bit. It passes `GLSZM_GREYDEPTH=64` on the IBSI side, so
it measures the overwrite too — a run that honoured the 64 would bin into MATLAB levels and miss.

That is why `matrix/glszm3d.md` now carries one cell for the two rows rather than one unmeasured row.

---

## The `GLSZM_GREYDEPTH == 0` cell

This is the value a run passing no `--3glszm/greydepth` reaches the feature with:
`compile_feature_settings()` zero-fills the family's settings vector and nothing writes the entry.
The first pass added `test_3d_glszm_default_greydepth_mechanics`, which asserts that the 0 really
arrives and that the sixteen features are finite there. Finiteness is mechanics; it says nothing
about the numbers.

`glszm3d.regression_ut_phantom_nobinning` pins them: sixteen `%.17g` goldens on `bench_ut57_3d` under
one case, `TEST_3D_GLSZM_DEFAULT_GREYDEPTH_REGRESSION`. One case for the sixteen because one phantom
read answers all of them, and the matrix at this setting is 3024 grey levels wide against `binCount`'s
20. They claim no oracle: PyRadiomics discretises before it counts zones and has no counterpart for
reading 2001 raw levels off this phantom. The *setting* is vetted on the fixture that is small enough
to carry an oracle, which is `glszm3d.pyradiomics_ibsi_gapped` above.

---

## A constant-intensity ROI is discarded, and should not be

`calculate()` opens with

```cpp
// intercept blank ROIs (equal intensity)
if (r.aux_min == r.aux_max) { invalidate (STNGS_NAN(s)); return; }
```

Those are two different conditions. A constant-intensity ROI is fully populated: at no binning or at
MATLAB binning it has one grey level, one 26-connected zone if its voxels touch, a size-zone matrix
with a single populated cell and sixteen finite features over it. On a 2×2×2 block of one intensity
that is `SAE = 1/64`, `LAE = 64`, `ZE = 0`, `GLV = 0`, `ZP = 1/8`. Nyxus returns the soft-NaN
sentinel — `--noval`, default `0.0` — for all sixteen.

The guard is doing real work at radiomics binning, where `to_grayscale_radiomix` divides by
`(max - min)`; it is unconditional, so it discards the two schemes that would have answered as well.
Negative control O makes one voxel of the constant block different, so the intercept does not fire,
and the family computes `SAE = 0.51020408163265307` on the result — the arithmetic is there, it is
the guard that is too wide.

Recorded as a **divergence** in `matrix/glszm3d.md` and filed in `PR/todo.md`. Narrowing it changes a
public feature's output on a reachable input, so it is `src/` work on its own branch; this pass pins
the current behaviour instead, in `test_3d_glszm_constant_roi_regression`, with the sentinel set to a
distinctive `-98765` so that a zero-filled feature buffer cannot satisfy the assertion.

---

## What the orphan would have done

`test_3d_glszm_regression.h` was 16 functions, 16 pins and no `#include` in `test_all.cc`. It built a
settings vector that set `GREYDEPTH=64` and **left `GLSZM_GREYDEPTH` at 0** — no binning — so the
file had never executed at the settings it carried. Whether that mattered was unmeasured; the
previous handoff explicitly refused to predict it from 3D NGTDM's. Measured now, on
`bench_ut57_3d`, against the sixteen values the file pinned:

| feature | orphan pin | `GLSZM_GREYDEPTH=0` (what it carried) | `=64` (the settled recipe) | `=100` (what its comment claimed) |
|---|---|---|---|---|
| `3GLSZM_GLN` | 2037 | 133.993 (93%) | 1349.40 (34%) | 1423.18 (30%) |
| `3GLSZM_GLNN` | 0.03 | 0.000541743 (98%) | 0.0330986 (10%) | 0.0186195 (38%) |
| `3GLSZM_GLV` | 106.5 | 331758 (311410%) | 84.6623 (21%) | 246.446 (131%) |
| `3GLSZM_HGLZE` | 2485.9 | 4.37559e+06 (175916%) | 2685.07 (8%) | 6055.68 (144%) |
| `3GLSZM_LAE` | 1377.1 | 2.71993 (100%) | 15936.4 (1057%) | 3165.34 (130%) |
| `3GLSZM_LAHGLE` | 1.24578e+06 | 9.82273e+06 (688%) | 1.69445e+07 (1260%) | 6.81014e+06 (447%) |
| `3GLSZM_LALGLE` | 1.9 | 1.21248e-06 (100%) | 18.9524 (897%) | 1.83125 (4%) |
| `3GLSZM_LGLZE` | 0.0005 | 3.26435e-07 (100%) | 0.000434908 (13%) | 0.000201074 (60%) |
| `3GLSZM_SAE` | 0.6 | 0.942958 (57%) | 0.564106 (6%) | 0.664555 (11%) |
| `3GLSZM_SAHGLE` | 1592 | 4.17417e+06 (262096%) | 1570.88 (1%) | 4177.65 (162%) |
| `3GLSZM_SALGLE` | 0.0003 | 3.01545e-07 (100%) | 0.000231076 (23%) | 0.000125736 (58%) |
| `3GLSZM_SZN` | 24582.1 | 213149 (767%) | 12492.9 (49%) | 31288.3 (27%) |
| `3GLSZM_SZNN` | 0.36 | 0.86178 (139%) | 0.306431 (15%) | 0.409345 (14%) |
| `3GLSZM_ZE` | 7.44 | 11.2509 (51%) | 7.34419 (1%) | 7.72174 (4%) |
| `3GLSZM_ZP` | 0.275 | 0.901265 (228%) | 0.148558 (46%) | 0.278521 (1%) |
| `3GLSZM_ZV` | 1362.3 | 1.48882 (100%) | 15891.1 (1066%) | 3152.45 (131%) |

**Wired in as it stood, all 16 of its assertions would have failed on their first run**, at its own
±10% band. The mechanism is not NGTDM's — nothing here is NaN, the values are finite and the family
computes — but the verdict is the same: an orphan is not merely untested, it is untested *at settings
nobody has ever executed*.

The pins reproduce at no configuration in the tree: 12 of 16 miss at the settled `=64`, and 13 of 16
miss at the `=100` the file's own header comment claimed, so they are a snapshot of an older state of
the code rather than of a recipe that got lost. They are replaced with a fresh `%.17g` run at
`glszm3d.regression_ut_phantom` — `GREYDEPTH=64`, `IBSI=false`, `GLSZM_GREYDEPTH=64`, the
configuration the retired 3D coverage sweep ran every 3D family at on this same phantom.

`GLSZM_GREYDEPTH=0` is not degenerate, and that is the difference from `NGTDM_RADIUS=0`: it is this
family's documented "no binning" default, `--3glszm/greydepth` is what overrides it, and a run that
passes no such flag produces finite values for all sixteen features.
`test_3d_glszm_default_greydepth_mechanics` asserts exactly that, so **no `src/` change is proposed
by this branch**.

---

## Range, identity and cross-table checks

Run mechanically over the oracle's own output by the generator, before anything is pinned:

- **Bounds**, 8 features: `SAE`, `ZP`, `GLNN`, `SZNN` in [0, 1]; `LAE` ≥ 1; `ZE`, `GLV`, `ZV` ≥ 0.
  8/8 in range.
- **Identities**, 4: `GLNN == GLN / Nz` and `SZNN == SZN / Nz` to `rel=1e-12`; `SAE ≤ LAE`;
  `LGLZE ≤ HGLZE`. 4/4 hold.
- **Cross-table**, 16: every feature recomputed from the pinned matrix alone and compared against the
  value the public extractor reported. 16/16 at `rel < 1e-9`. This is the check that caught the
  `jvector` error above.
- **Reverse check**: every GLSZM feature PyRadiomics produces is pinned; 0 unpinned. The generator
  exits non-zero if that stops being true.

The generator **opens `test_3d_glszm_pyradiomics.h`** and re-verifies the pins in it rather than
comparing against a literal copy of its own, and it counts braces rather than stopping at the first
`};`, so a nested-brace table cannot silently lose its last entry.

---

## Negative controls

Nine, each applied to the tree, built and run, then reverted. Every one failed as designed. Both
halves are controlled — a perturbed **golden** proves the assertion compares, a perturbed **input**
proves it compares the right thing:

| # | half | perturbation | result |
|---|---|---|---|
| A | golden | `{ 11, 634, 1 }` → `{ 11, 634, 2 }` in the matrix table | matrix test fails, `Which is: 1` vs `2`, `SCOPED_TRACE` naming grey level 11, zone size 634 |
| B | golden | delete the pinned cell `{ 20, 2, 2 }` | matrix test fails on the completeness count, 186 vs 185 |
| C | **input** | `cube[0] = hi;` after extraction, before binning | matrix test fails, 861 zones vs 860 |
| D | **input** | read the matrix at `bin_intensities_3d(..., -19)` instead of `-20` | matrix test fails, `Ng` 19 vs 20 |
| E | **input** | one voxel of the 4×4×3 volume, `4` → `3` | small-matrix test fails, `Ns` 4 vs 3 |
| F | golden | `3GLSZM_SAE` golden perturbed by rel=1.7e-8 | fails, `1.45e-09 > 5.31e-10` — and the old ±10% band would have passed it |
| G | golden | `3GLSZM_ZE` golden perturbed by rel=2e-3 | fails at its own looser band, `0.0141 > 0.00644` |
| H | golden | `3GLSZM_ZP` regression pin perturbed by rel=1e-6 | fails, `1.49e-07 > 1.49e-10` |
| I | **input** | mechanics test expects `GLSZM_GREYDEPTH == 1` | fails, reading 0 — so it reads the compiled settings, not a constant |

C is the control the previous family's session did not run. Its own pass planted a bad *golden* in a
matrix assertion, which proves the assertion compares but not that it reads the run under test; that
assertion turned out to be working from a hand-written copy of the phantom.

**The review pass re-ran the controls that its changes touch, and added the ones its changes are
about.** C and D perturbed the test's own copy of the cube and its own binning call; the matrix
assertion no longer has either, so they are retired rather than re-run. E ran on the old 4×4×3
volume, which no longer exists. Five controls were applied to the current tree, built and run, then
reverted:

| # | half | perturbation | result |
|---|---|---|---|
| A | golden | `{ 11, 634, 1 }` → `{ 11, 634, 2 }` (re-run) | matrix test fails, `1` vs `2`, trace naming grey level 11, zone size 634 |
| J | **input** | production fill loop skips zones of size 5 — `gather_size_zones()` untouched | matrix test fails on the same cell; this is the defect the rebuilt-table version could not see |
| K | **input** | the upper-Z offset block of `gather_size_zones()` removed | connectivity test fails, 11 zones vs 9; gapped test fails, 8 vs 6 |
| L | **input** | one voxel of `bench_cube4x4x3_zcross`, the `(1,3,3)` level-1 voxel → 4 | connectivity test fails, 8 zones vs 9 |
| M | **input** | one voxel of `bench_cube3_gapped_levels`, the bridging level-3 voxel → 5 | gapped test fails on the scalars |
| N | golden | `3GLSZM_ZP` default-configuration pin perturbed by rel=1e-6 | default-greydepth regression fails |
| O | **input** | the constant 2×2×2 block gets one different voxel, so the intercept does not fire | constant-ROI test fails, `0.51020408163265307` vs `-98765` |

K is the control that answers the fixture question directly: under the old volume it changes nothing,
because that volume's zones never crossed a slice.

---

## Provenance

- tool: pyradiomics 3.0.1, SimpleITK 2.3.1, Python 3.8.20, conda env `nyxus_oracle`
- generator: `tests/vetting/oracles/gen_glszm3d_pyradiomics.py`, run offline; CI never invokes it
- Nyxus values: `runAllTests --gtest_filter=*3D_GLSZM_DUMP_*`, Release/MSVC, `USEGPU=OFF`
- gtest suite: 912 cases, 911 pass, 1 skipped (`TEST_2D_GABOR_GPU_RUNS_MECHANICS`, CPU-only build).
  Baseline on the branch point is 872, so the delta is the 40 cases this branch adds — 20 in the
  first pass and 4 in the review pass, on top of the 16 regression functions the first pass wired in
