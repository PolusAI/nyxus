# 3D GLSZM vs PyRadiomics — vetting report

Family: 3D GLSZM, 16 public features. Oracle: **pyradiomics 3.0.1** (SimpleITK 2.3.1, Python 3.8, in
the `nyxus_oracle` conda env). Recipe: `glszm3d.pyradiomics_bincount20`. Fixture:
`bench_compat_liver_3d` (`compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`,
label 1). Generator: `tests/vetting/oracles/gen_glszm3d_pyradiomics.py`.

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

### The 4×4×3 volume

The phantom's matrix has 186 cells nobody can check by hand, so the connectivity itself is checked on
a fixture that can be: one populated 4×4 slice between two empty ones, four grey levels, **nine
zones**. Its matrix is

|            | size=1 | size=2 | size=3 |
|---|---|---|---|
| level=1 | 2 | 1 | 0 |
| level=2 | 1 | 0 | 1 |
| level=3 | 0 | 0 | 1 |
| level=4 | 2 | 0 | 1 |

This check existed before this pass but was named `_pyradiomics` while asserting hand-computed
expectations that no PyRadiomics run had ever produced — the function name is the oracle claim
(revet.txt §9), so the name was crediting an oracle the assertion did not use. The fix was to make
the claim true rather than to rename: `gen_glszm3d_pyradiomics.py` now runs `RadiomicsGLSZM` on the
same literal at `binWidth=1` and reproduces all seven cells. The hand-worked values were correct.

**This is not SPEC §5.2 self-consistency.** The volume is the fixture, not a copy of one — there is
no file to read it from — and its expected values come from an independent tool, not from the model
that wrote the volume.

### Connectivity

Both sides walk the full 26-voxel neighbourhood. `gather_size_zones()` spells out 8 in-slice, 8
upper, 8 lower and 2 strictly-vertical offsets; PyRadiomics' C extension uses full 3D connectivity by
construction. The 4×4×3 fixture is the direct check — its single populated slice makes every zone a
2D 8-connected component with a known answer, and dropping to 4-connectivity would split level 3's
diagonal chain into three zones. No repeat here of the 3D GLDZM 4-vs-8 defect.

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
`glszm3d.regression_ut_phantom` — `GREYDEPTH=64`, `IBSI=false`, `GLSZM_GREYDEPTH=64`, which is what
`make_3d_coverage_settings()` already runs every 3D family at on this same phantom.

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
assertion turned out to be working from a hand-written copy of the phantom. Here the cube comes back
out of `extract_3d_glszm()` and C is what demonstrates it.

---

## Provenance

- tool: pyradiomics 3.0.1, SimpleITK 2.3.1, Python 3.8.20, conda env `nyxus_oracle`
- generator: `tests/vetting/oracles/gen_glszm3d_pyradiomics.py`, run offline; CI never invokes it
- Nyxus values: `runAllTests --gtest_filter=*3D_GLSZM_DUMP_*`, Release/MSVC, `USEGPU=OFF`
- gtest suite: 892 cases, 891 pass, 1 skipped (`TEST_2D_GABOR_GPU_RUNS_MECHANICS`, CPU-only build).
  Baseline on the branch point is 872, so the delta is the 20 cases this branch adds
