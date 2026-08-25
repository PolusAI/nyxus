# 3D NGTDM vs PyRadiomics — vetting report

All five 3D NGTDM features reproduce PyRadiomics 3.0.1 **exactly** — every one identical in all 53
bits of the mantissa — on the fixture the family's test header already named. So the five `vetted`
rows were right, and what this audit changes is everything around them: the band they were asserted
at, the recipe they record, the matrix underneath them, and one defect that made the family
unusable in the configuration a default run reaches.

| what | before | after |
|---|---|---|
| oracle band | `agrees_gt(..., 10.)` = ±10% | `rel=1e-9` |
| NGTDM matrix (`n_i`, `p_i`, `s_i`) on the phantom | not asserted | 18 values pinned, per grey level |
| the hand-worked 4×4 example | 8 values at `agrees_gt(..., 1)` = **±100%** | 12 values at `rel=1e-9` |
| drift guards on `bench_ut57_3d` | 5 pins in a file nothing included | wired in, re-pinned at `%.17g` |
| `NGTDM_RADIUS` with no `--3ngtdm/radius` | 0 → every feature NaN | 1, guarded by a mechanics test |
| recorded provenance | the wrong two files and a GLCM sentence | the fixture the test actually loads |

---

## Tool and configuration

| | |
|---|---|
| Tool | PyRadiomics **3.0.1** (SimpleITK 2.3.1, Python 3.8), conda env `nyxus_oracle` |
| Generator | `tests/vetting/oracles/gen_ngtdm3d_pyradiomics.py` |
| Benchmark | `bench_compat_ngtdm_3d` — `compat_int_ngtdm_3d.nii` + `compat_seg_ngtdm_3d.nii`, label 57 |
| Recipe | `ngtdm3d.pyradiomics_binwidth1` |
| PyRadiomics settings | `binWidth=1`, `distances=[1]`, `resampledPixelSpacing=None`, `force2D=False`, `imageType=Original` |
| Nyxus settings | `GREYDEPTH=100`, `IBSI=false`, `NGTDM_GREYDEPTH=0`, `NGTDM_RADIUS=1` |

The phantom is 4×4×3 = 48 voxels: one populated slice of the discrete values 1..5 between two
all-zero slices, with **every** voxel labelled 57.

**The two grey-level scales coincide, they are not made to.** PyRadiomics' `binWidth=1` maps a value
to `floor(x/1) − floor(min/1) + 1`, and the phantom's minimum is 0, so its levels 0..5 become 1..6.
Nyxus does not bin at all at `NGTDM_GREYDEPTH=0`, but `D3_NGTDM_feature::calculate()` shifts every
level by one when the minimum is zero — landing on the same 1..6. Nothing is background on either
side. That is why a `rel=1e-9` band is honest here rather than optimistic: there is no discretisation
convention left to absorb.

### PyRadiomics cannot load this phantom through its public API

`imageoperations.getMask()` raises

```
ValueError: No labels found in this mask (i.e. nothing is segmented)!
```

whenever `numpy.unique(mask)` has a single entry. This mask is label 57 in all 48 voxels, so it trips
that check, and `RadiomicsFeatureExtractor.execute()` goes through `getMask()`. **The invocation the
test header recorded —**

```
pyradiomics mri.nii.gz liver.nii.gz --param settings1.yaml
```

**— has therefore never been runnable against this fixture**, and it named two files
(`ut_inten.nii` as both image and mask) that the test does not load either. The generator constructs
`radiomics.ngtdm.RadiomicsNGTDM` directly, which reaches the same feature code without the loader
check. That limitation is now recorded in `TOOLS.md` and in the benchmark entry, because it applies
to any future oracle run against a background-free mask, not only to this family.

### Two references, not one

Because the oracle had to be driven through a non-public entry point, `gen_ngtdm3d_pyradiomics.py`
carries `reference_ngtdm()`: a plain-numpy NGTDM built from the IBSI definition, in exact rational
arithmetic, with no `radiomics` import in its path. Every value in this report is produced by both.

| comparison | worst relative difference |
|---|---|
| PyRadiomics vs the independent reference, 6 levels × (`p_i`, `s_i`) | **2.7e-16** |
| PyRadiomics vs the independent reference, 5 features | **0** |
| PyRadiomics vs the independent reference, doc example, 4 levels | **0** |

So the numbers Nyxus is held to are the definition's, not one implementation's reading of it.

---

## Results — Nyxus reproduces all five bit-for-bit

Nyxus values from `test_3d_ngtdm_dump_pyradiomics()`, at `%.17g`:

| feature | PyRadiomics 3.0.1 | Nyxus | rel. diff | verdict |
|---|---|---|---|---|
| `3NGTDM_BUSYNESS` | 4.5534015564267669 | 4.5534015564267669 | 0 | agree |
| `3NGTDM_COARSENESS` | 0.030118770647251797 | 0.030118770647251797 | 0 | agree |
| `3NGTDM_COMPLEXITY` | 32.130372204003443 | 32.130372204003443 | 0 | agree |
| `3NGTDM_CONTRAST` | 0.23138014315250832 | 0.23138014315250832 | 0 | agree |
| `3NGTDM_STRENGTH` | 1.2458005968884538 | 1.2458005968884538 | 0 | agree |

Not "agree to 1e-15" — identical doubles. The band is set at `rel=1e-9` rather than at equality
because a cross-tool assertion should survive a compiler's reassociation of the same sum, not because
anything was measured off it.

**No feature in this family goes through `fast_log10`.** Coarseness, Contrast, Busyness, Complexity
and Strength are sums of products, absolute differences and squares — there is no logarithm anywhere
in `3d_ngtdm.cpp`. That is why this family holds the exact tier where 2D GLDM's `GLDM_DE`, 2D GLSZM's
entropy and 3D GLRLM's `RE` each needed a measured band.

---

## The matrix, not only the five scalars

All five features are contractions of one table over the grey levels present:

```
Coarseness = 1 / Σ p_i s_i
Contrast   = [ Σ_i Σ_j p_i p_j (i−j)² / (N_g,p (N_g,p − 1)) ] · [ Σ_i s_i / N_v,p ]
Busyness   = Σ p_i s_i / Σ_i Σ_j |i p_i − j p_j|
Complexity = Σ_i Σ_j |i−j| (p_i s_i + p_j s_j) / (p_i + p_j) / N_v,p
Strength   = Σ_i Σ_j (p_i + p_j) (i−j)² / Σ s_i
```

Eighteen numbers decide all five, and every one of those five is a sum — so two errors in the table
can cancel inside any of them. Five scalar assertions cannot tell that apart from a correct matrix.
PyRadiomics computes `P_ngtdm` **before** any feature formula runs, so intercepting it costs one
attribute read and reimplements nothing:

| grey level `i` | `n_i` | `p_i` | `s_i` |
|---:|---:|---|---|
| 1 | 32 | 0.6666666666666666 | 47.05118411000764 |
| 2 | 6 | 0.125 | 0.8266145619086798 |
| 3 | 2 | 0.041666666666666664 | 2.358288770053476 |
| 4 | 4 | 0.08333333333333333 | 9.10880296174414 |
| 5 | 1 | 0.020833333333333332 | 3.1923076923076925 |
| 6 | 3 | 0.0625 | 12.916289592760181 |

`test_3d_ngtdm_matrix_pyradiomics` drives `D3_NGTDM_feature::gather_zones` and `calc_NGTDM` over the
phantom's levels and asserts all 18, plus `N_v,p = 48` and the row order. Level 1 holding 32 of the
48 voxels is the two all-zero slices: they are ROI voxels, not padding.

**It takes those levels out of the same featurisation the five scalar assertions run**, rather than
from a hand-written copy of the phantom. That distinction is the whole point and it was not free —
the first version of this assertion carried a 48-entry literal of the phantom's levels, which is
SPEC §5.2's fixture-encodes-the-model shape: it would have kept passing after the loader started
producing something else, while claiming to back the same numbers the scalars do. `extract_3d_ngtdm`
now hands the ROI's voxel cube back beside `fvals`, and the assertion works from that. The one step
still reproduced is `calculate()`'s zero-min correction (`+1` on every level when the minimum is 0),
which at `NGTDM_GREYDEPTH=0` is the whole of its preamble; the minimum is **asserted**, not assumed,
because the shift is conditional on it.

This is the same finding the reviewer raised on #438 item 2 — a mean recomputed from a second
featurisation of the same slices, where "only the accident that both are deterministic kept them
agreeing". Here the two were not even reading the same input.

### Negative control

The rule this pass exists to satisfy is that a new assertion shape must be shown to fail. The pinned
`s_5` was moved from `3.1923076923076925` to `3.1923077242307694` — a relative error of **1e-8**,
chosen so that it exceeds the `rel=1e-9` band while level 5's share of every scalar (2% of `Σ p_i s_i`,
4% of `Σ s_i`) keeps all five features inside it.

| | result |
|---|---|
| `TEST_3D_NGTDM_MATRIX_PYRADIOMICS` | **FAILED**, `Google Test trace: grey level 5`, `3.19231e-08 > tolerance=3.19231e-09` |
| the five `TEST_3D_NGTDM_*_PYRADIOMICS` | all passed |
| `gen_ngtdm3d_pyradiomics.py` | exit 1, `FAIL ... i=5 s_i: oracle=3.1923076923076925 pinned=3.1923077242307696 rel=1e-08` |

So the per-level table catches an error in the matrix that every scalar assertion in the family is
blind to, and the generator catches it independently of the build.

**A second control, for the property the rewrite added:** one voxel of the cube the featurisation
returned was perturbed (`levels[20] += 1`) with every golden left alone. The matrix assertion failed
naming grey level 1 (`48.1176` against `47.0512`) while all five scalar assertions passed, since they
read `fvals` from the unperturbed run. That is the check the literal version could not have made —
it never touched the loader, so no change to what the loader produces could have reached it.

Both perturbations were reverted.

### The hand-worked example was asserted at ±100%

`test_3d_ngtdm_matrix_correctness_pyradiomics` compared eight values through
`agrees_gt(P[0], 0.375, 1)`. `agrees_gt`'s third argument is a **divisor** — `tolerance = golden /
frac` — so `1` makes the tolerance equal the golden and the assertion accepts anything in
`[0, 2×golden]`. That is not a loose band on the `P` and `S` vectors the whole family is derived
from; it is no band.

The values themselves were three-significant-figure copies of PyRadiomics' documentation, and one of
them is visibly rounded: `s_3` was pinned at `3.03` where the exact value is `91/30 =
3.033333333333333`, an 1.1e-3 relative error that `rel=1e-9` rejects and `1` never could. The test is
now `test_3d_ngtdm_docmatrix_pyradiomics`, pinning all twelve entries at full precision against a
fresh run.

Two details of that example are worth recording because both tools handle them the same way and a
reader of the PyRadiomics docstring may not expect it. Grey level 4 does not occur in the image;
both tools **drop empty levels**, so the table has four rows and not the five the docstring's table
shows. And the docstring's own table and its worked arithmetic disagree on `s_3` — the table says
`2.63`, the text computes `3.03`. The run agrees with the text.

---

## The family returns NaN in the default configuration

`Environment::compile_feature_settings()` (`src/nyx/env_features.cpp`) zero-fills each family's
settings vector and then writes back the ones that must not be zero. It covered `GLCM_OFFSET`,
`GLCM_GREYDEPTH` and `GLCM_NUMANG`; it did not cover `NGTDM_RADIUS`.

`NGTDM_RADIUS` is the Chebyshev radius of the neighbourhood. At 0, `gather_zones()`'s scan is

```cpp
for (int dz = -0; dz <= 0; dz++)
  for (int dy = -0; dy <= 0; dy++)
    for (int dx = -0; dx <= 0; dx++)
      { if (neig == centre) continue; ... }
```

— one iteration, which is the voxel of consideration, which is skipped. `nd` stays 0 for every voxel,
no zone is pushed, `N` is all zeros, `Nvc = 0`, and `P[i] = 0/0`.

Measured against the settings vector the environment actually compiles, on the phantom above:

```
[PROBE] default NGTDM_RADIUS=0 NGTDM_GREYDEPTH=0
[PROBE] fcode 699 = -nan(ind)      3NGTDM_COARSENESS
[PROBE] fcode 700 = -nan(ind)      3NGTDM_CONTRAST
[PROBE] fcode 701 = -nan(ind)      3NGTDM_BUSYNESS
[PROBE] fcode 702 = -nan(ind)      3NGTDM_COMPLEXITY
[PROBE] fcode 703 = -nan(ind)      3NGTDM_STRENGTH
```

Every 3D run that asks for these features and does not pass `--3ngtdm/radius` gets five NaNs. Nothing
caught it because every assertion in the tree sets the radius explicitly: the C++ tests build their
own `Fsettings`, and `tests/python/test_nyxus.py::test_3d_ngtdm_compatibility` calls
`set_metaparam("3ngtdm/radius=1")`.

**Fixed**, one line beside the GLCM default it mirrors, and guarded by
`test_3d_ngtdm_default_radius_mechanics`, which compiles the real settings vector and requires the
five values to be finite. The guard is the point: the value could be reverted by anyone reformatting
that block, and a `1` in a settings assignment does not look like a correctness constraint.

This is the same failure the tree already knows about in 2D GLCM —
`test_2d_glcm_contrast_nonzero_by_default_mechanics` exists for `GLCM_OFFSET=0`, and
`check_test_names.py`'s `KIND_EXCEPTIONS` still carries its reason. Worth a sweep: any
`NyxSetting` whose zero value is degenerate rather than merely a default needs an entry in that
block, and the two found so far were both found by accident.

---

## The drift guards ran for the first time

`test_3d_ngtdm_regression.h` held five functions, a five-entry golden table and a `#if 0` block, and
**no translation unit included it** — its five `TEST()` registrations did not exist either. It has
now been wired in, and two things about it are worth recording.

Its settings vector set `GREYDEPTH=64` and nothing NGTDM-specific, so `NGTDM_RADIUS` would have been
0 and all five assertions would have compared NaN against their pins on first execution. The file now
runs at `ngtdm3d.regression_ut_phantom` — `GREYDEPTH=64`, `NGTDM_GREYDEPTH=64`, `NGTDM_RADIUS=1` —
matching what the 3D peers' regression recipes do.

At that recipe the old pins turn out to have been roughly right and uselessly imprecise:

| feature | old pin | Nyxus, `%.17g` | rel. diff | at ±10% |
|---|---|---|---|---|
| `3NGTDM_COARSENESS` | 0.00004 | 4.1746559837294642e-05 | 4.2e-2 | passes |
| `3NGTDM_CONTRAST` | 0.66 | 0.63226607482802633 | 4.4e-2 | passes |
| `3NGTDM_BUSYNESS` | 46.0 | 44.389552850401223 | 3.6e-2 | passes |
| `3NGTDM_COMPLEXITY` | 2936.0 | 2819.3512285176689 | 4.1e-2 | passes |
| `3NGTDM_STRENGTH` | 0.024 | 0.024654440905359544 | 2.7e-2 | passes |

Every one would have passed its own ±10% band, so wiring the file in at the band it carried would
have found nothing. They are re-pinned at `%.17g` and the band tightened to `rel=1e-9`; the file
carries `test_3d_ngtdm_dump_regression()` to regenerate them.

Note the shape of that: the orphan's real problem was not that its numbers had rotted, it was that
its **settings** had never executed. A file nobody includes is untested at a configuration nobody has
run, and the configuration is the part that fails first.

---

## Range and identity checks on the pinned goldens

Run mechanically over every pin in the header by `gen_ngtdm3d_pyradiomics.py` (`range_checks`,
`cross_table_checks`), not spot-checked — 35 pinned values across three tables.

| check | scope | result |
|---|---|---|
| `> 0` | the five features | pass |
| `n_i ≥ 1` | both matrix tables (empty levels are dropped, so no row may be empty) | pass |
| `s_i ≥ 0` | both matrix tables | pass |
| levels ascending | both matrix tables | pass |
| `p_i == n_i / N_v,p` | both matrix tables, `rel=1e-12` | pass |
| `Σ n_i == N_v,p` (48 / 16) | both matrix tables | pass |
| `Σ p_i == 1` | both matrix tables, `rel=1e-12` | pass |
| **cross-table**: the five feature pins recomputed from the matrix pins alone | 5 values, `rel=1e-12` | pass |

Both groups were negative-controlled on the plumbing rather than only on the values, because the
failure mode #435 shipped was a bounds check that no longer gated the exit code:

| planted | result |
|---|---|
| `range_checks`: require `Σ p_i == 1.5` | exit 1, 2 failures named, **with every pin verification passing** |
| `cross_table_checks`: scale the derived value by 1.001 | exit 1, all 5 named |

The cross-table check is the one that earns its place. The two tables come from the same PyRadiomics
object but through different attributes, so nothing in the run itself stops one being edited without
the other; since every feature is a contraction of `(i, p_i, s_i)`, the matrix determines all five
scalars exactly. A copy-paste into one table and not the other cannot survive it.

As on 2D GLDZM: these catch a rotted pin, not a wrong definition. They are a complement to running
the oracle, not a substitute — the ±10% and ±100% bands this pass replaced would have passed every
one of them.

---

## Include hygiene

The family's two headers each opened with the same eight includes and the same twenty-line mock 3D
workflow. `test_3d_ngtdm_common.h` now holds the phantom accessor, the settings recipe and that
workflow; both files include it and repeat none of its includes.

What the audit turned up in the originals:

- **`agrees_gt` was never included.** Both files call it and neither reached `test_main_nyxus.h`;
  they compiled because `test_all.cc` includes it earlier in the translation unit. The same is true
  of `gatherRoisMetrics_3D`, `scanTrivialRois_3D`, `allocateTrivialRoisBuffers_3D` and
  `clear_slide_rois`, which are declared in `globals.h` — also reached only transitively.
- **`raw_nifti.h`** was included by the pyradiomics file and no symbol from it was used.
- `Fsettings` and `NyxSetting` came transitively through the feature header, `SimpleCube` and
  `PixIntens` through `image_cube.h` two levels down, `std::tuple` / `std::unordered_set` /
  `std::sort` through headers that happen to include them.

`test_3d_ngtdm_common.h` includes `test_main_nyxus.h` (which supplies `agrees_gt`, `globals.h`,
`roi_cache.h` and `environment.h`), the feature header, `fsystem.h`, and `<string>` / `<tuple>` /
`<vector>` for its own signatures — the shape `test_2d_gldm_common.h` and `test_2d_ngldm_common.h`
already use. Each oracle/regression file adds only what it spells itself.

Applying the reviewer's line (`code-review-818.md` §11 — *drop what a header the file already
includes explicitly supplies, and say so on that include line*) took three more out after the first
pass: `<iostream>` from the oracle and regression files and `<cmath>` from the mechanics file, all
three supplied explicitly by `test_main_nyxus.h` through the common header. `<iomanip>`,
`<algorithm>` and `<unordered_set>` stay, because nothing these files include actually supplies them
— that is the half of the rule that says not to strip what is only reachable by accident. Every
include now carries a trailing comment naming what it is for.

**All four reference tables are `static const`.** `code-review-818.md` §6 flagged new tables declared
`static` without `const`; the two most recently merged families (2D GLDM, 2D GLDZM) are `const`
throughout, and the pre-existing 3D tables are the ones that are not.

## History removed from the headers

Per SPEC and revet §6, the headers state current-state facts. Removed: the `#if 0` copy of an older
regression body, the provenance block naming `ut_inten.nii` as both image and mask, and the sentence
`(100 grey levels, offset 1, and asymmetric cooc matrix)` — a GLCM description, in the NGTDM file,
describing settings this family has no concept of. The recipe, the fixture and the two tools' binning
are recorded instead, and this report carries the rest.

---

## The generic 3D coverage sweep stays

`test_3d_ngtdm_coverage.h` still instantiates both parameterized suites for this family, and its
oracle half calls the same `assert_3d_ngtdm_feature_pyradiomics` the five named tests call — so five
of the sweep's cases duplicate five named assertions, and the family is fully oracle-backed, which is
the condition under which three other families (glcm, morphology, ngldm, glrlm) have retired their
instantiations.

It is deliberately not retired here, for a reason that is not just "open question 7":
`Test3DFeature_WITH_3P_EMBEDDED_GT`'s body is three assertions, and only the third is the duplicate.
The second, `assert_3d_feature_is_registered_and_computable`, checks the feature is registered in the
FeatureManager and computable through the generic path — which none of the five named tests do, since
they call the feature class directly. Retiring the sweep without porting that check would lose it.

So the retirement is a port rather than a delete for this family, the same verdict 3D GLCM and 3D
GLRLM reached for a different reason (a setting their two fixtures disagree on). Recorded rather than
done, because `test_3d_coverage_common.h` is shared with every other 3D family's branch.

## Registry corrections

All five rows were already `status=vetted`, `oracle=pyradiomics`, `agreement=agreed`, and that verdict
survives. What changed:

- `config_recipe` was **empty** on all five → `ngtdm3d.pyradiomics_binwidth1`.
- `tolerance` was **empty** on all five → `rel=1e-9`. An empty tolerance column on a vetted row is
  how a ±10% band goes unnoticed.
- `target_test` named `test_3d_ngtdm_pyradiomics.h`, which is where the assertions already were →
  cleared. `source` `tracker` → `audit`.
- `notes` read `test_3d_ngtdm_regression.h ORPHANED (not #included; never run) - left per decision A`
  on all five → replaced with what backs the row.
- `test_name` and `benchmark` were empty (the columns postdate these rows) →
  `TEST_NYXUS.TEST_3D_NGTDM_<F>_PYRADIOMICS` and `bench_compat_ngtdm_3d`.

No row changes status, and no feature in this family lacks an oracle.

## Reproduction

```
conda run -n nyxus_oracle python tests/vetting/oracles/gen_ngtdm3d_pyradiomics.py
python tests/vetting/audit/scan_ngtdm3d_coverage.py --check
python tests/vetting/check_coverage.py --check
python tests/vetting/check_test_names.py --check
runAllTests --gtest_filter=*3D_NGTDM*
```

Full steps, including how to regenerate each table from scratch: `ngtdm_3d_golden_regen.md`.
