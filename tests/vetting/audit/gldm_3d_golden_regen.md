# 3D GLDM — regenerating the goldens

Two tables, two recipes, two procedures. Neither runs in CI: reference tools are never a CI runtime
dependency (SPEC §4), so both are offline steps whose output is pasted into the header.

---

## 1. The PyRadiomics oracle table (`gldm_3d_pyradiomics_ref_vals` and the two matrix tables)

**Environment.** conda env `nyxus_oracle` — pyradiomics 3.0.1 needs Python ≤ 3.9; this one is 3.8.20
with SimpleITK 2.3.1 and numpy 1.23.5. Setup is in `tests/vetting/TOOLS.md`.

**Run, from the repository root:**

```
python tests/vetting/oracles/gen_gldm3d_pyradiomics.py
```

It prints three paste-ready blocks — the fourteen scalars, the phantom's 163 matrix cells, and the
4×4×3 volume's 9 cells — and then re-verifies everything already pinned in
`tests/test_3d_gldm_pyradiomics.h`, exiting non-zero on any mismatch, any pin it cannot produce, any
bound or identity violation, any cross-table disagreement, or any feature the oracle produces that
the header pins nothing for.

**Fixture.** `bench_compat_liver_3d` — `tests/data/nifti/compat_int/compat_int_mri.nii` +
`compat_seg/compat_seg_liver.nii`, label 1. Read from `__file__`; nothing is downloaded.

**Name mapping.** Strip the `3GLDM_` prefix and expand the abbreviation to PyRadiomics'
`original_gldm_<CamelCase>` — `DE` → `DependenceEntropy`, `SDLGLE` →
`SmallDependenceLowGrayLevelEmphasis`, and so on. The full table is `PYRAD` in the generator. All
fourteen map one-to-one. PyRadiomics additionally exposes `GrayLevelNonUniformityNormalized` and
`DependencePercentage`, both deprecated there because `Nz == Np` makes them degenerate; Nyxus
implements neither, so nothing is unmapped in either direction.

### Convention differences a regenerator will otherwise trip on

1. **Both matrix axes are compacted.** `RadiomicsGLDM._calculateMatrix()` deletes the rows of grey
   levels absent from the ROI *and* the columns of dependences no voxel has, keeping the survivors in
   `coefficients["ivector"]` and `coefficients["jvector"]`. **Read both axes from those vectors, never
   from the array index.** Nyxus keeps its matrix dense, so the two line up cell for cell only once
   the values are taken from the vectors. This is the same hazard `gen_glszm3d_pyradiomics.py`
   documents for `jvector`, except GLDM compacts *both* directions.
2. **Dependence is offset by one on both sides.** PyRadiomics puts a voxel with no dependent
   neighbour in column `j = 1`; Nyxus starts `nd = 1` in `3d_gldm.cpp`. Do not "correct" for a shift.
3. **`Np` must be counted off the mask**, not summed out of the matrix, or the `Nz == Np` identity
   becomes a tautology and stops being evidence about the ROI boundary.
4. **`GREYDEPTH` is inert for this family.** `D3_GLDM_feature::calculate()` reads `GLDM_GREYDEPTH`
   and `IBSI` only. The recipe records `GREYDEPTH=100` because the settings vector carries it, not
   because it does anything.
5. **The matrix and the scalars come from two different entry points.** `run()` goes through the
   public `RadiomicsFeatureExtractor`; `phantom_matrix()` reproduces its preprocessing
   (`checkMask`, then `cropToTumorMask`) to reach the class directly, because the extractor does not
   hand the matrix back. Nothing guarantees those stay in step — the cross-table check is what does,
   by requiring the fourteen scalars to fall out of the intercepted matrix.

### If a pin changes

A changed *scalar* means Nyxus moved, PyRadiomics moved, or the recipe drifted. Establish which
before re-pinning: the generator's `--` output shows the fresh oracle value beside the pinned one, and
`TEST_3D_GLDM_DUMP_PYRADIOMICS` shows Nyxus' own value beside the pin with the residual. A pin is
PyRadiomics' output, so it must round-trip essentially exactly (`PIN_ROUNDTRIP_RELTOL = 1e-12`); that
constant is **not** the band the C++ asserts at, which is set from the Nyxus-vs-PyRadiomics
measurement and lives in the header.

---

## 2. The snapshot table (`gldm_3d_regression_ref_vals`)

These are Nyxus' own values and establish no vetting (SPEC §1). Regenerate from inside the test
binary, not from a tool:

```
runAllTests --gtest_filter=*3D_GLDM_DUMP_REGRESSION*
```

It prints each feature at `setprecision(17)` in the shape the table wants, beside the current pin.
Paste the block over `gldm_3d_regression_ref_vals`.

**Recipe `gldm3d.regression_ut_phantom`** — `bench_ut57_3d`
(`tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57) at `GREYDEPTH=64`,
`IBSI=false`, `GLDM_GREYDEPTH=64`. The ROI extracts as an 82×96×70 cube with intensity range
[1024, 3024].

**`GLDM_GREYDEPTH`'s sign is the scheme, not just the size.** Positive is MATLAB-style level-count
binning, negative is PyRadiomics-style bin counting, and **0 is the IBSI no-binning path**. So this
recipe and `gldm3d.pyradiomics_bincount20` describe different quantities on different phantoms, and a
value from one says nothing about the other.

### Why this table was regenerated rather than carried forward

The previous table could not be reproduced at any configuration, and the pattern of the failure is
worth recording because it identifies *which* axis moved:

| feature | old pin | `GLDM_GREYDEPTH=64` | `GLDM_GREYDEPTH=0` |
|---|---|---|---|
| `3GLDM_HGLE` | 1957.2 | 1957.1946 | 4275550.79 |
| `3GLDM_GLN` | 6481.0 | 6480.830 | 157.97 |
| `3GLDM_GLV` | 153.1 | 153.0947 | 341996.30 |
| `3GLDM_DE` | 8.4 | 8.40347 | 11.48909 |
| `3GLDM_SDE` | 0.26 | 0.15361 | 0.86996 |
| `3GLDM_LDE` | 34.77 | 40.6396 | 2.17252 |
| `3GLDM_DV` | 13.6 | 14.616 | 0.61191 |

At `GLDM_GREYDEPTH=0` nothing is close. At `GLDM_GREYDEPTH=64` the **grey-level axis** reproduces to
four or five digits while every quantity touching the **dependence axis** does not — `SDE` 1.69×,
`LDE` 0.86×, `DV` 0.93×, and all four grey-by-dependence cross-terms. That split says the old table
came from a run whose dependence distribution differed from today's, not from a rounding artefact. It
is not recoverable from settings, so it was re-measured.

**A regenerator should expect this shape.** If a future table half-matches, compare axis by axis
before assuming a rounding difference: a marginal that survives while the joint distribution moves
localises the change.

### Two defects the old table carried

Both were in a file that **no `#include` reached**, so none of it had ever executed. The file is
wired into `test_all.cc` now and all fourteen functions run.

1. **`3GLDM_LGLE` was pinned to `3GLDM_SDE`'s value** — `0.26` in both slots, against a measured
   `0.00073572128237550161`. A factor of **353**.
2. **`test_3d_gldm_lgle_regression()` asserted `Feature3D::GLDM_SDE`**, so `3GLDM_LGLE` had no drift
   guard at all and `3GLDM_SDE` had two.

Both halves of one paste. `scan_gldm3d_coverage.py --check` now catches this class directly: it
diffs each table's keys against the functions that read them and those against the `TEST()`
registrations, and reports an unread key, a name/argument mismatch, and a duplicated assertion
separately. Run against the pre-fix file it names all three.

---

## Verifying a regeneration

1. `python tests/vetting/oracles/gen_gldm3d_pyradiomics.py` → `ALL CHECKS PASSED`, exit 0.
2. `python tests/vetting/audit/scan_gldm3d_coverage.py --check` → `clean`, exit 0.
3. `python tests/vetting/check_coverage.py --check` and `check_test_names.py --check` → exit 0.
4. `runAllTests --gtest_filter=*3D_GLDM*` → all cases pass.
5. Full `runAllTests` → the total moves only by cases deliberately added or removed.
