# Nyxus Oracle-Vetting Test Framework — SPEC

**Status.** Specification only. This document defines the framework; it changes nothing in the test
tree. Registry seeding, config recipes, config matrices, oracle generators, and any test
moves/renames are follow-up work that begins **only after this spec is approved and merged** (§9).

**Goal.** Every number Nyxus produces should be *verified against an independent tool* (or a
closed-form value) to close — ideally exact — agreement. Where a Nyxus feature depends on
configuration, we pick the config that makes it directly comparable to a chosen reference tool
(PyRadiomics, ImageJ/FracLac, CellProfiler, MATLAB, scikit-image, SimpleITK/ITK, scipy, or the
IBSI reference tables), and vet it there.

This document defines: the vocabulary, the test taxonomy, the single source of truth for coverage,
naming conventions for tests and benchmark data, tolerance policy, how oracle goldens are produced
and kept honest, and a template + checklist for adding a vetted feature.

---

## 1. Core principle — vetting is a property of an *assertion*, not of a feature

A feature is not "vetted" or "unvetted" in the abstract. Vetting is a property of a specific
**(feature × config × reference)** assertion:

> "Nyxus `GLCM_CONTRAST_AVE`, computed with `ibsi=True, coarse_gray_depth=8`, equals
> PyRadiomics `Contrast` (v3.0.1, symmetricalGLCM=True, binWidth=1) to within 1%."

Consequences:

- **Regression/snapshot tests never claim vetting.** They pin current output to catch drift.
- **Oracle tests are the only place vetting is established**, on the config where the reference is
  valid. A feature counts as vetted iff at least one oracle assertion covers it.
- The same feature may be *vetted on config A and only snapshotted on config B* — that is expected
  and must be legible from names/files, never encoded as contradictory "vetted/unvetted" labels on
  the same value.

---

## 2. Test taxonomy — four kinds, kept in separate files

| Kind | Question it answers | Claims correctness? | File suffix |
|------|---------------------|---------------------|-------------|
| **Oracle** | Does Nyxus match an external tool / closed form? | **Yes** | `_<oracle>` |
| **Regression** | Did Nyxus's output change? | No (drift guard) | `_regression` |
| **Invariant** | Does output obey a required property/bound/relation? | Partial | `_invariant` |
| **Mechanics** | Does the plumbing work (gating, defaults, I/O)? | No | `_mechanics` |

`<oracle>` ∈ the §4 tokens: { `pyradiomics`, `radiomicsj`, `skimage`, `mirp`, `matlab`, `cellprofiler`, `mitk`,
`feature2djava`, `wndcharm`, `imea`, `imagej`, `fraclac`, `pydicom`, `opencv`, `ibsi`, `analytic` }.

A single feature family (e.g. GLCM) typically has: one oracle file per reference that covers it,
plus one regression file. Correctness lives in the oracle files; the regression file is a pure
snapshot.

---

## 3. Single source of truth — the coverage registry (git-tracked)

The registry is **one row per assertion** — a `(feature × config recipe × oracle)` triple — in
`tests/vetting/oracle_coverage.csv`. A feature can (and for confidence should) have several rows: the
same feature vetted by `pyradiomics`, `radiomicsj`, and `feature2djava` is three rows. It will be
seeded from the audit already produced (`.planning/test-vetting-audit/`) and kept under version control.

Columns:

| column | meaning |
|--------|---------|
| `dim` | 2D / 3D / IMQ |
| `feature` | enum name (e.g. `GLCM_CONTRAST_AVE`, `3GLCM_CONTRAST`) |
| `family` | GLCM / GLRLM / IntensityHistogram / FirstOrder / Morphology / Moments / … |
| `oracle` | one of the §4 tokens, or `—` for regression/invariant rows |
| `config_recipe` | id of the config recipe (see §5), e.g. `glcm.ibsi_identity` |
| `outcome` | `vetted` (matches oracle) / `regression` (snapshot only) / `invariant` / `not_tested` / `not_implemented` / `dropped_invalid` |
| `tolerance` | `exact` / `rel=1e-3` / `rel=1e-2` / … |
| `test_file`, `test_name` | where the assertion lives (empty if not yet written) |
| `benchmark` | benchmark-data id (see §6) |
| `source` | where the verdict comes from: `in-tree` (a test in this repo makes the assertion) / `tracker` (an offline harness run not migrated in — `target_test` names where it belongs) / `audit` |
| `notes` | provenance, caveats |

### 3.1 The headline metric — "vetted by ≥1 oracle"

The primary rollup, per feature: **is there at least one row with `outcome=vetted`?** A generated
report (`tests/vetting/coverage_report.md`) shows, per family and overall:

- `# features vetted by ≥1 oracle` / `# real features`  ← the number we drive to 100%
- redundancy: `# features vetted by ≥2 oracles` (higher confidence)
- gaps: features with only `regression` / `not_tested` rows, and which oracle *could* cover them

`tests/vetting/check_coverage.py` (later) both generates this report and flags drift between the
registry and the test tree (e.g. an `outcome=vetted, oracle=pyradiomics` row with no matching test).

The north star: every real (implemented) feature has ≥1 `vetted` row.

---

## 4. Oracle tools catalog

Reference tools we vet against. Versions are recorded in golden **provenance only**; the short
`token` is what appears in test/file names and the registry (versionless, so a tool upgrade does not
churn names).

| token | Tool (version) | Vets (families) | Interface |
|-------|----------------|-----------------|-----------|
| `pyradiomics` | PyRadiomics 3.0.1 | GLCM/GLRLM/GLSZM/GLDM/NGTDM, first-order, shape | Python / Docker |
| `radiomicsj` | RadiomicsJ 2.1.2 | texture, first-order, shape (IBSI-compliant) | Java |
| `skimage` | scikit-image 0.24+ | moments, shape descriptors | Python |
| `matlab` | MATLAB Image Processing Toolbox semantics, run as GNU Octave + `image` (see note) | GLCM (graycomatrix/graycoprops), regionprops (morphology), moments | Octave CLI |
| `cellprofiler` | CellProfiler 4.2.1 | MeasureTexture (Haralick), MeasureObjectSizeShape, intensity, **MeasureObjectNeighbors** (neighbor count / percent-touching / closest-distance) | pipeline |
| `mitk` | MITK 2023.04 | texture, first-order, shape | C++ CLI |
| `feature2djava` | NIST Feature2DJava 1.5.0 (WIPP plugin) | 2D features — sibling implementation, many features 1:1 | Java |
| `wndcharm` | WND-CHARM 1.60 | Haralick, Tamura, Zernike, Gabor, radial, … (Nyxus lineage) | C++ / Python |
| `imea` | imea 0.3.3 | 2D/3D shape / morphology (diameters, elongation, …) | Python |
| `imagej` | ImageJ / Fiji | GLCM texture, shape descriptors, intensity | Java / macro |
| `fraclac` | FracLac (ImageJ plugin) | fractal dimension (box-count) | Java / macro |
| `mirp` | MIRP | texture, first-order, shape (IBSI-compliant) | Python |
| `pydicom` | pydicom 3.0.2 | DICOM decode + `RescaleSlope`/`RescaleIntercept` → true HU (CT) | Python |
| `opencv` | OpenCV 4.13 (`cv2`) | image-quality focus measures (`cv2.Laplacian(...).var()`), filters/convolution | Python |
| `ibsi` | IBSI reference tables | texture + first-order consensus values | hardcoded |
| `analytic` | closed-form / hand-computed | any (on tiny fixtures) | in-test |

Notes:
- **`feature2djava` and `wndcharm` are the highest-value oracles for Nyxus-original features** (the
  ones with no radiomics counterpart — Zernike, Tamura, Gabor, radial, caliper, chords): Nyxus
  descends from WND-CHARM, and Feature2DJava is the WIPP sibling, so agreement can be near-exact.
- **`radiomicsj`, `mitk`, and `mirp`** give second/third/fourth IBSI-compliant opinions alongside
  `pyradiomics` — useful for the "≥1 oracle" goal and for cross-checking pyradiomics itself.
- **Fractal dimension** (`FRACT_DIM_BOXCOUNT`) → `fraclac` (the ImageJ box-count plugin) is the
  reference; already used on the fractal branch. `FRACT_DIM_PERIMETER` (divider/Richardson) has no
  tool → `analytic` + cross-method convergence.
- scikit-image (`skimage`) is an accepted mainstream oracle (added 2026-07; covers image
  moments, shape descriptors). `mahotas`, `DIPlib`, and `Centrosome` are NOT accepted — features
  only they cover remain regression/analytic.
- **A token may cover one stage of a feature rather than all of it, and then the row must say so.**
  `oracle` holds a single token, so a family whose reference supplies only part of the pipeline
  carries the narrowing in `notes` — the same arrangement as `matlab` above, where the token names
  the semantics and every artifact repeats what produced the numbers. 2D `gabor` is the worked case:
  scikit-image supplies the kernel, the WND-CHARM count-ratio score has no scikit-image equivalent
  and is reproduced from Nyxus' own definition, and the `f0=0` kernel is analytic. Such a family
  belongs in `not_covered.md` §E as well, so the coverage percentage is not read as more than it is,
  and its generator must include the **negative control** that measures the gap — for gabor, scoring
  the same feature off `skimage.filters.gabor` moves values by up to 1.005.
- **`opencv`** is the reference for the focus/blur measures Nyxus took from the imaging literature
  rather than from radiomics: `FOCUS_SCORE` is the Pech-Pacheco et al. (2000) variance of the
  Laplacian, which is exactly `cv2.Laplacian(img, cv2.CV_64F, ksize=1).var()`. Prefer it over
  `skimage.filters.laplace`, whose border mode is not settable to zero padding. Note that
  CellProfiler publishes features *named* `FocusScore`/`LocalFocusScore` that are a different
  statistic (normalized variance of the raw image) — same name, not an oracle for these.
- **`pydicom`** is the reference DICOM reader for CT / Hounsfield Units: it decodes the stored
  pixels and applies `RescaleSlope`/`RescaleIntercept` to true HU. It vets the `--preserve-hu` path —
  the loader-pixel values and the first-order HU stats (MIN/MAX/MEAN/INTEGRATED) on a real
  `CT_small.dcm` slice are pinned from pydicom, not recomputed from the Nyxus offset formula, so the
  assertion is a genuine oracle rather than a self-consistency snapshot (§5.2). Offline generator
  only; never a CI runtime dependency (the derived TIFF fixture is committed).
- **`matlab` names the reference semantics, not the vendor product that ran them.** The goldens
  under this token are produced by **GNU Octave + the `image` package**, standing in for MATLAB
  license-free (TOOLS.md; MIGRATION 5.13) — Octave's `regionprops` matches MATLAB's definitions
  on what we vet, including the +1/12 pixel finite-size second-moment correction that decides
  the ellipse triple. So `oracle=matlab` means an Octave run unless a table's provenance header
  says otherwise, and every artifact presenting the token — registry `notes`, the audit
  coverage CSVs, the vetting reports — repeats which tool produced the numbers rather than
  leaving `matlab` to imply a licensed run. `test_3d_firstorder_matlab.h` is a licensed-run
  exception: its provenance pins MATLAB R2026a and its generator calls the named native statistics.
  `graycoprops` is the one gap: Octave's `image` has no equivalent, so GLCM has no Octave path
  and is vetted against `pyradiomics` / `ibsi` instead.
- Flag any family no listed tool covers as we go.

**Rule: reference tools are never CI runtime dependencies.** Goldens are generated *offline* by a
checked-in generator (`tests/vetting/oracles/gen_<family>_<token>.*`) and pinned as literals with
full provenance. CI only builds Nyxus and compares to the pinned goldens.

---

## 5. Config-matching recipes

Because some features are config-sensitive, each oracle assertion names a **config recipe** — the
exact Nyxus settings that make the feature directly comparable to the tool, paired with the tool's
own settings. Recipes are defined once in `tests/vetting/config_recipes.md` and referenced by id.

Example recipe `glcm.pyradiomics.ibsi_identity`:

```
Nyxus:       ibsi=True, coarse_gray_depth=<Ng>, GLCM_OFFSET=1, angles={0,45,90,135}
             (=> symmetric matrix + identity binning)
PyRadiomics: symmetricalGLCM=True, binWidth=1, distances=[1], force2D=True,
             force2Ddimension=0, weightingNorm=None, label=1
Fixture requirement: every grey level 1..Ng present (else PyRadiomics re-indexes levels)
Matched features: ACOR, ASM, CLUPROM, CLUSHADE, CLUTEND, CONTRAST, CORRELATION, DIFAVE,
                  DIFENTRO, DIFVAR, ENTROPY(=JE), HOM2(=IDM), ID, IDM, IDMN, IDN, IV, JAVE,
                  JE, JMAX, INFOMEAS1/2, SUMAVERAGE, SUMENTROPY, VARIANCE
Not matched (no counterpart): DIS, HOM1(dup), JVAR, SUMVARIANCE, ENERGY(dup)
```

A feature that only matches a tool under one recipe is documented as such; on other configs it is
snapshot-only, and that is stated positively ("vetted on recipe X; regression on default"), never
as a bare "unvetted".

### 5.1 Config matrix → many candidate tests → keep valid, drop invalid

A feature family has several config knobs; their cross-product is a **matrix of config points**, and
each point is a *candidate* test. Not every point is worth a test — the framework makes the triage
explicit rather than leaving it implicit in scattered files.

For each config point we classify it:

- **VALID (→ oracle test):** some listed tool, set to its matching config, computes the same
  quantity → assert Nyxus == that tool. Becomes a `_<oracle>` test + a `vetted` registry row.
- **VALID-BUT-PRODUCTION-ONLY (→ regression test):** a real default config that no external tool
  reproduces (a Nyxus convention) → keep as a `_regression` snapshot (drift guard), `regression` row.
- **INVALID (→ drop):** degenerate or nonsensical combination, or one no one runs → dropped, with a
  one-line reason and a `dropped_invalid` row so the decision is recorded, not silently omitted.

Each family gets a `tests/vetting/matrix/<family>.md` enumerating the knobs, the points, and each
point's verdict + oracle. Worked example (GLCM):

| ibsi | symmetric | binning | verdict | oracle / reason |
|------|-----------|---------|---------|-----------------|
| True | True | identity (Ng=distinct levels) | VALID | pyradiomics, radiomicsj, matlab, ibsi — recipe `glcm.ibsi_identity` |
| False | True | binWidth-style (radiomics compat) | VALID | pyradiomics (compat path) — recipe `glcm.radiomics_compat` |
| False | False | matlab, Ng=100 | VALID-prod-only | difference-based subset coincides; rest = Nyxus convention → `test_glcm_regression` |
| any | any | offset/distance = 0 | INVALID | degenerate (dx=dy=0 self-cooc); covered by a `_mechanics` guard, not an oracle |

The point: the number of tests is *derived* from the matrix, and every cell has an explicit
disposition — a reader can see why a given config is (or isn't) an oracle test.

### 5.2 How the matrix is generated

Not a blind cross-product sweep (that explodes and is mostly meaningless). The rule is: **curated
axes + values + oracle mapping (human judgment), verdicts measured empirically (automated).**

1. **Axes = the settings the feature actually reads.** Extract them from the feature's `calculate()`
   — the `STNGS_*` / `NyxSetting`s it consumes. Family-specific and finite (GLCM: `ibsi`,
   `symmetric_glcm`, binning/`GLCM_GREYDEPTH`, `GLCM_OFFSET`/distance, `angles`; first-order reads
   almost none). Settings the feature ignores are not axes → the matrix stays small.

2. **Values = production-meaningful and/or tool-matchable.** Per axis, a small discrete set that is
   either a real production config or settable to match a reference tool — *not* a full numeric sweep
   (`binning ∈ {identity, matlab, radiomics-binWidth}`, `ibsi ∈ {T,F}`, `Ng ∈ {8,64}`; not "Ng
   1..256"). Keeps it to tens of cells.

3. **Oracle mapping (the human part).** Each cell records the *equivalent* config of each candidate
   oracle — the recipe (§5). This encodes definitional knowledge of both tools and cannot be
   auto-derived; it is authored once per (family, tool). A cell may map to several oracles, or none
   (a Nyxus-only convention).

4. **Verdicts are measured, not asserted.** A generator `tests/vetting/oracles/gen_matrix_<family>.*`
   does, per cell: run Nyxus at the cell's config + each mapped oracle at its config on a benchmark,
   record per-feature relative error, then classify each `(cell, feature)`:
   - **VALID(oracle)** — a mapped oracle agrees within tolerance → oracle test + `vetted` row
   - **VALID-prod-only** — a real default no tool reproduces → `regression` row
   - **INVALID** — degenerate (e.g. `offset=0`) or not a real config → `dropped_invalid` row + reason

   It emits `matrix/<family>.md` (the table *with the measured errors*) and candidate registry rows
   for human review.

**Why measured, not hand-labeled.** Hand-labeling is exactly what failed before: `ACOR` was
hand-marked oracle-vetted on `ibsi=False` yet diverges ~43% there; `ENTROPY` sat as a "convention"
when it was a normalization bug. Running Nyxus vs. the oracle at each cell catches both — a cell that
*should* agree by definition but doesn't is either a recipe error or a Nyxus bug, so matrix
generation doubles as a bug finder.

Division of labor: **humans decide what's plausible** (axes, values, mappings); **the harness decides
what's true** (measured agreement). Worked GLCM cells:

| cell (ibsi, symmetric, binning, Ng) | maps to | measured | verdict |
|---|---|---|---|
| (T, T, identity, 8) | pyradiomics / radiomicsj / matlab @ symmetric, binWidth=1 | ASM 1e-10, CONTRAST 5e-10, … | VALID → oracle test |
| (F, F, matlab, 100) | pyradiomics @ symmetric | CONTRAST 0.3%, ACOR **43%**, JE **9%** | difference-subset VALID; rest prod-only |
| (·, ·, ·, offset=0) | — | CONTRAST ≡ 0 | INVALID → `_mechanics` guard |

---

## 6. Naming conventions

### 6.1 Test files
`test_<dim>_<family>_<kind-or-oracle>.{h,py}`

- Oracle: `test_2d_glcm_pyradiomics.py`, `test_2d_glcm_ibsi.h`, `test_3d_glcm_pyradiomics.h`
- Regression: `test_2d_glcm_regression.h`, `test_3d_glszm_regression.h`
- Invariant: `test_2d_morphology_hull_invariant.py`
- Mechanics: `test_2d_glcm_mechanics.h`, `test_3d_nifti_mechanics.h`

**`<dim>` is `2d` or `3d`, and it is mandatory.** Nyxus computes the same family in both
dimensions from different code (`glcm.cpp` vs `3d_glcm.cpp`) against different oracles, so a
name without the token says which family an assertion covers but not which implementation —
and `dim` is the registry's first column precisely because a `(feature × config × oracle)` row
is meaningless without it. A file whose subject genuinely has no image dimensionality — the
Arrow/Parquet writers, environment init, ROI blacklisting, the framework's own self-tests —
takes no token; that absence is the claim that the test is dimension-independent, so it is
checked rather than assumed (see the checker's `DIM_AGNOSTIC` list, each entry with a reason).

Two families sit outside the `2d`/`3d` split: `imq` (image quality) carries `IMQ` in the
registry's `dim` column, and it names its own dimension, so `test_imq_opencv.h` takes no
token. Fixtures, data tables and the gtest translation unit are named by §6.3, not by this
rule — `test_data.h`, `test_dsb2018_data.h`, `test_all.cc`. That exemption covers *fixtures*, not
goldens: a table of oracle values in an exempt file is outside both this rule and §6.3.1, which is
how `gabor_truth` sat unchecked in `test_gabor_truth.h` until the GABOR re-vetting moved it next to
its assertions.

### 6.2 Test functions / gtest cases
`test_<dim>_<family>[_<subject>]_<oracle>` — the dim token says which implementation, the
oracle suffix makes vetting status self-evident:

- `test_2d_glcm_ave_pyradiomics` — 2D GLCM angle-averaged vs PyRadiomics
- `test_3d_glcm_contrast_pyradiomics` — the 3D implementation of the same family
- `test_2d_intensity_histogram_dispersion_analytic` — IH dispersion vs closed form
- `test_2d_morphology_basic_imea` — area/perimeter/diameters vs imea
- `test_2d_glcm_regression` — snapshot drift guard (no oracle)

A function's dim token must match its file's: a `test_3d_*` function in a `test_2d_*` file is
the same category of error as a `_regression` function in an `_ibsi` file.

gtest macro name = uppercased function, prefixed `TEST_NYXUS.` per existing convention.

`assert_*` is the helper prefix, and what makes a function a helper is that a test calls it -
not the prefix. **A nullary `assert_*` invoked directly by a `TEST()` case is that case's test
function and takes the `test_` name**, because every rule above keys on the `test_` prefix: under
`assert_` the function's dim token, kind suffix and file-purity go unchecked, and the
`case == UPPER(function)` mirror falls through to the weak "case ends in a kind token" branch. A
helper that genuinely is one takes arguments - the fixture, the golden table, the config - which
is also what stops a case from calling it directly.

**Enforced, not aspirational.** `tests/vetting/check_test_names.py --check` fails on any
violation of §6.1/§6.2 — dim tokens, file names, function suffixes, helper prefixes, gtest
suite and case names. It runs in CI beside `check_coverage.py` and as a pytest case, with a
second case that plants one defect of each class and requires the checker to report them, so
it cannot decay into a no-op.

### 6.2.1 Which file an assertion belongs in — read the registry columns in this order

`oracle_coverage.csv` carries both a verdict (`status`, `oracle`) and a destination (`target_test`).
They can disagree, because verdicts were corrected over time while `target_test` kept the value it was
seeded with. **The verdict wins:**

The row's `dim` column fixes the file's dim token (§6.1), so only the kind is in question:

1. `status=vetted` with `oracle=X` → the assertion belongs in `test_<dim>_<family>_X.{h,py}`, and the
   function ends in `_X`. A `target_test` naming a `_regression` file for such a row is stale.
2. `status=regression` (no oracle) → `test_<dim>_<family>_regression.{h,py}`, function ends
   `_regression`.
3. `target_test` decides only when it is consistent with 1-2 — then it is the authority on *which*
   file (e.g. which of several oracle files, or a `.py` rather than a `.h`).

Two checks before moving anything, both learned the hard way:

- **Does the destination already assert it?** Often the assertion was migrated in an earlier wave and
  only the registry is stale — then repoint the registry and move no code (Wave 13: all 40 rows).
- **Do the values support the claim?** Moving a pinned Nyxus snapshot into a `_<oracle>` file
  manufactures a vetting claim the numbers do not back. Where the reference tool has no golden in the
  tree at all (MIRP today), the assertion stays where it is and the row is recorded as backlog — a
  rename wave cannot invent an oracle.

### 6.3 Benchmark / fixture data
Names must describe **shape + salient property + provenance**, not opaque indices.

`bench_<shape><size>_<property>` for inline fixtures:
- `bench_dense8_alllevels` — 8×8, contains every grey level 1..8 (for level-sensitive texture)
- `bench_line5` — 5-pixel 1-D line (tiny hand-computable histogram)
- `bench_ramp9` — 9×9 horizontal intensity ramp (directional contrast)

Standard reference phantoms keep their canonical names (`ibsi_digital_phantom`).

Data files: `<modality>_<content>[_<label>].<ext>` — e.g. `mri_liver_seg.nii`,
`ct_lung_labels.nii`. (Rename opaque `compat_int_mri.nii` etc. accordingly.)

Each benchmark has a one-line registry entry in `tests/vetting/benchmarks.md`: id, shape, why it
exists, which recipes/tests use it.

### 6.3.1 Golden reference tables
`<family>_<dim>_<oracle>_ref_vals` — the same three facts as the file name (§6.1), in the same
order, so a table and the file that holds it read alike:

- `gldm_2d_regression_ref_vals`, `glcm_2d_ibsi_ref_vals`, `firstorder_3d_matlab_ref_vals`

Three qualifiers, used only when needed:

| Situation | Suffix | Example |
|---|---|---|
| Several tables share one family/dim/oracle | `_<subject>` before `_ref_vals` | `moments_2d_skimage_shape_ref_vals` |
| Per-feature tolerances beside the values | `_ref_tols` | `firstorder_2d_pyradiomics_ref_tols` |
| Values keyed by ROI label, not feature | `_ref_vals_by_label` | `neighbor_2d_cellprofiler_ref_vals_by_label` |
| One value per direction, not one per feature | `_ref_vals_by_angle` | `glcm_3d_pyradiomics_ref_vals_by_angle` |

**A table lives in the file whose assertions read it.** Its `<family>_<dim>_<oracle>` must therefore
match that file's own `<family>_<dim>_<kind-or-oracle>` (§6.1) — the table and its file make the same
claim, or one of them is wrong. `morphology_2d_fraclac_ref_vals` belongs in
`test_2d_morphology_fraclac.h`, beside the assertions that read it.

**`_common.h` headers hold fixtures, never values.** Settings builders, ROI loaders, feature-calculation
scaffolding — anything a test needs in order to *produce* a number — are shared freely. Reference data
and the assertions that compare against it are not, because a shared table is read by callers the table
cannot see. That is not hypothetical: one shared lookup spanning four tables let `_matlab` and
`_skimage` functions resolve their "oracle" values out of a `_regression` snapshot, which is precisely
the §6.2.1 defect, reached through file layout instead of through naming. Keeping values next to their
assertions makes the mismatch visible in one file rather than inferable across three.

**A family's header is included only from that family.** `test_<dim>_<family>_common.h` holds that
family's fixtures. Including it from a second family is not free: it puts a family's file in the
include graph of assertions that never name it, so a fixture edit lands on tests its own family
cannot see, and reaching one ROI builder drags in every feature header its neighbours needed. It is
also the road back to the defect above, since a header read from two families is exactly the scope
in which a shared table stops being readable in one place.

Two families wanting the same fixture is usually a fixture doing too much, so narrow it first: a
family's fixture computes what that family asserts, and the prerequisites those features actually
read. Radial needs the contour before `RadialDistributionFeature`; Zernike needs nothing but the
loaded ROI. Each builds it in its own file, in a few lines, and pins the same goldens as the
fifteen-feature pass they used to borrow. What survives narrowing is usually configuration rather
than computation — here `make_shape2d_settings()`, the config every 2D shape golden is measured
under — and that goes to `test_main_nyxus.h`, the harness header every test already includes, so it
keeps one definition. A header that belongs to no family — `test_3d_coverage_common.h`, named for
the `_coverage` kind, or `test_feature_calculation_common.h`, used from both dims — is listed in
`FAMILY_NEUTRAL` with its reason, the way `DIM_AGNOSTIC` carries "this file has no dimension".

**Enforced.** `check_test_names.py` rejects any `#include "test_…"` naming a family other than the
including file's. `test_all.cc` is the one exemption: it is the translation unit, so including every
assertion header is its job.

An assertion helper is bound to the table it reads, so it moves with it. A helper serving several
oracle files is the shape to look for: it means one table is answering to several `<oracle>` claims,
and the fix is to split the helper per oracle, not to relocate the table.

**A table is `const`.** Reference data is read-only by definition, and here `const` does more than
document that: without it `operator[]` compiles, and on a key the table does not hold it
*default-inserts* 0 instead of failing. The assertion then compares against a golden that does not
exist and passes whenever the computed value is also 0 — which is exactly the case a golden table is
there to catch, a feature that silently produced nothing. The phantom key also survives for the rest
of the run, so a later `.count()` guard on it succeeds. `const` makes `operator[]` a compile error
and forces `.at()`, which throws naming the key it could not find. Read a table with `.at()`; use
`.count()` first only where a clearer message is wanted. A `ref_vals_list` is a `std::vector`, whose
`operator[]` reads rather than inserts, so an index into one is not part of this rule.

A table's name is a claim, so `_regression` in it means the numbers are pinned Nyxus output and
establish no vetting, exactly as for test functions (§1). It follows that **one table holds one
oracle**: a table mixing imea-vetted keys with snapshot keys has no honest `<oracle>` to put in its
name, and the fix is to split it, not to pick the majority.

**Enforced.** `check_test_names.py` applies this to every file-scope table whose name contains
`golden`, `oracle`, `reference`, `_gt`, `ref_vals` or `ref_tols`, and to every table declared through
a §6.3.1 alias or reached through an accessor returning a function-local static — a set that
deliberately includes the conforming suffixes, so a table cannot drift back to an old name without
tripping the check. It rejects a missing suffix, a missing dim token, an `<oracle>` segment that is
not a §4 token (`glcm_2d_mahotas_ref_vals` fails, since the registry does not accept mahotas), a
reference table declared in a `_common.h`, a reference table declared without `const`, and a
reference table declared as a raw `std::unordered_map` rather than through an alias. That last one is what keeps the set closed: a
table identified only by the words in its name is one rename away from being invisible to the check,
whereas `ref_vals_map<double>` says what it is regardless. One value type, `double`, for the same
reason — `agrees_gt`/`ASSERT_NEAR` compare in double, so a `float` table narrowed its literals only
to widen them again at the assertion. Tables that do not conform yet live in `TABLE_EXCEPTIONS`
with the reason, the same way `KIND_EXCEPTIONS` and `DIM_AGNOSTIC` carry theirs; an entry there is a
piece of tracked work, not a permanent waiver.

### 6.4 Golden provenance (mandatory)
Every pinned oracle golden must record, at its definition site: **tool + version + exact config +
generator script path**. Generators live in `tests/vetting/oracles/gen_<family>_<oracle>.py` and are
runnable to regenerate the numbers.

---

## 7. Tolerance policy

| Situation | Tolerance | Rationale |
|-----------|-----------|-----------|
| Analytic / closed form | `exact` (abs 1e-9) | no estimator disagreement |
| Same-definition oracle, same binning | `rel=1e-3` | float + aggregation-order only |
| Cross-tool, definitional edge differences (binning edges, angle aggregation) | `rel=1e-2` | documented residual |
| Known method divergence (e.g. box-count vs divider fractal) | stated band + citation | not a bug |

Every non-exact tolerance carries a one-line reason. A tolerance loose enough to pass a known-bad
value is a bug in the test.

---

## 8. Authoring checklist (new vetted feature)

1. Pick the reference tool and the **config recipe** that makes Nyxus comparable (§5).
2. Pick/define a **benchmark** whose properties actually exercise the feature (§6.3).
3. Generate goldens **offline** with the checked-in generator; record tool version + config.
4. Write the assertion in the correct **oracle file**, named per §6, at the tolerance per §7.
5. If the feature is also snapshotted elsewhere, ensure that file is labeled regression-only.
6. Update `oracle_coverage.csv` (status → `vetted`, oracle, recipe, tolerance, test, benchmark).
7. Verify in CI (build Nyxus, compare to pinned goldens).

---

## 9. Decisions & rollout

**Settled:**
- Home directory: `tests/vetting/` (spec + registry + recipes + matrices + generators).
- Registry format: **CSV** (diff-friendly).
- Naming scheme (§6) and renaming of existing tests (`test_ibsi_*`, `test_compat_3d_*`,
  `test_glcm.h`, …): **approved**.

**Deferred:**
- *How* each tool is run to generate goldens (Docker / venv / Java / MATLAB / pipeline / macro):
  TBD — decided per tool, before its first use.

**Rollout — begins only after this spec is approved and merged:**
1. Scaffold `tests/vetting/`: seed `oracle_coverage.csv` from the audit (`.planning/test-vetting-
   audit/`), add `config_recipes.md`, `matrix/<family>.md`, `oracles/gen_*`, `check_coverage.py`,
   and the generated `coverage_report.md`.
2. **GLCM** as the worked reference migration (oracle + regression + mechanics + matrix + registry
   rows), applying the naming/renames.
3. Roll the pattern out family by family, driving *features vetted by ≥1 oracle* toward 100%.

Nothing in this list is done in this PR — it is the plan the merged spec authorizes.
