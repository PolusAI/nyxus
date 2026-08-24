# Oracle tools — local setup & coverage (research findings)

How to stand up each oracle tool locally (Docker or Python venv; MATLAB excluded — license). One
research pass per tool; see per-tool detail below and the setup matrix first.

## Setup matrix

| token | version | setup | feasibility | one-line |
|-------|---------|-------|:-----------:|----------|
| `pyradiomics` | 3.0.1 | **Docker** `radiomics/pyradiomics:latest` (pin by `@sha256`) | high | already in use; pip blocked on Py3.11 |
| `mirp` | 2.6.0 | **conda** `conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy` (venv/pip also works) | high | no Docker needed; full IBSI incl. NGLDM/GLDZM/GLSZM |
| `imea` | 0.3.3 | **venv** `pip install imea==0.3.3` + `numpy<1.24` | high (2D only) | 2D morphology; **3D is heightmap, not voxel — unusable for Nyxus 3D** |
| `feature2djava` | 1.5.0 | **Docker** `wipp/wipp-feature2djava-plugin:1.5.0` (125 MB, exists) | high | NIST/WIPP sibling; intensity + basic shape + Haralick |
| `radiomicsj` | **2.1.3 / 2.1.18** | **Docker/jar** (Maven → shaded uber-jar; Java 11) | high | **2.1.2 does not exist**; Maven Central only; full IBSI + fractal |
| `cellprofiler` | 4.2.8 | **conda** (see below; Docker `cellprofiler/cellprofiler:4.2.1` also works) | high | run the module in-process, headless; needs `JDK_HOME`/`JAVA_HOME` |
| `wndcharm` | 1.60 | **Docker (custom build)** Ubuntu 18.04 + Py2.7 + swig | high | no pip/no image; Nyxus-lineage: Haralick/Tamura/Zernike/Gabor/Chebyshev/radial |
| `imagej` | pinned tarball | **download** Fiji `ImageJ-linux64 --headless` | med-high | morphology/intensity/GLCM headless (GLCM via batch wrapper) |
| `fraclac` | — | ImageJ plugin (GUI) **+ headless-macro reimpl** | med-high* | plugin is GUI-only, but its shifting-grid method runs headless via our macro (*see reconciliation) |
| `mitk` | 2023.04 | **build-once Docker** (`ClassificationCmdApps` config) | med | no prebuilt image; ~2–3 h one-time CLI-only build → reusable pinned image |
| `pydicom` | 3.0.2 | **venv** `pip install pydicom` (pure-Python) | high | DICOM decode + `Rescale*` → HU; offline fixture/golden gen for `--preserve-hu` (CT) |
| `octave` | 10.3.0 / 11.3.0 | **conda** `conda install -c conda-forge octave octave-statistics` (no Docker, no license) | high | license-free MATLAB substitute (MIGRATION 5.13); `mean`/`median`/`std`/`var`/`skewness`/`kurtosis`/`prctile`/`quantile` all present. Add `octave-image` for `regionprops`/`bweuler` (2D morphology) |
| `skimage` | 0.26.0 | **conda** — already present in `nyxus_mirp` (mirp pulls scikit-image, scipy, numpy) | high | no separate env needed; `gabor_kernel` + `regionprops` + `moments*`; see the skimage note below |

## conda route (verified 2026-08, 2D GLCM vetting)

Both texture oracles resolve from conda-forge, which is less setup than Docker and puts the tool a
`conda run` away from the repo. Used to generate every golden in `test_2d_glcm_{pyradiomics,mirp}.h`.

```bash
conda create -n nyxus_oracle -c conda-forge python=3.9  pyradiomics simpleitk numpy   # -> v3.0.1
conda create -n nyxus_mirp   -c conda-forge python=3.11 mirp numpy                    # -> 2.6.0

conda run -n nyxus_oracle python tests/vetting/oracles/gen_glcm_pyradiomics.py
conda run -n nyxus_mirp   python tests/vetting/oracles/gen_glcm_mirp.py
```

Gotchas hit while doing it:

- **pyradiomics needs Python <= 3.9** on conda-forge (the pip-on-3.11 block noted in the matrix is
  the same wall); ask for the interpreter explicitly or the solver picks a newer one and fails.
- **mirp exposes no `__version__`.** Pin it with `importlib.metadata.version("mirp")` - a
  `getattr(mirp, "__version__", ...)` fallback silently writes "unknown" into the provenance line.
- **mirp logs at INFO onto stdout**, interleaving progress lines with the golden table it prints.
  `logging.disable(logging.INFO)` before the call; setting a level on the root logger does not work
  because mirp configures its own logger during the run.
- **pyradiomics prints a warning per run** ("GLCM is symmetrical, therefore Sum Average = 2 x Joint
  Average") on stderr - expected, not an error.
- **pyradiomics GLCM names do not line up with the Nyxus ones, and three actively mislead**:
  `3GLCM_ASM` is PyRadiomics' **JointEnergy** (not `Energy`), `3GLCM_JVAR` is its **SumSquares** (not
  a variance-named feature), and `3GLCM_JMAX` is **MaximumProbability**. Map by definition, not by
  name. `MCC` has no Nyxus counterpart; `Dissimilarity` is deprecated as equivalent to
  `DifferenceAverage`.
- **PyRadiomics reports one value per feature over its whole direction set.** For 3D that is the
  Nyxus `*_AVE` aggregation over the 13 angles, not a per-angle value - so a PyRadiomics golden is
  the reference for the `*_AVE` feature, and only reaches the per-angle base feature through
  `calc_ave`.
- **The inverse-difference family is version-sensitive.** Re-running pyradiomics 3.0.1 against 3D
  GLCM goldens that had been pinned by an earlier version reproduced 19 of 23 bit-for-bit but moved
  `Id` 5.5%, `Idm` 7.8%, `Idn` 7.7% and `Correlation` 0.6%. If a golden in that group disagrees by a
  few percent, suspect the tool version before suspecting Nyxus - and record the version, which is
  why `gen_glcm3d_pyradiomics.py` prints `radiomics.__version__` into the table header.
- **A per-family PyRadiomics class needs `force2D=True` on a single-slice volume.** Driving one
  feature class directly (`radiomics.gldm.RadiomicsGLDM(image, mask, ...)`) rather than the
  extractor is the cheapest way to get one family's values, but without `force2D=True` +
  `force2Ddimension=0` it applies its 3D kernel to a `(1, y, x)` array and reaches for neighbours
  that are not there. The symptom is plausible-looking numbers, not an error.
- Feeding either tool a numpy array is enough - no file on disk. PyRadiomics wants
  `sitk.GetImageFromArray` with an explicit `SetSpacing`; MIRP takes `image=`/`mask=` arrays
  directly, shaped `(z, y, x)`.
- **mirp's column suffixes differ per family, so match the stem exactly.** NGLDM columns carry
  their configuration (`ngl_lde_d1_a0.0_2d`) and have to be matched with a `startswith(stem +
  "_d1")`; GLDZM columns carry none (`dzm_sde_2d`), because a zone distance is a property of the
  ROI mask rather than a setting. A generator copied from the NGLDM one and left on the `_d1`
  prefix match finds no column and raises rather than silently mispinning, but the fix is not
  obvious from the error.
- **mirp's per-family numeric settings must be floats.** `ngldm_distance=1` raises
  `TypeError: The ngldm_distance parameter is expected to contain floating point values of 1.0 or
  greater`; pass `1.0`. Same for `ngldm_difference_level`. The message names the parameter, so the
  failure is at least self-explanatory - but it happens during settings construction, before any
  image is read, which makes it look like an import or environment problem.
- **mirp's NGLDM column names do not track the Nyxus abbreviations.** It writes `lge`/`hge` where
  Nyxus writes `LGLE`/`HGLE`, and `perc`/`entr`/`energy` where Nyxus writes `P`/`ENT`/`ENE`
  (`ngl_ldlge` = `NGLDM_LDLGLE`, `ngl_dc_perc` = `NGLDM_DCP`). Map by meaning, not by string
  similarity - the full table is in `audit/ngldm_2d_golden_regen.md`.
- **The same applies to GLSZM**, under a different set of substitutions: mirp calls the family
  `szm` and spells "zone size" as `zs` where Nyxus writes `SZN`/`ZV`/`ZE` (`szm_zsnu` =
  `GLSZM_SZN`, `szm_zs_entr` = `GLSZM_ZE`), and abbreviates the emphases as `lge`/`hge` again
  (`szm_szlge` = `GLSZM_SALGLE`). Under `by_slice=True` every column is suffixed `_2d`. Full table
  in `audit/glszm_2d_golden_regen.md`.
- **PyRadiomics' `binCount` maps to the Nyxus setting of the family being matched, not the generic
  one.** The 3D texture calculators bin on `GLCM_GREYDEPTH` / `GLRLM_GREYDEPTH` / ... and ignore
  `GREYDEPTH` entirely; a negative value activates radiomics binCount-based binning, so
  `binCount: 20` is `<FAMILY>_GREYDEPTH = -20`. Left unset the per-family setting defaults to 0,
  which means **no binning at all** rather than a default depth - the features then run on raw
  intensities and are silently off by orders of magnitude, not by a few percent. A grey-level
  emphasis feature far above what the bin count allows (3D GLRLM `HGLRE` at 4.3e6 against a ~4e3
  ceiling) is the symptom.

## skimage — no env of its own (verified 2026-08, 2D GABOR vetting)

`scikit-image` needs no setup step: `nyxus_mirp` already carries it (0.26.0, with scipy 1.17.1 and
numpy 2.4.6) because mirp depends on it, so `conda run -n nyxus_mirp python …` runs any skimage
generator. Check the version before pinning goldens — it is a transitive dependency, so it moves
when mirp is updated, and the version belongs in the provenance line either way.

Gotchas hit while vetting GABOR against it:

- **`gabor_kernel(frequency=0)` is undefined.** It sizes its own support from the frequency, so a
  zero-frequency filter has to be supplied in closed form (at f0 = 0 the envelope and the carrier
  are both identically 1, i.e. a flat window).
- **`gabor_kernel` and `skimage.filters.gabor()` are not interchangeable as oracles — measured.**
  The kernel function gives you the filter and lets you convolve it the way the feature under test
  does; `gabor()` additionally imposes skimage's own border handling (`mode="reflect"`) and its
  untruncated kernel support. Scoring Nyxus' GABOR off `gabor()` instead moves values by up to
  **1.0**, the whole range of the feature, because an aspect ratio like Nyxus' `gamma=0.1` makes the
  analytic kernel hundreds of pixels wide (369×113 at the lowest frequency) while the feature crops
  it to 16×16 — only 7.9% of the kernel's L1 mass survives there. Run both legs before describing a
  filter feature as "vetted against skimage": which convolution the goldens assume is the claim.
- **A count-based feature hides small kernel differences.** GABOR scores a pixel count over a
  threshold, so it only moves in steps of `1/baseline_count`. Print that step (generator part C)
  before believing that an exact match proves the filters agree to machine precision — it does not,
  and the honest claim is the one the tolerance can support.

## CellProfiler headless on Windows (verified 2026-08, 2D neighbour and 2D radial vetting)

`cellprofiler` resolves from conda-forge with its own Python, so no Docker image is needed for the
two modules this tree uses as oracles -- `MeasureObjectNeighbors` and
`MeasureObjectIntensityDistribution`:

```bash
conda create -n nyxus_cellprofiler -c conda-forge python=3.9 cellprofiler=4.2.8
conda run -n nyxus_cellprofiler python tests/vetting/oracles/gen_neighbor_cellprofiler.py
conda run -n nyxus_cellprofiler python tests/vetting/oracles/gen_radial_cellprofiler.py
```

Three things stop the run before any module executes, none of them with a useful message:

- **`JDK_HOME` and `JAVA_HOME` must be set to `<env>\Library\lib\jvm`.** Without them the process
  exits **127** with no traceback at all -- CellProfiler starts a JVM through `python-javabridge`
  during import.
- **The working directory must be on the same drive as the installed package.** Importing any module
  under `cellprofiler.modules` pulls in `cellprofiler.gui.help.content`, which calls
  `os.path.relpath` between the package directory and the cwd. Across drives that raises
  `ValueError: path is on mount 'C:', start on mount 'D:'` at import time. `cd` to any directory on
  the package's drive and pass absolute paths to the script.
- **`create_settings()` has already added one object group and one bin group.** Calling
  `add_object()` / `add_bin_count()` yourself leaves a second group whose `object_name` is still
  `"None"`, and the run dies with `KeyError: 'None'` from the object set. Configure
  `module.objects[0]` and `module.bin_counts[0]` in place.

Driving a module takes no pipeline file: build `cellprofiler_core.object.Objects` with a label
array, an `ImageSetList().get_image_set(0)` with a `cpi.Image` in `[0, 1]`, and call
`module.run(Workspace(...))`. `MeasureObjectIntensityDistribution`'s measurements come back as
`RadialDistribution_<Stat>_<image>_<bin>of<n>` with 1-based bin numbers.

**Its centre rule is a maximum and can tie.** The centre is the pixel of maximum distance-to-edge
(`centrosome.cpmorphology.distance_to_edge` then `maximum_position_of_labels`, which is
`scipy.ndimage.maximum_position` underneath). When several pixels attain that maximum -- 8 of 26 on
the `shape2d_morphology` fixture -- which one comes back depends on the label image's shape, so
CellProfiler's own answer changes when you change the padding around the ROI. Report the tie set from
the generator rather than assuming the centre is well defined.

## centrosome as a Zernike cross-check (verified 2026-08-20, 2D zernike vetting)

`centrosome` arrives with the CellProfiler env and exposes the Zernike machinery directly, without
going through a module or a workspace:

```python
from centrosome.zernike import construct_zernike_polynomials, get_zernike_indexes
idx = get_zernike_indexes(10)                     # (n, m) for n < 10 -> 30 pairs
Z   = construct_zernike_polynomials(X, Y, idx)    # complex Z_nm at each (x, y)
```

Two things make it usable as an independent reference rather than a second opinion from the same
tool:

- **It takes the geometry as an argument.** `X` and `Y` are offsets already divided by whatever
  radius you choose, so the polynomials can be evaluated at *another* implementation's disk. That is
  what separates "does this recurrence compute R_nm" from "do these two tools agree about where the
  disk is" -- and those are different questions with different answers.
- **Its angle convention is `z = y + i*x`**, i.e. `atan2(x, y)`, which is the usual `atan2(y, x)`
  reflected. Magnitudes are unaffected: the difference is a constant unit-modulus factor per `(n,m)`,
  so `|sum I * Z|` is identical. Phases are not.

`get_zernike_indexes(limit)` emits n ascending then m ascending with `n - m` even, the same order
`zernike.cpp` walks -- worth asserting rather than assuming, which the generator does.

`construct_zernike_polynomials` zeroes `r > 1` itself but keeps `r == 0`, where a caller that skips
`r < DBL_EPSILON` would drop a pixel. It only matters when a pixel sits exactly on the centre.

## Corrections / notable findings

- **`cellprofiler` runs in-process from a conda env, no Docker.** Used for the 2D neighbour family
  (`gen_neighbor_cellprofiler.py`). A full `pip install cellprofiler` is unbuildable on Windows --
  it hard-pins `mysqlclient==1.4.6`, which has no py39 wheel and needs the MySQL C headers -- so
  install the pieces instead:
  ```
  conda create -n nyxus_cellprofiler python=3.9
  conda install -c conda-forge "openjdk=11" "numpy=1.26" cython setuptools wheel
  pip install --no-build-isolation python-javabridge cellprofiler-core centrosome
  pip install --no-deps "cellprofiler==4.2.8"      # the MODULE package, without the bad pins
  ```
  `numpy<2` is not optional: python-javabridge's C extension uses the numpy 1.x C API and fails to
  build against 2.0. The measurement modules live in `cellprofiler`, not `cellprofiler-core`, which
  is why the last line is needed at all.
  Three gotchas when running it:
  - **`JDK_HOME` and `JAVA_HOME` must point at `<env>\Library\lib\jvm`** or javabridge cannot load
    the JVM and the process dies with a bare **exit 127** and no Python traceback -- it reads as a
    missing interpreter, not a missing JVM.
  - Call `cellprofiler_core.preferences.set_headless()` **before** constructing `Measurements()`,
    or it imports wx for config and fails.
  - `Objects().segmented` is indexed `[row=y, col=x]`, and CellProfiler treats border-touching
    objects specially -- pad the label image (3 px is enough) or the edge ROIs measure differently.


- **`mirp` runs fine from conda-forge**, which is simpler than a venv on Windows because it brings
  its own Python: `conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`. Two gotchas
  when scripting it, both hit while vetting the 2D intensity histogram:
  - MIRP logs at INFO **onto stdout**, interleaving progress lines with whatever your generator
    prints. `logging.basicConfig(level=...)` is not enough — MIRP configures its own logger during
    the run — so call `logging.disable(logging.INFO)` *before* `import mirp`.
  - `extract_features()` returns **every** family in one frame, and every column is suffixed with
    the discretisation it was computed at (`ih_mean_fbn_n6`, `cm_contrast_d1_3d_v_mrg_fbn_n6`).
    Filter on the suffix and map by name, so changing the bin count cannot silently read a column
    computed at the old one.
  - **The suffix encodes the neighbourhood too, not only the discretisation.** NGLDM columns come
    back as `ngl_<feature>_d1_a0.0_3d_fbn_n64` — `d1` is the distance and `a0.0` the difference
    level (alpha), i.e. the IBSI coarseness parameters, ahead of the `_3d_fbn_n64` part. Match on
    the stem and assert the full suffix separately (`gen_ngldm3d_mirp.py` does), or a generator that
    "found its column" may be reading a different neighbourhood than the one it configured.
  - The env has **no NIfTI reader** — neither SimpleITK nor nibabel — so a 3D generator that loads a
    `.nii` phantom appears to need two envs and an intermediate `.npy`. It does not: the phantoms
    under `tests/data/nifti/phantoms/` are uncompressed single-file NIfTI-1 (`magic == b"n+1"`), and
    the header is 20 lines of `numpy.frombuffer` against a frozen on-disk format — `dim[8]` at offset
    40, `datatype` at 70, `pixdim[8]` at 76, `vox_offset` at 108, voxels x-fastest so reshape to
    `(z,y,x)`. `oracles/gen_morphology3d_mirp.py` does exactly that and stays single-env. Worth
    copying for any other MIRP generator on a 3D phantom.
  - For a **morphology** run, pass `base_discretisation_method="none"`: the family is computed from
    mask geometry, so a bin count would be meaningless, and MIRP then leaves the `morph_*` columns
    unsuffixed.
- **Octave `quantile()` takes a method number 1..9** (`quantile(v, p, m)`), and `prctile()` is
  method 5 — useful when checking whether a Nyxus percentile matches a tool's *native* percentile
  rather than a reimplementation of Nyxus' own. See
  `audit/intensity_histogram_2d_analytic_vetting_report.md` for why that distinction matters.
- **`skimage.measure.moments_normalized` answers the CENTRAL-moment question only.** Nyxus also
  exposes normalized *raw* moments (`NORM_SPAT_MOMENT_pq`, `IMOM_NRM_pq`) = `m_pq /
  m_00^((p+q)/2+1)`, for which skimage has no native function. Do not vet one against the other:
  on the 48x40 moments fixture they are 0.3876 vs 0.0999 at p=2,q=0. Build the normalized raw value
  from `moments()` and skimage's own exponent instead; see
  `audit/moments_2d_skimage_vetting_report.md`. Related trap: Nyxus `m_pq` has p on x and q on y, so
  index arrays `A[x, y]` and skimage's `moments(A)[i,j]` maps to `i==p, j==q` with no transpose.
- **`radiomicsj` 2.1.2 is a phantom version** — no GitHub releases/tags; Maven Central publishes
  2.1.3 … 2.1.18 (no 2.1.0/2.1.1/2.1.2). Pin **2.1.3** (earliest, nearest the requested tag) or
  **2.1.18** (latest). Ships a thin jar → must shade an uber-jar; has a built-in commons-cli `main()`
  and a bundled IBSI-phantom self-test (`-t -tdt 0`).
- **`fraclac` — reconciliation.** The FracLac *plugin* is GUI/AWT-bound and cannot run headless
  (confirmed on image.sc). BUT its distinguishing method — **shifting-grid box counting** (scan
  multiple grid origins per box size, take the MINIMUM count = true covering number, which removes
  grid-registration bias) — was already reimplemented as a **headless ImageJ macro** during the
  fractal-dim work: `shiftgrid_boxcount.ijm` (run via `fiji --headless -macro`), with a numpy
  reference `ref_boxcount.py`. So `fraclac` IS a usable headless fractal oracle in the form of that
  macro. Nuance: the macro is *our reimplementation of FracLac's method*, so it is a **from-method
  reference implementation**, not the FracLac tool emitting numbers; for true-tool goldens run the
  FracLac GUI once interactively (offline) and pin those, with the macro as the CI cross-check.
  Combined with **`radiomicsj` (Fractal family)** and **`imea` (box-counting)** — both headless —
  `FRACT_DIM_BOXCOUNT` has three independent headless oracles. `FRACT_DIM_PERIMETER` (divider/
  Richardson) has no tool → analytic + cross-method convergence.
  (Artifacts currently in scratchpad `fraclac/`; relocate to `tests/vetting/oracles/fraclac/` during
  rollout.)
- **`mitk` (revised — build-once Docker is viable, feasibility med).** No prebuilt image exists
  anywhere (Docker Hub / GHCR / Quay / CI all checked). BUT `MitkCLGlobalImageFeatures` builds via a
  dedicated CLI-only config **`-DMITK_BUILD_CONFIGURATION=ClassificationCmdApps`** — no Workbench,
  and Qt comes from apt (not compiled, unlike the old nolden image), so the superbuild is just
  ITK/VTK/Boost/GDCM/DCMTK/CTK/Qwt/Vigra: **~2–3 h one-time build, ~25–40 GB peak, few-hundred-MB
  runtime.** Bake into a pinned multi-stage image (heavy builder → slim runtime) and `docker run`
  forever after; runs headless via `QT_QPA_PLATFORM=offscreen`. A concrete Dockerfile recipe is in
  the research notes (relocate to `tests/vetting/oracles/mitk/Dockerfile` at rollout). Only fragile
  step: `ldd`-tracing the external `.so` set into the slim runtime.
  **Go/no-go:** GO only if a 4th independent IBSI opinion is wanted and the one-time build is
  acceptable; otherwise pyradiomics + mirp + radiomicsj already cover the same IBSI families with
  zero build.
- **`imea` 3D** is a *heightmap* analyzer, not a voxel-mask analyzer → good 2D-morphology oracle,
  **not usable for Nyxus true-3D** features.
- Setup split: **Docker** — pyradiomics, feature2djava, cellprofiler, radiomicsj (custom), wndcharm
  (custom), imagej (wrap tarball). **venv** — mirp, imea. Both are available on this box (Docker
  29.5, Python 3.11.2).
- **Reproducibility rule** (all tools): pin the exact version *and* record the resolved dependency
  set (Docker `@sha256`, `pip freeze`, `mvn dependency:tree`) alongside each golden, plus the
  discretisation/aggregation config used.
- **`octave` (license-free MATLAB substitute — no Docker needed).** A conda env with GNU Octave +
  the `statistics` package is a near-drop-in replacement for `matlab-ind/`'s stats-toolbox calls
  (`regionprops`, `graycomatrix`, `mean`/`median`/`std`/`var`/`skewness`/`kurtosis`/`prctile`/
  `quantile` all match within a few %; the one real gap is `graycoprops`, absent in Octave, needing
  a ~15-line reimplementation from the normalized GLCM — not relevant to firstorder).

  **Setup** (one-time):
  ```
  conda create -n octave_verify -c conda-forge octave octave-statistics
  ```
  (`octave-io` is not a real conda-forge package name despite showing up in some notes — don't try
  to install it; core Octave already reads flat files via `dlmread`/`csvread`, which is all a
  first-order vetting script needs.)

  **Use** (headless, one-shot eval):
  ```
  source <conda-base>/bin/activate octave_verify
  octave-cli -q --eval "pkg load statistics; x = dlmread('intensities.csv'); disp(prctile(x, 10))"
  ```
  or point `octave-cli -q` at a `.m` script for a multi-statistic dump (`printf('P10=%.15g\n',
  prctile(x,10));` etc. — see `tests/vetting/audit/firstorder_2d_golden_regen.md` for a fuller
  example script).

  **Known gotcha, found the hard way (see `tests/vetting/audit/firstorder_2d_matlab_vetting_report.md`):**
  Octave's default `skewness()`/`kurtosis()` use the same biased/population moment convention Nyxus
  does, so those two land close enough to look "MATLAB-vetted" even when they were never checked
  against a real MATLAB/Octave run. But `prctile()`/`quantile()` use standard order-statistic
  interpolation, which is a **different algorithm** from Nyxus's own percentile method (a fixed
  100-bin histogram with linear interpolation *within* the containing bin —
  `TrivialHistogram::calc_percentiles` in `src/nyx/features/histogram.h`). The two converge only
  when bins are densely populated; on a 150-pixel fixture they diverged by 1–2.5%. Conclusion: don't
  assume a percentile-derived Nyxus feature (`P*`, `INTERQUARTILE_RANGE`, `QCOD`, `ROBUST_MEAN*`) is
  oracle-vetted just because its golden matches Nyxus's own output — always cross-check against a
  real `prctile()`/`quantile()` call on the same fixture before trusting a `matlab`-labeled golden
  for these.

  **`regionprops` (2D morphology).** Needs the `image` package on top of `statistics`:
  ```
  conda create -n octave_verify -c conda-forge octave octave-statistics octave-image
  octave tests/vetting/oracles/gen_morphology_matlab.m      # from the repository root
  ```
  `pkg load image` gives `regionprops`, `bweuler` and `bwperim`. Three conventions bite, all
  recorded in `audit/morphology_2d_golden_regen.md`: centroids are 1-based pixel centres (Nyxus is
  0-based), `BoundingBox` puts its corner at min−0.5, and `Extrema` returns sub-pixel *corners* in a
  fixed 8-point order whose offset back to pixel centres is direction-specific (−0.5 for a left/top
  coordinate, −1.5 for a right/bottom one) — a uniform −0.5 is wrong.

  Watch the perimeter vocabulary in particular: `nnz(bwperim(BW))` counts perimeter **pixels**,
  `regionprops('Perimeter')` returns a boundary **length**, and neither equals scikit-image's
  `measure.perimeter`. A golden labelled `matlab` in this family turned out to be scikit-image's
  (`audit/morphology_2d_matlab_vetting_report.md`).

  Octave's own `containers.Map` supports only `()` indexing — `m{k}` is an error — which matters
  when a generator merges several pinned-golden tables.

## Coverage by Nyxus family → which oracles (the "≥1 oracle" picture)

| Nyxus family | headless oracles available |
|--------------|----------------------------|
| First-order / intensity | pyradiomics, mirp, radiomicsj, feature2djava, cellprofiler |
| First-order on CT / Hounsfield (`--preserve-hu`) | **pydicom** (decode + rescale → HU reference) |
| Intensity histogram | mirp, radiomicsj, (analytic) |
| GLCM (Haralick) | pyradiomics, mirp, radiomicsj, feature2djava, cellprofiler, imagej, wndcharm |
| GLRLM / GLSZM / GLDM / NGTDM | pyradiomics, mirp, radiomicsj |
| **NGLDM / GLDZM** | **mirp, radiomicsj** (NOT pyradiomics) |
| Morphology / shape (basic) | imea, cellprofiler, feature2djava, imagej, pyradiomics(shape2D), mirp(morph), radiomicsj |
| Caliper / Feret / Martin / Nassenstein / chords | **imea** (primary), wndcharm |
| Moments — Zernike | wndcharm, cellprofiler(2D) |
| Moments — raw/central/Hu | wndcharm; else analytic (thin — flag) |
| Gabor / Tamura / radial(Radon) | **wndcharm** (only) |
| Fractal (box-count) | radiomicsj, imea, fraclac (headless shifting-grid macro), imagej(approx) |
| Neighbor: `CLOSEST_NEIGHBOR*_DIST` | **cellprofiler** `MeasureObjectNeighbors` (1:1, centroid-Euclidean) + scipy analytic |
| Neighbor: `NUM_NEIGHBORS`, `PERCENT_TOUCHING` | **cellprofiler** (with-caveat: disk-dilation vs Euclidean contour-dist; ±1 / perimeter diffs) + scipy analytic (bit-exact) |
| Neighbor: `*_ANG`, `ANG_BW_NEIGHBORS_*` | scipy/numpy analytic only (CP `AngleBetweenNeighbors` is a different quantity) |
| Image quality (IMQ) | none → analytic / reference impl |

Takeaway: nearly every family has ≥1 headless oracle. Texture/first-order are richly covered (3–5
tools); morphology well-covered; the Nyxus-original set (Zernike/Gabor/Tamura/radial) rides on
`wndcharm`; NGLDM/GLDZM ride on `mirp`/`radiomicsj`. **Neighbor features are partly tool-vetted**:
CellProfiler `MeasureObjectNeighbors` covers the distances (1:1) and count/percent-touching
(with-caveat), while the angle features use a `scipy.cKDTree`-on-boundary analytic oracle that
reproduces Nyxus's definitions exactly. Genuine oracle gaps that stay analytic-only: the neighbor
**angle** features, **IMQ**, and **raw/central/Hu moments** (wndcharm/analytic).

### Neighbor features — CellProfiler reconciliation

`cellprofiler` `MeasureObjectNeighbors` (Docker image already in the catalog) is a real oracle here.
Config to match Nyxus: objects = neighbors = the label set; method + distance chosen to match Nyxus's
`PERCENT_TOUCHING` touch definition — **"Within a specified distance", distance=`pixel_distance`**
matches the radius-based touch (Nyxus `main`), while **"Adjacent"** (`strel_disk(1.5)` = 8-neighborhood)
matches the 8-connected `sqdist≤2` touch introduced by PR #359. Outputs: `Neighbors_NumberOfNeighbors_*`,
`Neighbors_PercentTouching_*`, `Neighbors_First/SecondClosestDistance_*`, `Neighbors_AngleBetweenNeighbors_*`
(the last is NOT Nyxus's `ANG_BW_NEIGHBORS_*` — different quantity). A `scipy.cKDTree` boundary-pixel
recipe reproduces Nyxus's contour-min-distance, centroid direction-angle, and mode exactly — the
primary oracle for the angle features and a bit-exact backstop for the rest.

## Suggested rollout order (easy+high-value first)

1. **venv tools, zero-friction:** `mirp`, `imea` — pure pip, cover NGLDM/GLDZM + morphology/caliper.
2. **prebuilt Docker:** `pyradiomics` (done), `feature2djava` (sibling; intensity+shape+Haralick),
   `cellprofiler`.
3. **custom Docker builds:** `radiomicsj` (Maven→uber-jar; full IBSI + fractal), `wndcharm`
   (Py2.7 image; Zernike/Gabor/Tamura/radial — the only source for those).
4. **special-case:** fractal via `radiomicsj`/`imea` + the existing `fraclac` shifting-grid macro
   (relocate from scratchpad); `imagej` as a general second opinion.
5. **optional (build-once Docker):** `mitk` — one ~2–3 h `ClassificationCmdApps` build yields a
   reusable pinned image; do it only if a 4th independent IBSI opinion is wanted.
