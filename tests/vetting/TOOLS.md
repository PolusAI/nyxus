# Oracle tools — local setup & coverage (research findings)

How to stand up each oracle tool locally (Docker or Python venv; MATLAB excluded — license). One
research pass per tool; see per-tool detail below and the setup matrix first.

## Setup matrix

| token | version | setup | feasibility | one-line |
|-------|---------|-------|:-----------:|----------|
| `pyradiomics` | 3.0.1 | **Docker** `radiomics/pyradiomics:latest` (pin by `@sha256`) | high | already in use; pip blocked on Py3.11 |
| `mirp` | 2.6.0 | **venv** `pip install mirp==2.6.0` (Py3.11, pure-Python) | high | no Docker needed; full IBSI incl. NGLDM/GLDZM |
| `imea` | 0.3.3 | **venv** `pip install imea==0.3.3` + `numpy<1.24` | high (2D only) | 2D morphology; **3D is heightmap, not voxel — unusable for Nyxus 3D** |
| `feature2djava` | 1.5.0 | **Docker** `wipp/wipp-feature2djava-plugin:1.5.0` (125 MB, exists) | high | NIST/WIPP sibling; intensity + basic shape + Haralick |
| `radiomicsj` | **2.1.3 / 2.1.18** | **Docker/jar** (Maven → shaded uber-jar; Java 11) | high | **2.1.2 does not exist**; Maven Central only; full IBSI + fractal |
| `cellprofiler` | 4.2.1 | **Docker** `cellprofiler/cellprofiler:4.2.1` (pin digest) | high | headless `-c -r`; maybe `xvfb-run`; feed mask as "Objects" |
| `wndcharm` | 1.60 | **Docker (custom build)** Ubuntu 18.04 + Py2.7 + swig | high | no pip/no image; Nyxus-lineage: Haralick/Tamura/Zernike/Gabor/Chebyshev/radial |
| `imagej` | pinned tarball | **download** Fiji `ImageJ-linux64 --headless` | med-high | morphology/intensity/GLCM headless (GLCM via batch wrapper) |
| `fraclac` | — | ImageJ plugin (GUI) **+ headless-macro reimpl** | med-high* | plugin is GUI-only, but its shifting-grid method runs headless via our macro (*see reconciliation) |
| `mitk` | 2023.04 | **build-once Docker** (`ClassificationCmdApps` config) | med | no prebuilt image; ~2–3 h one-time CLI-only build → reusable pinned image |

## Corrections / notable findings

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

## Coverage by Nyxus family → which oracles (the "≥1 oracle" picture)

| Nyxus family | headless oracles available |
|--------------|----------------------------|
| First-order / intensity | pyradiomics, mirp, radiomicsj, feature2djava, cellprofiler |
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
