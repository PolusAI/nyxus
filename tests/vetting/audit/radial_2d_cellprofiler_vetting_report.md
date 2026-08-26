# 2D radial intensity distribution vs CellProfiler — vetting report

**Verdict: not vetted, and not vettable as implemented.** `FRAC_AT_D`, `MEAN_FRAC` and `RADIAL_CV`
carry CellProfiler's three `RadialDistribution_*` names and CellProfiler's three help strings, and
compute a different quantity under each of them. 21 of the 24 (feature × bin) values disagree by
more than 1%; the 3 that "agree" are bins both tools leave empty. No tolerance absorbs this — it is
six independent definition and implementation differences, not float drift.

The three rows stay `status=regression`. Nothing is demoted, because nothing was ever promoted: the
registry recorded `status=regression` with an empty `oracle` column and a `candidate_oracle` naming
CellProfiler. This report is the answer to that candidacy.

- Oracle: `cellprofiler` 4.2.8 (module package), `cellprofiler-core` 4.2.8.1, `centrosome` 1.2.3,
  numpy 1.26.4, scipy 1.10.1, python 3.9.23, headless. The generator reads all five back from the
  installed distributions and refuses to run against any other, so this line is checked rather than
  asserted.
- Module: `MeasureObjectIntensityDistribution`, `center_choice="These objects"`, `bin_count=8`,
  `wants_scaled=True`, Zernikes off. Recipe `radial.cellprofiler_8bin`.
- Fixture: `shape2d_morphology_{intensity,mask}` (`test_data.h`) — one 26-pixel concave ROI with one
  interior hole in an 8×8 grid, total intensity 1048. Recipe `radial.shape2d_native`.
- Generator: `tests/vetting/oracles/gen_radial_cellprofiler.py`. It parses the fixture out of
  `test_data.h`, the pinned vectors out of `test_2d_radial_regression.h` and the centre/radius out of
  `test_2d_radial_mechanics.h`, so it holds no copy of anything it checks.
- Regeneration: `tests/vetting/audit/radial_2d_golden_regen.md`.
- **No second oracle exists to corroborate this one.** `TOOLS.md`'s family matrix lists `wndcharm`
  under "Gabor / Tamura / radial", but wndcharm's radial features are Radon-transform coefficients,
  a different quantity from a radial intensity distribution, and it is a Python 2.7 source build. No
  other tool in the §4 catalog implements `FracAtD`/`MeanFrac`/`RadialCV`. CellProfiler is both the
  first and the only candidate, which is why its rejection settles the family rather than opening a
  second comparison.

---

## 1. What each tool computes

Read from `src/nyx/features/radial_distribution.cpp` and from
`cellprofiler/modules/measureobjectintensitydistribution.py` (the `do_measurements` body).

| step | CellProfiler | Nyxus |
|---|---|---|
| centre | the pixel of maximum distance-to-edge (`distance_to_edge` → `maximum_position_of_labels`) | the pixel minimising (max − min) squared distance to the contour (`Pixel2::find_center`) |
| distance from centre | geodesic, propagated inside the mask | straight-line Euclidean |
| radial coordinate | `d_centre / (d_centre + d_edge + 0.001)`, **per pixel** | `d_centre / r_max`, one global `r_max` for the whole ROI |
| bin index | `int(nd × n_bins)`, clipped at `n_bins` | `int(nd × (n_bins − 1))`, clamped to `n_bins − 1` |
| `FracAtD` | bin intensity ÷ ROI intensity | **bin pixel count ÷ ROI pixel count** |
| `MeanFrac` | `FracAtD ÷ (bin count ÷ ROI count)` — dimensionless, ~1 | **the bin's raw mean intensity** |
| `RadialCV` | CV of the 8 wedge **means**, over the **non-empty** wedges | CV of the 8 wedge **sums**, over **all 8** including empty ones |

The two header comments in `radial_distribution.h` are CellProfiler's wording verbatim — "Fraction of
total stain in an object at a given radius", "Fraction of total intensity normalized by fraction of
pixels at a given radius". Neither describes what the function below it returns.

## 2. The pinned goldens are exactly reproducible

Before comparing anything, every pinned value was checked against a written-down model of the
implementation (`nyxus_model` in the generator), and against a fresh Windows build.

- model vs header: **24 of 24 bit-identical**, `rel = 0` throughout.
- fresh build vs header: **24 of 24 bit-identical**. Measured by temporarily replacing the
  assertion with `actual[i] == golden_values[i]` and rebuilding; all 8 radial cases passed. The
  shipped band is therefore `rel=1e-9` (`agrees_gt(..., 1e9)`), which is the family's registry
  tolerance and the tightest drift guard the house helper offers. It was previously
  `ASSERT_NEAR(..., 1e-9)` **absolute** with an empty registry tolerance column.

So the numbers in the header are what the program produces, and the disagreement below is entirely
about what the program computes.

**What that model does *not* establish.** `nyxus_model` is a reimplementation of Nyxus' own
algorithm, and it takes the centre pixel and the normalising radius as **inputs**, parsed out of
`test_2d_radial_mechanics.h` — which is to say, out of Nyxus' own output. It therefore checks the
binning and the three statistics *given* a centre and a radius; it cannot check the centre and the
radius, because both come from approximate searches no clean reimplementation reproduces (§6 defects
1 and 2, which is why those are pinned in a mechanics test instead). A match against a
reimplementation of Nyxus' procedure is not a match against a reference implementation of the
statistic (SPEC §5.2) — the independent check is §4, and its verdict is §3.

## 3. The comparison

`nyxus` is the pinned golden; `cellprofiler` is the fresh run at the recipe above.

| feature | bin | nyxus | cellprofiler | rel |
|---|---:|---:|---:|---:|
| FRAC_AT_D | 0 | 0.038461538460059175 | 0.022900762036442757 | 0.405 |
| FRAC_AT_D | 1 | 0 | 0 | 0 |
| FRAC_AT_D | 2 | 0.11538461538017751 | 0 | 1 |
| FRAC_AT_D | 3 | 0.1538461538402367 | 0.16698472201824188 | 0.0854 |
| FRAC_AT_D | 4 | 0.3076923076804734 | 0.011450381018221378 | 0.963 |
| FRAC_AT_D | 5 | 0 | 0.30438932776451111 | ∞ |
| FRAC_AT_D | 6 | 0.11538461538017751 | 0.4942747950553894 | 3.28 |
| FRAC_AT_D | 7 | 0.26923076922041422 | 0 | 1 |
| MEAN_FRAC | 0 | 50.999999948999999 | 0.59541981294750823 | 0.988 |
| MEAN_FRAC | 1 | 0 | 0 | 0 |
| MEAN_FRAC | 2 | 53.333333315555556 | 0 | 1 |
| MEAN_FRAC | 3 | 50.749999987312499 | 0.72360046207904738 | 0.986 |
| MEAN_FRAC | 4 | 47.374999994078124 | 0.29770990647375412 | 0.994 |
| MEAN_FRAC | 5 | 0 | 0.98926531523466033 | ∞ |
| MEAN_FRAC | 6 | 33.666666655444445 | 1.2851144671440116 | 0.962 |
| MEAN_FRAC | 7 | 21.999999996857142 | 0 | 1 |
| RADIAL_CV | 0 | 2.6457513106495707 | 0 | 1 |
| RADIAL_CV | 1 | 0 | 0 | 0 |
| RADIAL_CV | 2 | 1.298797520721114 | 0 | 1 |
| RADIAL_CV | 3 | 1.024429214739045 | 0.34309516188156874 | 0.665 |
| RADIAL_CV | 4 | 0.64750329537582818 | 0 | 1 |
| RADIAL_CV | 5 | 0 | 0.29649096645612216 | ∞ |
| RADIAL_CV | 6 | 1.3575192606324717 | 0.25528607865752179 | 0.812 |
| RADIAL_CV | 7 | 1.3284260624865412 | 0 | 1 |

The `MEAN_FRAC` column is the clearest single reading: CellProfiler's values sit around 1 because the
quantity is normalised by the ROI's mean intensity; Nyxus' sit between 22 and 53 because they are
that mean intensity in absolute units. They cannot be brought together by a tolerance.

## 4. The CellProfiler run was verified independently

CellProfiler's numbers were reproduced from the published definitions with numpy/scipy only —
`distance_transform_edt` for the distance to the edge, a Dijkstra propagation for the geodesic
distance from the centre, then the normalisation, binning and three statistics of §1 — with no
`centrosome` call in the path. Agreement with the module's own output is **6.47e-08 worst case across
all 24 values**, which is CellProfiler storing the image as float32. So the table in §3 is the
tool's answer and not an artefact of how the module was driven.

The rebuild is `cellprofiler_independent()` in the generator and runs on every invocation, which
fails the run if it ever stops reproducing the module. The one step it takes from the tool is which
of the tied centre pixels to use — see §5; an independent tie-break would compare two different ROIs
rather than two implementations.

## 5. This fixture could not vet the family even after a fix

CellProfiler's centre is the pixel of maximum distance-to-edge. On this ROI that maximum is
**1.4142135623730951, attained by 8 of the 26 pixels** — `(1,2) (1,3) (2,1) (2,2) (2,4) (3,1) (3,5)
(4,2)`. Which one `scipy.ndimage.maximum_position` returns depends on the shape of the label image,
so CellProfiler's own answer moves when the padding around the ROI changes; it was observed to do
exactly that while this report was being written. The generator prints the tie set on every run for
that reason.

A ROI whose distance-to-edge maximum is unique is a precondition for this family ever vetting against
CellProfiler, and it is independent of the six divergences. Twenty-six pixels spread over eight
radial bins is also too thin to distinguish the two binning rules from a coincidence — three of the
eight bins are empty on one side or the other.

## 6. Defects found, with evidence

None of these are fixed here: all six change public feature values, which belongs on its own branch.
Defects 1, 2 and 3 are pinned in `test_2d_radial_mechanics.h`, so a correction cannot land silently.
Read those pins as a record of today's behaviour, not as acceptance criteria — a fix **must** change
every number in that file, which is why it is labelled known-defect characterization and credited to
no feature in the registry. `radial_2d_golden_regen.md` §5 is the corrective work's checklist: what
has to be true before the family can be promoted.

**1. The traced contour is returned one pixel off in both axes.**
`ContourFeature::buildRegularContour` (`src/nyx/features/contour.cpp:648-681`) places the ROI into an
image padded by one pixel on every side at `px.x - base_x + 1, px.y - base_y + 1`, traces there, and
then converts back with `p.x += base_x; p.y += base_y` — the `base` is restored, the `+1` pad never
is. Every contour pixel is therefore reported one pixel right and one pixel down of where it is.
On this ROI that puts **7 of the 18 contour pixels outside the ROI altogether**, while shifting all
18 back by (−1,−1) lands every one of them on a ROI pixel. No test in the tree looks at contour
coordinates — `test_2d_contour_analytic.h` asserts only how many contours come back — which is why
this has never fired.

*Blast radius.* A uniform translation is harmless to anything that only measures the contour against
itself, so `PERIMETER`, the Feret/caliper set, the fractal divider walk and the neighbour features
(which compare two contours that are shifted equally) are unaffected. It is the callers that mix
contour coordinates with *pixel* coordinates that are wrong:
`radial_distribution.cpp` (this family), `roi_radius.cpp` (`ROI_RADIUS_MEAN/MAX/MEDIAN`),
`circle.cpp` (`DIAMETER_INSCRIBING_CIRCLE`, `DIAMETER_CIRCUMSCRIBING_CIRCLE`, which pass the ROI
centroid alongside the contour), and `2d_geomoments_basic.cpp:117-122` (every `WEIGHTED_*` moment,
which weights each raw pixel by its distance to the contour). `cache.cpp:300-330` rebases the pixel
cloud and the contour onto the ROI bounding box separately for the GPU path and inherits the same
one-pixel gap between them.

**2. `Pixel2::find_center` and `Pixel2::max_sqdist` return non-extremal answers here.**
Both are coarse-to-fine searches (`src/nyx/features/pixel.cpp:13-141`) that step through the contour
assuming it is an ordered walk. `LR::merge_multicontour` concatenates the outer contour and the
hole's, so the sequence they walk is two walks joined end to end and the descent settles in the wrong
basin. Measured on this ROI against a full linear scan of the same contour:

| quantity | searched | exact scan |
|---|---|---|
| centre pixel | (3,4) | (4,4) |
| max squared distance from the centre | 10 | 13 |

The radius the feature normalises by is 23% short of the largest distance actually present, so **4 of
the 26 pixels have `r/r_max > 1`**. `exact_min_sqdist()` already exists beside `min_sqdist()` for
exactly this reason and is used by the neighbour features; there is no `exact_max_sqdist`. The
`ROI_RADIUS_MEDIAN` demotion records the same approximation as a second defect underneath its units
bug (`morphology_2d_skimage_vetting_report.md`).

**3. The bin index is scaled by `num_bins - 1`.**
`radial_distribution.cpp:81` computes `int(rat * (n-1))` and clamps to `n-1`. With `n = 8` that is 7
rings of width 1/7 plus a last bin nothing reaches unless `r >= r_max`. Combined with defect 2 the
last bin ends up holding 7 of the 26 pixels — 3 sitting exactly on `r_max` and the 4 beyond it —
more than any of the 7 real rings. CellProfiler scales by `bin_count` and gets 8 equal rings.

**4. `FRAC_AT_D` returns a pixel-count fraction.**
`get_FracAtD()` is `radial_count_bins[i] / cached_num_pixels`. The name, the header comment and
CellProfiler all say fraction of *intensity*. Intensity never enters the computation, so the feature
is blind to the image entirely — it is a function of the mask alone.

**5. `MEAN_FRAC` is not a fraction.**
`get_MeanFrac()` is `radial_intensity_bins[i] / radial_count_bins[i]`, the bin's mean intensity in
absolute units. CellProfiler's `MeanFrac` divides that by the ROI's mean intensity. The header
comment states CellProfiler's definition.

**6. `RADIAL_CV` averages over empty wedges.**
`get_RadialCV()` divides by `num_bins` unconditionally, so a ring occupied in 3 of its 8 wedges is
treated as one occupied in 3 and empty in 5, which inflates the CV; CellProfiler masks the empty
wedges out. It also uses wedge *sums* where CellProfiler uses wedge *means* — equivalent only when
every wedge holds the same number of pixels. `banded_wedges` stores `size_t`, but `PixIntens` is
already an unsigned integer, so that container introduces no additional truncation.

## 7. What this PR did instead

- Re-pinned nothing — the 24 goldens were already at full precision and reproduce bit-exactly.
- Replaced the `ASSERT_NEAR(..., 1e-9)` *absolute* band with `agrees_gt(..., 1e9)` (`rel=1e-9`), and
  filled the three empty `tolerance` cells in the registry.
- Split the family into three applicable SPEC §2 kinds — `_regression.h` (the 24 pins plus the
  convention checks below), `_invariant.h` (3 required properties), and `_mechanics.h` (3
  known-defect pins) — plus `_common.h` for the shared fixture.
- Sorted the property checks by whether they survive a change of definition, which is the entry test
  for `_invariant.h`. `FRAC_AT_D` being a partition of `[0, 1]` summing to one, an empty bin being
  zero in all three tables, and `RADIAL_CV` lying in `[0, sqrt(num_bins - 1)]` hold under
  CellProfiler's definitions as well as Nyxus', so they are invariants. `FRAC_AT_D × pixel_count`
  being a whole number, `MEAN_FRAC` lying inside the ROI's raw intensity range, and the two tables
  reconstructing the ROI's total intensity hold only because of the conventions §6 defects 4 and 5
  describe — CellProfiler's `FracAtD` and `MeanFrac` satisfy none of the three — so they moved to
  `_regression.h` as `TEST_2D_RADIAL_BIN_CONVENTIONS_REGRESSION`, characterization beside the pins
  they characterize. An invariant a correct fix would break is not an invariant.
- Labelled `_mechanics.h` as known-defect characterization in its header and on each assertion, and
  removed it from the three registry rows' `current_test`: it pins §6 defects 1-3 and is diagnostic,
  not correctness coverage for the features. The omission is recorded in `not_covered.md` §A.1 so
  it reads as a decision rather than a gap.
- Rewrote the three header comments in `src/nyx/features/radial_distribution.h`. They were
  CellProfiler's help strings verbatim over functions computing something else (§1); they now state
  the pixel-count fraction, the raw bin mean and the all-8-wedge CV of sums, and say that which
  semantics is intended is unresolved. No behaviour changes.
- Made the generator's stale-verdict alert per feature rather than one counter over all 24 cells, at
  SPEC §7's `rel=1e-2` cross-tool band, and left the 1% cutoff as the descriptive report line only.
  The old counter fired only when the whole family agreed; a run in which all eight `FRAC_AT_D` bins
  came inside the band while `MEAN_FRAC` still diverged would have exited 0 and left its rejection
  unexamined. Agreement now requires re-vetting on a suitable fixture; this tie-dependent fixture
  does not establish promotion. Verified by feeding the alert exactly that case.
- Made the generator read `cellprofiler`, `cellprofiler-core`, `centrosome`, `numpy` and `scipy`
  versions from the installed distributions, print them, and refuse to run on a mismatch (a missing
  distribution counts as one). `--allow-version-drift` continues with a warning and says on the
  final line that the run is not the recorded provenance. The versions above are that check's
  output on the run in §3.
- Added range and identity checks over the pinned literals to the generator, so they are asserted of
  what the header *says* and not only of what a build computes: every `FRAC_AT_D` entry a fraction of
  a whole pixel count summing to one, every `MEAN_FRAC` entry inside the ROI's intensity range on a
  non-empty bin, `RADIAL_CV` within `sqrt(num_bins - 1)`, the empty bins consistent across all three
  tables, and the two intensity tables reconstructing the ROI total. That path is stdlib-only
  (`--skip-cellprofiler`).
- Negative-controlled every new assertion shape. For the per-bin regression: swapping the `FRAC_AT_D`
  bin-3 and bin-6 goldens leaves the sum exactly 1 and every bin count a whole number, so all the
  invariants still pass, while the per-bin assertion fails naming `FRAC_AT_D[3]`. For the invariants
  and the convention characterization: `FRAC_AT_D[0] += 0.5`, `MEAN_FRAC[2] = 1000`,
  `MEAN_FRAC[2] += 1` and `RADIAL_CV[2] = 3` each fire one case and nothing else — the first three
  now land on `TEST_2D_RADIAL_BIN_CONVENTIONS_REGRESSION`, which is where those checks moved, and
  `+= 0.5` is the instructive one: it leaves `FRAC_AT_D[0] × 26` a whole 14, so it falls past the
  whole-count check onto the reconstruction identity. For the generator's pin checks: raising
  `MEAN_FRAC[6]` from 33.67 to 333.67 fires the range check and the reconstruction identity
  together. For the mechanics frame guard: shifting the test's own copy of the contour back by
  (−1,−1) — i.e. simulating a fixed `buildRegularContour` — takes the "lands on a ROI pixel after
  unpadding" count from 18 to 12 and fails the assertion, which is what shows it is a detector for
  the offset and not a restatement of it.
- Negative-controlled the two alerts the review asked for. Feeding the generator a CellProfiler run
  in which all eight `FRAC_AT_D` bins match the pins while `MEAN_FRAC` and `RADIAL_CV` still diverge:
  the old counter reported "14 of the 24 disagree" and exited 0, while the per-feature alert exits 1
  naming `FRAC_AT_D` for re-vetting. Moving one of those bins 2% off the pin returns it to a pass,
  and 0.5% off — inside the band — keeps it a fail, so the alert is the band and not an equality
  test. For the version check: `cellprofiler 4.2.9` installed and `cellprofiler-core` absent each exit 2, and
  `--allow-version-drift` turns both into a warning plus a closing note that the run is not the
  recorded provenance.
