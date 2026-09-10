# Audit: `test_2d_firstorder_pyradiomics.h` goldens vs. a fresh PyRadiomics run

**Verdict: all 18 mapped goldens reproduce exactly.** No drift, no tolerance-tier violations.

## Provenance confirmed

- Pulled image: `docker pull radiomics/pyradiomics@sha256:eea20621c9e77afd049871e1a4e7308844a57d399343b087f6a4e86c3dab1923`
  → digest resolved unchanged (pinned digest, not `:latest`).
- `pyradiomics --version` inside the container: `v3.0.1.post2+g9ccbec1` — matches the header's
  claimed version exactly.
- `SimpleITK` inside the container: `2.0.1 (ITK 5.1)`.
- Container Python: `/opt/conda/bin/python` (3.8.6, per the CSV's own
  `diagnostics_Versions_Python` field).

## Fixture

`pixelIntensityFeaturesTestData` (`tests/test_data.h`) parsed directly from the header (regex over
the `{x, y, intensity}` triples, not hand-copied) — confirmed **154 pixels**, bounding box
x∈[0,12], y∈[0,17] (13×18). Image = float array with each listed pixel's intensity, 0 elsewhere;
mask = 1 exactly at listed pixels, 0 elsewhere — matching how `load_test_roi_data` in
`tests/test_main_nyxus.h` builds the ROI (every triple in the array is a foreground pixel, nothing
padded in as background inside the bbox).

## Reproduction steps (exact commands run)

```bash
# 1. Pull pinned image
docker pull radiomics/pyradiomics@sha256:eea20621c9e77afd049871e1a4e7308844a57d399343b087f6a4e86c3dab1923

# 2. Introspect the container's bundled interpreter/toolchain
docker run --rm <image> python -c "import SimpleITK; print(SimpleITK.Version())"
docker run --rm <image> pyradiomics --version

# 3. Build image.nrrd/mask.nrrd using the container's own SimpleITK
#    (build_fixture.py parses tests/test_data.h's pixelIntensityFeaturesTestData
#     and writes image.nrrd/mask.nrrd; scratch dir mounted at /data)
docker run --rm -v <scratchdir>:/data <image> python /data/build_fixture.py

# 4. params.yaml
#    setting: {binCount: 64, force2D: true, label: 1}
#    featureClass: {firstorder: []}

# 5. Run pyradiomics headlessly
docker run --rm -v <scratchdir>:/data <image> \
  pyradiomics /data/image.nrrd /data/mask.nrrd -p /data/params.yaml -o /data/out.csv -f csv
```

Warning emitted (expected, not an error): `Fixed bin Count enabled! However, we recommend using a
fixed bin Width` — this is pyradiomics nagging about its own default recommendation, irrelevant to
correctness of the run since the recipe explicitly calls for `binCount=64`.

## Results

All values below are the freshly-generated PyRadiomics v3.0.1.post2+g9ccbec1 output for this exact
fixture/recipe, compared against the literals hardcoded in
`firstorder_2d_pyradiomics_ref_vals`.

| Feature | Hardcoded | Fresh (PyRadiomics) | Rel. diff | Within declared tolerance? | Notes |
|---|---|---|---|---|---|
| MEAN | 32566.38961038961 | 32566.38961038961 | 0.0 | yes (exact, 1e-6) | |
| MEDIAN | 29803.5 | 29803.5 | 0.0 | yes (exact) | |
| MIN | 11079.0 | 11079.0 | 0.0 | yes (exact) | |
| MAX | 64090.0 | 64090.0 | 0.0 | yes (exact) | |
| RANGE | 53011.0 | 53011.0 | 0.0 | yes (exact) | |
| VARIANCE_BIASED | 215592327.38067126 | 215592327.38067126 | 0.0 | yes (exact) | PyRadiomics `Variance` is population (`/N`); matches Nyxus `VARIANCE_BIASED` definition directly. |
| VARIANCE | 215592327.38067126 | 215592327.38067126 | 0.0 | yes (well within the 1e-2 tier reserved for the Bessel gap) | Golden itself is just the PyRadiomics population-variance number reused as VARIANCE's reference; the ~6.54e-03 divergence documented in the header is a Nyxus `VARIANCE` (sample, `/N-1`) vs. this golden difference, checked at Nyxus-test time, not part of this oracle-reproduction audit. |
| SKEWNESS | 0.45025675970449414 | 0.45025675970449414 | 0.0 | yes (exact) | |
| KURTOSIS | 1.9278887207100905 | 1.9278887207100905 | 0.0 | yes (exact) | PyRadiomics `Kurtosis` is non-excess (+3), matching Nyxus `KURTOSIS` convention as documented. |
| ENERGY | 196528957184.0 | 196528957184.0 | 0.0 | yes (exact) | |
| ROOT_MEAN_SQUARED | 35723.41052638121 | 35723.41052638121 | 0.0 | yes (exact) | |
| MEAN_ABSOLUTE_DEVIATION | 12833.084499915672 | 12833.084499915672 | 0.0 | yes (exact) | |
| ROBUST_MEAN_ABSOLUTE_DEVIATION | 10440.618496000001 | 10440.618496000001 | 0.0 | yes (exact) | |
| INTERQUARTILE_RANGE | 26116.25 | 26116.25 | 0.0 | yes (well within the 5e-2 tier) | |
| P10 | 16329.0 | 16329.0 | 0.0 | yes (well within the 5e-2 tier) | |
| P90 | 53295.0 | 53295.0 | 0.0 | yes (well within the 5e-2 tier) | |
| ENTROPY | 5.54700500819408 | 5.54700500819408 | 0.0 | yes (exact) | |
| UNIFORMITY | 0.0252993759487266 | 0.0252993759487266 | 0.0 | yes (exact) | |

## Conclusion

Every hardcoded golden in `firstorder_2d_pyradiomics_ref_vals` reproduces byte-for-byte
(to float64 printing precision)
from a fresh, independent run of the exact pinned PyRadiomics image against the documented fixture
and recipe. No feature falls outside its own declared tolerance tier — nothing to flag as a vetting
failure. The header's provenance block (version, digest, recipe, fixture) is accurate and the data
is reproducible as claimed.
