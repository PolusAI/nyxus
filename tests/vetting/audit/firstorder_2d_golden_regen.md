# Regenerating 2D firstorder oracle goldens

Covers the two oracles the `test_2d_firstorder_*.h` files actually use: **pyradiomics** and
**matlab** (recommend switching the latter to the license-free **Octave** realization per
`MIGRATION.md` 5.13 / memory `octave-matlab-oracle`). The `ibsi` file uses the pinned IBSI digital
phantom reference values (already in SPEC/`tests/data`) - no regeneration needed unless IBSI
publishes an errata.

Fixture for both: `pixelIntensityFeaturesTestData` (`tests/test_data.h`) - a single-ROI, single-slice
2D label image with intensity range [11079, 64090].

## pyradiomics (test_2d_firstorder_pyradiomics.h)

Recipe = `firstorder.pyradiomics_default`: binCount=64, single 2D slice, spacing=1, label=1.

1. Export the fixture as a NRRD/NIfTI image + label mask pair (`image.nrrd`, `mask.nrrd`) - write a
   throwaway script that dumps `pixelIntensityFeaturesTestData` (x, y, intensity) into a 2D numpy
   array and mask, then `SimpleITK.WriteImage`.
2. Pull the pinned image (never build pyradiomics into CI):
   ```
   docker pull radiomics/pyradiomics@sha256:eea20621c9e77afd049871e1a4e7308844a57d399343b087f6a4e86c3dab1923
   ```
3. Params file (`params.yaml`) matching the recipe:
   ```yaml
   setting:
     binCount: 64
     force2D: true
     label: 1
   featureClass:
     firstorder: []
   ```
4. Run headless, capture stdout as the golden source:
   ```
   docker run --rm -v $PWD:/data radiomics/pyradiomics@sha256:eea20621c9e77afd049871e1a4e7308844a57d399343b087f6a4e86c3dab1923 \
     pyradiomics /data/image.nrrd /data/mask.nrrd -p /data/params.yaml -o /data/out.csv -f csv
   ```
5. Map PyRadiomics field names -> Nyxus `Feature2D` per the existing table in
   `test_2d_firstorder_pyradiomics.h` (note the two known non-matches: `Kurtosis` is non-excess
   [+3], and `Variance` is population `/N` == Nyxus `VARIANCE_BIASED`, not `VARIANCE`).
6. Pin into the header: version (`pyradiomics --version` inside the container), the `@sha256`
   digest, the params file content, and the generation date - same shape as the provenance block
   already at the top of the file. Do **not** commit the generator script's *invocation* as a CI
   step; keep it in `tests/vetting/oracles/gen_firstorder_pyradiomics.py` per SPEC (offline-only).

## matlab / octave (test_2d_firstorder_matlab.h)

Per TOOLS.md, MATLAB itself isn't installable in this environment (license) - use Octave, which
memory records as a verified near-drop-in for the stats functions this family needs (`mean`,
`median`, `mode`, `std`, `var`, `skewness`, `kurtosis`, `quantile`, `mad`, matched within 5%; no
`graycoprops` gap here since firstorder never touches GLCM).

1. Install: `apt install octave octave-statistics` (or conda-forge `octave` +
   `octave --eval "pkg install -forge statistics"`).
2. Dump the same fixture's intensities to a flat vector (`intensities.csv` - one value per pixel,
   order doesn't matter for firstorder stats).
3. Octave script (`gen_firstorder_octave.m`), headless:
   ```octave
   pkg load statistics;
   x = csvread('intensities.csv');
   printf('MEAN=%.15g\n', mean(x));
   printf('MEDIAN=%.15g\n', median(x));
   printf('MODE=%.15g\n', mode(x));
   printf('STANDARD_DEVIATION=%.15g\n', std(x));          % sample, N-1
   printf('STANDARD_DEVIATION_BIASED=%.15g\n', std(x,1)); % population, N
   printf('VARIANCE=%.15g\n', var(x));
   printf('VARIANCE_BIASED=%.15g\n', var(x,1));
   printf('SKEWNESS=%.15g\n', skewness(x));
   printf('KURTOSIS=%.15g\n', kurtosis(x));                % Octave kurtosis() is non-excess
   printf('EXCESS_KURTOSIS=%.15g\n', kurtosis(x) - 3);
   printf('P01=%.15g\n', quantile(x, 0.01));
   printf('P10=%.15g\n', quantile(x, 0.10));
   ...
   ```
   Run: `octave-cli -q gen_firstorder_octave.m > out.txt`
4. `UNIFORMITY` needs the histogram config, not a stats-toolbox call directly: bin the fixture at
   GREYDEPTH=20 (matching Nyxus's non-IBSI histogram path, `IBSI=false`) the same way
   `intensity.cpp` does, then compute `sum(p_i^2)` over the normalized bin counts.
   `COVERED_IMAGE_INTENSITY_RANGE` needs the slide min/max (0/65535 per the test's fixture) folded
   in as `(roi_max-roi_min)/(slide_max-slide_min)` - this one is Nyxus-specific, not a stock Octave
   function, so it's really an analytic cross-check rather than a true oracle call; document it as
   such rather than implying Octave computed it natively.
5. Pin: Octave version (`octave --version`), package versions (`pkg list`), the script, and generation
   date into the header comment - this closes the SPEC 6.4 gap the file's own header comment already
   flags ("no MATLAB version, no exact config and no generator script path is written down").
6. While regenerating, also fix the **UNIFORMITY registry conflict** noted in the CSV: confirm the
   value obtained is the GREYDEPTH=20 Octave/MATLAB one, then correct
   `tests/vetting/oracle_coverage.csv`'s matlab-oracle UNIFORMITY row so `target_test` reads
   `test_2d_firstorder_matlab.h` instead of the stale `test_2d_firstorder_regression.h`.

## After regenerating either oracle

- Update the corresponding `ref_vals_map` table and its provenance comment in the header.
- Re-run `pytest tests/vetting/check_coverage.py` (or the CI job that runs it) so
  `coverage_report.md` regenerates.
- Re-vet, don't re-baseline: if a value moves outside its current tolerance tier, that's either a
  Nyxus bug or a tolerance that was already too loose - don't widen it just to make the new number
  pass (CLAUDE.md / SPEC 7).
