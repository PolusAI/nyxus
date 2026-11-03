# Regenerating 2D first-order goldens

## MATLAB

The licensed MATLAB R2026a generator is checked in at
`tests/vetting/oracles/gen_firstorder2d_matlab.m`:

```text
matlab -batch "run('tests/vetting/oracles/gen_firstorder2d_matlab.m')"
```

It downloads `pixelIntensityFeaturesTestData` and `test_2d_firstorder_matlab.h` from the moving
`PolusAI/nyxus` `main` tree, computes 31 values from native MATLAB built-ins, checks the generated
and pinned feature sets in both directions, and prints paste-ready literals. The exact functions,
config, tolerances and three unsupported semantics are recorded in
`firstorder_2d_matlab_vetting_report.md`.

Do not reconstruct `UNIFORMITY`, `MEDIAN_ABSOLUTE_DEVIATION`, or `ROBUST_MEAN` in this generator.
Their dispositions are intentional: PyRadiomics for `UNIFORMITY`, regression for the two semantic
mismatches.

## Octave (percentile rows)

The pinned P01, P10, P25, P75, P90, P99, INTERQUARTILE_RANGE and QCOD are an Octave surrogate,
generated on GNU Octave 11.3.0 by `tests/vetting/oracles/gen_firstorder2d_percentiles_octave.m`:

```text
octave-cli tests/vetting/oracles/gen_firstorder2d_percentiles_octave.m
```

It reads `pixelIntensityFeaturesTestData` and `test_2d_firstorder_matlab.h` from the checkout,
takes `quantile(x, p/100, 1, 7)`, the rule `prctile(..., Method="inclusive")` takes, and fails if
any of the eight values leaves the `rel=1e-9` band of its pinned golden. The 3D twin is
`gen_firstorder3d_percentiles_octave.m`, run the same way.

## PyRadiomics

The existing PyRadiomics provenance and reproduction procedure remain in
`firstorder_2d_pyradiomics_vetting_report.md`. Its 64-bin recipe is distinct from the MATLAB/default
recipe and from the 20-bin entropy regression point.
