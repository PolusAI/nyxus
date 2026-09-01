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

## PyRadiomics

The existing PyRadiomics provenance and reproduction procedure remain in
`firstorder_2d_pyradiomics_vetting_report.md`. Its 64-bin recipe is distinct from the MATLAB/default
recipe and from the 20-bin entropy regression point.
