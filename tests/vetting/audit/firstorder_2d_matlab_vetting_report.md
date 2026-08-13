# Audit: `test_2d_firstorder_matlab.h` goldens vs. an independent Octave run

**Verdict: the "matlab" oracle label is not backed by an actual independent MATLAB/Octave run for
10 of the 34 features. The percentile-family values (P01, P10, P25, P75, P90, P99,
INTERQUARTILE_RANGE, QCOD, ROBUST_MEAN, ROBUST_MEAN_ABSOLUTE_DEVIATION) match Nyxus's own
non-standard percentile algorithm exactly, and diverge from what real Octave's `prctile`/`quantile`
actually compute on the same data by up to ~2.5% — far outside the file's own 0.1% default
tolerance. The remaining 24 features (basic moments/extrema/energy) do check out, because they
happen to use textbook formulas both tools agree on.**

## Method

- Environment: local conda env `octave_verify` (`/home/samee/miniforge3/envs/octave_verify`),
  GNU Octave 10.3.0 + `octave-statistics` 1.8.x from conda-forge — no Docker needed, this was
  already provisioned on the machine.
- Fixture: `pixelIntensityFeaturesTestData` parsed directly out of `tests/test_data.h` by regex
  (not hand-copied) — confirmed 154 pixels, `sum(intensities) = 5015224` (matches the golden
  `INTEGRATED_INTENSITY` exactly, a good early consistency check).
- Two independent computations were run against the same 154-value intensity vector:
  1. **A faithful Python re-implementation of Nyxus's own algorithm**, read directly out of
     `src/nyx/features/intensity.cpp` and `src/nyx/features/histogram.h` — its 100-bin
     linear-interpolation percentile method (`TrivialHistogram::calc_percentiles`, independent of
     the `GREYDEPTH` setting), its sample-stddev-denominator hyperskewness/hyperflatness, etc.
  2. **Real GNU Octave**, using its stock `mean`/`median`/`std`/`var`/`skewness`/`kurtosis`/`prctile`/
     `quantile` functions — this is what an actual "vetted against MATLAB" claim should mean.

## Result table

| Feature | Golden (file) | (1) Nyxus-algorithm replica | (2) Real Octave | Verdict |
|---|---|---|---|---|
| MEAN | 32566.38961038961 | 32566.38961038961 | 32566.3896103896 | ✅ agrees with both |
| MEDIAN | 29803.5 | 29803.5 | 29803.5 | ✅ agrees with both |
| MODE | 19552 | 19552 | — (no stock mode() needed, trivial) | ✅ |
| MIN / MAX / RANGE | 11079 / 64090 / 53011 | same | same | ✅ trivial |
| STANDARD_DEVIATION (sample) | 14730.96831710767 | 14730.968317107667 | 14730.9683171077 | ✅ agrees with both |
| STANDARD_DEVIATION_BIASED (pop) | 14683.06260221863 | 14683.062602218628 | 14683.0626022186 | ✅ agrees with both |
| VARIANCE (sample) | 217001427.5596299 | 217001427.5596299 | 217001427.55963 | ✅ agrees with both |
| VARIANCE_BIASED (pop) | 215592327.3806713 | 215592327.38067126 | 215592327.380671 | ✅ agrees with both |
| SKEWNESS | 0.450256759704494 | 0.45025675970449425 | 0.450256759704494 | ✅ agrees with both (Octave's *default* `skewness()` happens to be the population/biased convention Nyxus uses — its `skewness(x,0)` unbiased variant gives 0.4547, a different number) |
| KURTOSIS | 1.927888720710090 | 1.9278887207100899 | 1.92788872071009 | ✅ agrees with both (same caveat: Octave default, not the unbiased variant) |
| EXCESS_KURTOSIS | -1.07211127928991 | -1.0721112792899101 | (kurtosis-3 = -1.0721112792899) | ✅ |
| MEAN_ABSOLUTE_DEVIATION | 12833.08449991567 | 12833.084499915669 | — (matches by construction, plain formula) | ✅ |
| STANDARD_ERROR | 1187.055255225567 | 1187.0552552255667 | — (std/sqrt(n), plain formula) | ✅ |
| ROOT_MEAN_SQUARED / ENERGY | 35723.41.../196528957184 | same | — (plain formulas) | ✅ |
| INTEGRATED_INTENSITY | 5015224 | 5015224 | — (plain sum) | ✅ |
| **P01** | **11895.3694** | **11895.369400000001** | **12081.4** | ❌ golden matches (1), not real Octave `prctile` — diverges 1.56% |
| **P10** | **16107.472** | **16107.472** | **16329** | ❌ diverges 1.36% from real Octave |
| **P25** | **19074.82583333333** | **19074.825833333332** | **19552** | ❌ diverges 2.44% from real Octave |
| **P75** | **45801.205** | **45801.205** | **45723** | ❌ diverges 0.17% from real Octave |
| **P90** | **53381.778** | **53381.778** | **53360.7** | ❌ diverges 0.04% (smallest gap, still not exact) |
| **P99** | **63416.7603** | **63416.7603** | **63380.96** | ❌ diverges 0.06% from real Octave |
| **INTERQUARTILE_RANGE** | **26726.37916666667** | **26726.379...** | **26171** (75-25 via prctile) | ❌ diverges ~2.1% |
| **QCOD** | **0.411960763064047** | **0.4119607630640474** | not a stock function; derived from prctile P25/P75 above | ❌ same divergence propagates |
| **ROBUST_MEAN** | **31421.368** | **31421.368** | (mean of values within [P10,P90] via real prctile bounds ≈ different subset) | ❌ depends on the diverging P10/P90 |
| **ROBUST_MEAN_ABSOLUTE_DEVIATION** | **10440.618496000001** | **10440.618496000001** | same dependency | ❌ same root cause |
| HYPERSKEWNESS | 1.978293086605381 | 1.9782930866053812 | n/a — not a MATLAB toolbox function | ⚠️ not oracle-vettable in the stock-MATLAB sense; internally self-consistent with Nyxus's own documented formula (denominator uses the *sample* stddev, an unusual choice worth a comment where it's used) |
| HYPERFLATNESS | 5.126659243028459 | 5.126659243028459 | n/a | ⚠️ same as above |
| UNIFORMITY_PIU | 29.477577192725725 | 29.477577192725725 | n/a (PIU formula, not a stats-toolbox function) | ⚠️ Nyxus-specific, analytically confirmed correct, not "MATLAB-vetted" |
| UNIFORMITY (GREYDEPTH=20) | 0.0647664 | 0.06476640242874007 | n/a | ⚠️ Nyxus-specific histogram formula |
| COV | 0.4523365498399634 | 0.4523365498399634 | trivially std/mean | ✅ |
| MEDIAN_ABSOLUTE_DEVIATION | 12693.84415584416 | 12693.844155844155 | trivially mean(abs(x-median)) | ✅ |
| COVERED_IMAGE_INTENSITY_RANGE | 0.8088960097657740 | (roi_range)/(slide_range) = 53011/65535 = 0.808896009765774 | n/a, Nyxus-specific + needs slide props | ✅ pure arithmetic, confirmed |

## Root cause

Nyxus's percentile calculation (`TrivialHistogram::calc_percentiles`, `src/nyx/features/histogram.h`)
bins intensities into a **fixed 100-bin histogram** over `[min, max]` and linearly interpolates the
target rank *within the bin it falls in*. This is a legitimate percentile-estimation method, but it
is **not** the piecewise-linear-on-sorted-values method `prctile`/`quantile` use in MATLAB or Octave.
The two methods agree closely only when a bin happens to contain few points near the target
percentile; they diverge by 1–2.5% here because several of `bins100_`'s 100 bins are sparsely
populated on this 154-pixel fixture, so a whole bin's width gets attributed to a single interpolation
step.

Since every percentile-family golden in the file matches (1) — the Nyxus-side replica — to 10+
significant digits, and none of them match (2) — real Octave — these numbers were almost certainly
captured from **Nyxus's own live output**, not from an actual MATLAB/Octave run, despite the file
being named/labeled `_matlab` and its header comment asserting "All 34 values below are MATLAB
reference values." That comment is very likely inaccurate for at least these 10 features.

## What this means for `oracle_coverage.csv`

Per SPEC's definition ("vetting is a property of a (feature × config × reference) assertion"), the
10 percentile-family rows currently marked `outcome=vetted, oracle=matlab` are **not actually
vetted** — the assertion only checks Nyxus against itself. They should be downgraded to
`regression` (self-snapshot) until either:
- a real MATLAB/Octave `prctile`/`quantile`-based golden is generated and the test tolerance is
  widened to the ~2-3% these two legitimate-but-different percentile conventions actually differ
  by (a "definitional" tier per SPEC 7, same treatment already given to VARIANCE vs pyradiomics's
  Bessel-factor gap), or
- Nyxus's percentile algorithm itself is reconsidered (100 fixed bins is coarse for small ROIs;
  IBSI's own percentile convention doesn't bin at all).

The 24 other rows (basic moments, extrema, energy, and the Nyxus-specific formulas with no MATLAB
equivalent) hold up fine and can stay `vetted`/self-consistent as currently recorded — this is not
a wholesale problem with the file, just with the percentile-derived third.

## Reproducing this

```
source /home/samee/miniforge3/bin/activate octave_verify   # provides Octave 10.3.0 + statistics pkg
# fixture_intensities.csv: one intensity per line, extracted from tests/test_data.h's
# pixelIntensityFeaturesTestData via regex (see this report's method section)
octave-cli -q --eval "pkg load statistics; x=dlmread('fixture_intensities.csv'); disp(prctile(x,10))"
```
