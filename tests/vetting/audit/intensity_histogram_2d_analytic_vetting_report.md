# Audit: the 2D intensity-histogram `_VAL` family against closed forms

**Verdict: 19 of 23 quantities are vetted; 4 are demoted to `regression`.** The 19 reproduce their
closed form to 7.5e-15 at worst. The 4 percentile-domain features (`P10_VAL`, `P90_VAL`, `IQR_VAL`,
`QCoD_VAL`) carry no independent reference and are now pinned as drift guards instead. One defect
was found and fixed on the way: `IH_ROBUST_MEAN_IDX` was reported 0-based while the rest of the
index family is 1-based.

Covers the phantom table in `tests/test_2d_intensity_histogram_analytic.h` and
`tests/vetting/oracles/gen_intensity_histogram_analytic.py`.

## Why the oracle is `analytic` and not a tool

MIRP and PyRadiomics both stop at the discretised histogram - bin indices and bin counts - which is
the Nyxus `_IDX` family, vetted against MIRP in
`intensity_histogram_2d_mirp_vetting_report.md`. Neither reports the same statistics carried back
into the intensity domain, so `_VAL` has no tool counterpart to compare against and SPEC 4's
`analytic` oracle applies: closed forms written from the published definitions, evaluated by a
checked-in generator so the goldens are reproducible rather than merely asserted.

The generator imports numpy and nothing else (`conda run -n nyxus_oracle python
gen_intensity_histogram_analytic.py`), and it recomputes the discretisation from the phantom pixels
rather than reading anything out of Nyxus.

## The correction this report exists to make

`config_recipes.md` described `_VAL` as "the same statistics over bin centers = affine transform of
the IDX distribution (VAL = binWidth·IDX for spreads; +minVal offset for locations)". Measured at
recipe `ih.ibsi_fbn` on the phantom, that rule is right for seven features, wrong in degree for one,
and describes the wrong quantity entirely for eight more. **Five conventions coexist:**

| convention | map from `_IDX` | features | worst deviation |
|---|---|---|---:|
| bin-centre, location | `VAL = lo + (IDX-0.5)·b` | MEAN, MEDIAN, MODE, ROBUST_MEAN | 2.2e-16 |
| bin-centre, scale | `VAL = b·IDX` | MAD, MEDAD, RMAD | 1.2e-16 |
| bin-centre, squared scale | `VAL = b²·IDX` | VARIANCE | 2.1e-16 |
| domain-invariant | `VAL = IDX` | SKEWNESS, EXCESS_KURTOSIS, ENTROPY, UNIFORMITY | 1.3e-15 |
| **no map exists** | — | P10, P90, IQR, QCoD, MIN, MAX, RANGE, COV | — |

`b` is `IH_BIN_SIZE` and `lo` the ROI minimum. So the bin-centre group *is* affine in the underlying
distribution - the centre map `c = lo + (i-0.5)·b` is affine in `i` - but each statistic carries it
with its own degree of homogeneity, which is why one blanket rule cannot cover MEAN, MAD and
VARIANCE at once. `VARIANCE` in particular is off by a whole factor of `b` under the documented
rule.

The eight features with no map are three separate cases:

- **MINIMUM, MAXIMUM, RANGE** are the untouched voxel values. `IH_MAXIMUM_VAL` is 6, the largest
  intensity in the mask; the affine image of `IH_MAXIMUM_IDX = 6` would be 5.583.
- **P10, P90, IQR, QCoD** are the grouped-data percentile
  `P(p) = L + b·(n·p - F)/f` over the bin the CDF crosses. `IH_P90_VAL` is 4.3125 where the affine
  image of `IH_P90_IDX = 4` would be 3.917. **These four are demoted — see below.**
- **COV** is `sqrt(VARIANCE_VAL)/MEAN_VAL`, a ratio of two differently-scaled quantities, so the
  `b` does not cancel to anything expressible in `COEFFICIENT_OF_VARIATION_IDX`: `b·COV_IDX` is
  0.6768 against the reported 0.6126.

## Demoted: the four percentile-domain `_VAL` features

These were the ones to be suspicious of, because the closed form reproduced Nyxus to 1e-16 — which
proves only that the generator restates Nyxus' own method. The test is whether a *reference
implementation's own percentile function* lands on the same number, so both tools were run at all
nine of their documented methods on the same 74 voxels:

| p | Nyxus `_VAL` | Nyxus `_IDX` | numpy, all 9 methods | Octave `quantile`, methods 1–4,7 | Octave 5 (`prctile`) | Octave 6 | Octave 8 | Octave 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 1.1233333 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.25 | 1.3083333 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.75 | 3.734375 | 4 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| 0.90 | **4.3125** | **4** | **4.0** | 4.0 | **4.2** | 5.0 | 4.4667 | 4.4 |

`IH_P90_VAL = 4.3125` matches **none** of the eighteen native results. It is not the tools' default
(4.0), and it is not 4.2 — the interpolated variant the IBSI reference manual names as a
*non-reference* alternative, which Octave's `prctile` happens to reproduce exactly. The gap to
`prctile` is 2.7%.

That is the same shape, and roughly the same magnitude, as the six first-order percentile rows
demoted in PR #427: a golden that agrees with Nyxus' internal histogram interpolation to 15 digits
while diverging from every real `prctile`/`quantile`. Per SPEC, vetting is agreement with an
*independent* reference, and there is none here — so `IH_P10_VAL`, `IH_P90_VAL`,
`IH_INTERQUANTILE_RANGE_VAL` and `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL` move to
`status=regression` with the goldens kept as drift guards in
`test_2d_intensity_histogram_regression.h`.

Two things this does **not** say. It is not a claim that Nyxus is wrong: the grouped-data
percentile is a reasonable thing for a histogram-domain feature to report, and the divergence is a
definitional choice. And it does not touch the `_IDX` half — `IH_P10_IDX`/`IH_P90_IDX` are the
discrete grey-level percentile, they equal what all eighteen native methods return by default, MIRP
agrees to double precision, and they stay `vetted`.

Reproduce with `quantile(v, p, 1, method)` in Octave — the third argument is `DIM`, so a
three-argument call silently computes along a dimension instead of selecting a method.

## Result table

The 19 vetted rows. The four demoted ones agree with the closed form just as tightly
(`IQR_VAL` 3.7e-16, `QCoD_VAL` 3.5e-16, `P10_VAL` 0.0, `P90_VAL` 2.1e-16) — which is the point:
tight agreement with a restatement of Nyxus' own method is not evidence, so they are listed in the
demotion section above rather than here.

| feature | Nyxus | closed form | rel |
|---|---|---|---:|
| `IH_BIN_SIZE` | 0.8333333333333334 | 0.8333333333333334 | 0.0e+00 |
| `IH_COEFFICIENT_OF_VARIATION_VAL` | 0.6126160317845603 | 0.6126160317845603 | 0.0e+00 |
| `IH_ENTROPY_VAL` | 1.2656115555865246 | 1.2656115555865246 | 0.0e+00 |
| `IH_EXCESS_KURTOSIS_VAL` | -0.354620480687835 | -0.35462048068783236 | 7.5e-15 |
| `IH_MAXIMUM_VAL` | 6 | 6 | 0.0e+00 |
| `IH_MEAN_ABSOLUTE_DEVIATION_VAL` | 1.2935232529827128 | 1.2935232529827125 | 1.7e-16 |
| `IH_MEAN_VAL` | 2.3738738738738743 | 2.373873873873874 | 1.9e-16 |
| `IH_MEDIAN_ABSOLUTE_DEVIATION_VAL` | 0.9572072072072073 | 0.9572072072072073 | 0.0e+00 |
| `IH_MEDIAN_VAL` | 1.4166666666666667 | 1.4166666666666667 | 0.0e+00 |
| `IH_MINIMUM_VAL` | 1 | 1 | 0.0e+00 |
| `IH_MODE_VAL` | 1.4166666666666667 | 1.4166666666666667 | 0.0e+00 |
| `IH_NUM_BINS` | 6 | 6 | 0.0e+00 |
| `IH_RANGE_VAL` | 5 | 5 | 0.0e+00 |
| `IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL` | 0.9281948466622115 | 0.9281948466622112 | 3.6e-16 |
| `IH_ROBUST_MEAN_IDX` | 1.7462686567164178 | 1.7462686567164178 | 0.0e+00 |
| `IH_ROBUST_MEAN_VAL` | 2.0385572139303485 | 2.038557213930348 | 2.2e-16 |
| `IH_SKEWNESS_VAL` | 1.0838207225574554 | 1.0838207225574572 | 1.6e-15 |
| `IH_UNIFORMITY_VAL` | 0.5124178232286339 | 0.5124178232286339 | 0.0e+00 |
| `IH_VARIANCE_VAL` | 2.1149105186267354 | 2.114910518626735 | 2.1e-16 |

`SKEWNESS`, `EXCESS_KURTOSIS`, `ENTROPY` and `UNIFORMITY` are the four the generator does not
recompute independently of their `_IDX` partners - they are invariant under the centre map or depend
only on the bin counts - so MIRP vets both halves and the closed forms here only confirm the
invariance holds numerically.

## Defect found: `IH_ROBUST_MEAN_IDX` was 0-based

`intensity_histogram.cpp` accumulates every bin index over `i = 0 .. N-1` and shifts each one to the
family's 1-based reporting convention as it is emitted - `meanIndex + 1`, `medianIndex + 1`,
`minimumIndex + 1`, `p10Index + 1`, `p90Index + 1`, `maximumIndex + 1`, `modeIndex + 1`,
`maximumGradientIndex` and `minimumGradientIndex` both seeded at `i + 1`. `robustMeanIndex` was
emitted bare, so it came out exactly 1 low.

Three independent measurements, all taken before the fix:

| check | expected | reported |
|---|---|---|
| trimmed mean of the 1-based indices, IBSI phantom | 1.7462686567164178 | 0.746268656716418 |
| the same, on the 17-px tail-trimming fixture | 2.9333333 | 1.9333333 |
| the family's own centre map `lo + (IDX-0.5)·b`, against `IH_ROBUST_MEAN_VAL` | 2.038557213930348 | 1.205223880597015 |

`IH_ROBUST_MEAN_VAL` was correct throughout; only the index half was shifted. Two corroborating
details: `coefficientOfVariationIndex` in the same function divides by `(meanIndex + 1)`, so the code
already treats the 1-based value as the reportable one; and `docs/source/featurelist.rst` states
plainly that "the `...IDX` features [are computed] over the (1-based) bin indices", so the published
contract was the one the other eight features kept.

**How it survived.** The feature's only assertion was a golden in
`test_2d_intensity_histogram_dispersion_robust_analytic` labelled "hand-computed ... derived
independently of intensity_histogram.cpp". It was not: at 1.933333333 it matches the code and not
the definition. This is the circular-provenance hazard the family-PR series exists to catch, and it
was reachable only because running the MIRP coverage invariant forced the question of what vets the
one feature MIRP cannot.

**Fixed here.** `robustMeanIndex + 1` at the emit site, which is on the shared `compute()` path, so
the in-RAM and out-of-core implementations move together. Both goldens are corrected: the phantom
value now comes from the generator, and the 17-px fixture's is derived in a comment from the bin
counts. Users of `IH_ROBUST_MEAN_IDX` will see values 1 higher than in previous releases.

## What this report does and does not establish

The goldens come from the generator, so "golden == fresh run" only shows the pin is reproducible.
The vetting claim rests on the closed forms being written from the published definitions rather than
read out of `intensity_histogram.cpp` - which is precisely the property the previous
`IH_ROBUST_MEAN_IDX` golden lacked, and the reason it is worth stating separately each time.
