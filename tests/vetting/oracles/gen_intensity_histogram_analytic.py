"""OFFLINE golden generator for the phantom table in tests/test_2d_intensity_histogram_analytic.h.

It covers the whole `_VAL` family plus `IH_ROBUST_MEAN_IDX`. The `_VAL` features carry the
intensity-histogram statistics back into the intensity domain, and no external tool reports them:
MIRP and PyRadiomics both stay in the discretised domain, which is the Nyxus `_IDX` family.
`IH_ROBUST_MEAN_IDX` - the mean over the [P10,P90]-trimmed histogram - is a Nyxus feature with no
counterpart in either tool or in the IBSI reference set, so it has no oracle but this one. So the
oracle here is `analytic` (SPEC 4) - closed forms, written from the published definitions rather
than from intensity_histogram.cpp, and evaluated here so the goldens are reproducible instead of
merely asserted.

    python tests/vetting/oracles/gen_intensity_histogram_analytic.py     # numpy only

Three conventions live in the `_VAL` family, and the point of this generator is that each is stated
and computed independently:

  bin centre        MEAN, VARIANCE, MEDIAN, MODE - the statistic over the bin centre each voxel
                    falls in, c_i = lo + (i - 0.5) * binsize.
  grouped-data      P10, P90, and the IQR/QCoD built from P25/P75 -
  percentile        P(p) = L + binsize * (p*n - F) / f over the bin the CDF crosses. These four are
                    NOT vetted: no reference implementation reproduces them (numpy's nine methods
                    and Octave's nine all answer the sample percentile, which is what the _IDX half
                    reports), so they are `status=regression` drift guards pinned in
                    test_2d_intensity_histogram_regression.h. They are still emitted here because
                    that is where the pinned numbers come from.
  intensity domain  MINIMUM, MAXIMUM, RANGE - the untouched voxel values.

SKEWNESS, EXCESS_KURTOSIS, ENTROPY and UNIFORMITY are invariant under the affine map from bin index
to bin centre (the first two) or depend only on the bin counts (the last two), so their `_VAL` and
`_IDX` values coincide exactly and MIRP vets both; they are not repeated here.
"""
import json
import os
import re

import numpy as np

from ibsi_phantom import phantom_slices

NBINS = 6   # IH_PHANTOM_NBINS in the C++ tests


def discretise(values, nbins=NBINS):
    """IBSI fixed-bin-number discretisation -> (1-based bin index per voxel, lo, binsize)."""
    lo, hi = float(values.min()), float(values.max())
    binsize = (hi - lo) / nbins
    idx = np.clip(np.floor(nbins * (values - lo) / (hi - lo)).astype(int) + 1, 1, nbins)
    return idx, lo, binsize


def histogram_percentile(p, counts, cum, n, lo, binsize):
    target = p * n
    k = int(np.searchsorted(cum, target, side="left"))     # 0-based bin
    before = cum[k - 1] if k else 0.0
    return lo + binsize * (k + (target - before) / counts[k])


def run():
    values = np.array([v for a, m in phantom_slices() for v in a[m > 0].tolist()], float)
    idx, lo, binsize = discretise(values)
    n = len(values)
    counts = np.array([(idx == b).sum() for b in range(1, NBINS + 1)], float)
    cum = np.cumsum(counts)
    centre = lo + (idx - 0.5) * binsize

    def pct(p):
        return histogram_percentile(p, counts, cum, n, lo, binsize)

    p25, p75 = pct(0.25), pct(0.75)
    mean_c, median_c = float(centre.mean()), lo + (int(np.argmax(cum >= 0.5 * n)) + 0.5) * binsize
    sd_c = float(centre.std())
    # the robust window is the voxel set whose bin index lies in [P10_IDX, P90_IDX], i.e. the
    # distribution with its tails trimmed at the same percentiles the _IDX features use
    p10_idx = int(np.argmax(cum >= 0.10 * n)) + 1
    p90_idx = int(np.argmax(cum >= 0.90 * n)) + 1
    window = (idx >= p10_idx) & (idx <= p90_idx)
    robust = centre[window]
    return {
        # bin-centre statistics
        "IH_MEAN_VAL": mean_c,
        "IH_VARIANCE_VAL": float(centre.var()),
        "IH_MEDIAN_VAL": median_c,
        "IH_MODE_VAL": lo + (int(np.argmax(counts)) + 0.5) * binsize,
        "IH_SKEWNESS_VAL": float((((centre - mean_c) / sd_c) ** 3).mean()),
        "IH_EXCESS_KURTOSIS_VAL": float((((centre - mean_c) / sd_c) ** 4).mean() - 3.0),
        "IH_MEAN_ABSOLUTE_DEVIATION_VAL": float(np.abs(centre - mean_c).mean()),
        "IH_MEDIAN_ABSOLUTE_DEVIATION_VAL": float(np.abs(centre - median_c).mean()),
        "IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL": float(np.abs(robust - robust.mean()).mean()),
        "IH_ROBUST_MEAN_VAL": float(robust.mean()),
        "IH_COEFFICIENT_OF_VARIATION_VAL": sd_c / mean_c,
        # the one _IDX feature no tool reports, so it is closed-form here rather than by MIRP;
        # bin indices are 1-based, the convention the whole _IDX family is read in
        "IH_ROBUST_MEAN_IDX": float(idx[window].mean()),
        # shape of the bin-count distribution alone, so identical in either domain
        "IH_ENTROPY_VAL": float(-sum((c / n) * np.log2(c / n) for c in counts if c)),
        "IH_UNIFORMITY_VAL": float(sum((c / n) ** 2 for c in counts)),
        # interpolated histogram percentiles
        "IH_P10_VAL": pct(0.10),
        "IH_P90_VAL": pct(0.90),
        "IH_INTERQUANTILE_RANGE_VAL": p75 - p25,
        "IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL": (p75 - p25) / (p75 + p25),
        # intensity domain
        "IH_MINIMUM_VAL": float(values.min()),
        "IH_MAXIMUM_VAL": float(values.max()),
        "IH_RANGE_VAL": float(values.max() - values.min()),
        # the histogram's own shape
        "IH_NUM_BINS": float(NBINS),
        "IH_BIN_SIZE": binsize,
    }


def verify_pins(header, table, computed, tol=1e-9):
    """Compare EVERY golden pinned in `table` against this run.

    Checking a hand-picked subset is the failure mode this exists to avoid: it silently stops
    covering whatever is added to the table later. Anything the generator cannot produce is
    reported rather than skipped quietly.
    """
    text = re.sub(r"//[^\n]*", "", open(header, encoding="utf-8").read())
    body = text.split(table, 1)[1].split("{", 1)[1].split("};", 1)[0]
    ok, checked, unknown = True, 0, []
    for name, pinned in re.findall(r'\{\s*"(\w+)"\s*,\s*([-0-9.e+]+)\s*\}', body):
        if name not in computed:
            unknown.append(name)
            continue
        got, want = computed[name], float(pinned)
        if abs(got - want) > tol * max(1.0, abs(want), abs(got)):
            print("  FAIL %s: oracle=%r pinned=%s" % (name, got, pinned))
            ok = False
        checked += 1
    print("  %s: verified %d pinned goldens against this run" % (table, checked))
    if unknown:
        print("  not produced by this generator: %s" % ", ".join(sorted(unknown)))
        ok = False
    return ok


def main():
    values = run()
    print("// analytic closed forms on the IBSI digital phantom, fixed bin number, %d bins" % NBINS)
    print("// generated by tests/vetting/oracles/gen_intensity_histogram_analytic.py")
    for feature in sorted(values):
        print('    {"%s", %.17g},' % (feature, values[feature]))
    print("\n// raw:", json.dumps(values, sort_keys=True))

    # Both tables this generator feeds. The robust-window tables in those same files come from the
    # 17-px tail-trimming fixture, not this one, so they are deliberately not listed here.
    print("\n=== re-verify every pinned golden this generator produces ===")
    here = os.path.dirname(os.path.abspath(__file__))
    tests = os.path.join(here, "..", "..")
    ok = verify_pins(os.path.join(tests, "test_2d_intensity_histogram_analytic.h"),
                     "intensity_histogram_2d_analytic_phantom_ref_vals", values)
    ok &= verify_pins(os.path.join(tests, "test_2d_intensity_histogram_regression.h"),
                      "intensity_histogram_2d_regression_ref_vals", values)
    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- do not paste goldens"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
