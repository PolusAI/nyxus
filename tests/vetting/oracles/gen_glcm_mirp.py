"""OFFLINE golden generator for tests/test_2d_glcm_mirp.h (SPEC 6.4).

Runs MIRP GLCM on the same dense phantom as gen_glcm_pyradiomics.py, under recipe
glcm.ibsi_identity, and prints the golden table to paste into the oracle test. Run offline; CI
never invokes it - the reference tool is not a runtime dependency.

    conda create -n nyxus_mirp -c conda-forge python=3.11 mirp
    python tests/vetting/oracles/gen_glcm_mirp.py

Provenance to record at the pinned goldens: tool=mirp, version (printed by this script),
config=`by_slice=True, base_discretisation_method="none", glcm_distance=1, glcm_spatial_method=
"2d_average"`, fixture=the dense phantom below.

MIRP is the IBSI reference implementation, so it reports the two features PyRadiomics dropped as
duplicates - dissimilarity and sum variance - as quantities of their own, which is why the Nyxus
features carrying those names are vetted here rather than through an equality argument.

The fixture is the dense phantom (every grey level 1..8, every level pair populated), a second
configuration next to the IBSI phantom of test_2d_glcm_ibsi.h. Both were run against this tool; see
tests/vetting/audit/glcm_2d_mirp_vetting_report.md.

`2d_average` is the aggregation the Nyxus *_AVE features implement (features computed per angle,
then averaged); the per-angle features are checked against the same golden by averaging the 4
angles in the test.
"""
import json
import logging
import os
import re
from importlib import metadata

import numpy as np
import mirp

# MIRP narrates its progress at INFO into the same stream as the golden table, and it configures
# its logger during the run, so the level has to be suppressed globally rather than set on a logger
logging.disable(logging.INFO)

# MIRP feature column -> the Nyxus features that quantity covers
MIRP_TO_NYXUS = {
    "cm_auto_corr":           ["GLCM_ACOR"],
    "cm_clust_prom":          ["GLCM_CLUPROM"],
    "cm_clust_shade":         ["GLCM_CLUSHADE"],
    "cm_clust_tend":          ["GLCM_CLUTEND"],
    "cm_contrast":            ["GLCM_CONTRAST"],
    "cm_corr":                ["GLCM_CORRELATION"],
    "cm_diff_avg":            ["GLCM_DIFAVE"],
    "cm_diff_entr":           ["GLCM_DIFENTRO"],
    "cm_diff_var":            ["GLCM_DIFVAR"],
    "cm_dissimilarity":       ["GLCM_DIS"],
    "cm_energy":              ["GLCM_ASM", "GLCM_ENERGY"],
    "cm_info_corr1":          ["GLCM_INFOMEAS1"],
    "cm_info_corr2":          ["GLCM_INFOMEAS2"],
    "cm_inv_diff":            ["GLCM_ID", "GLCM_HOM1"],
    "cm_inv_diff_mom":        ["GLCM_IDM", "GLCM_HOM2"],
    "cm_inv_diff_mom_norm":   ["GLCM_IDMN"],
    "cm_inv_diff_norm":       ["GLCM_IDN"],
    "cm_inv_var":             ["GLCM_IV"],
    "cm_joint_avg":           ["GLCM_JAVE"],
    "cm_joint_entr":          ["GLCM_JE", "GLCM_ENTROPY"],
    "cm_joint_max":           ["GLCM_JMAX"],
    "cm_joint_var":           ["GLCM_JVAR", "GLCM_VARIANCE"],
    "cm_sum_avg":             ["GLCM_SUMAVERAGE"],
    "cm_sum_entr":            ["GLCM_SUMENTROPY"],
    "cm_sum_var":             ["GLCM_SUMVARIANCE"],
}

SUFFIX = "_d1_2d_avg"   # distance 1, features averaged over the 4 in-slice angles


def dense_phantom():
    """8x8, every grey level 1..8 present, one-pixel background border -> (intensity, mask)."""
    i, j = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    tile = ((i + 2 * j) % 8) + 1
    assert set(np.unique(tile).tolist()) == set(range(1, 9)), "phantom must be dense in 1..8"
    intensity = np.zeros((1, 10, 10), float)
    mask = np.zeros((1, 10, 10), int)
    intensity[0, 1:9, 1:9] = tile
    mask[0, 1:9, 1:9] = 1
    return intensity, mask


def run():
    intensity, mask = dense_phantom()
    res = mirp.extract_features(image=intensity, mask=mask, by_slice=True,
                                base_discretisation_method="none",   # the phantom is already discrete
                                glcm_distance=1.0, glcm_spatial_method="2d_average")
    df = res[0] if isinstance(res, list) else res
    return {c: float(df.iloc[0][c]) for c in df.columns if c.endswith(SUFFIX)}


def verify_pins(header, table, computed, tol=5e-3):
    """Compare EVERY golden pinned in `table` against this run.

    Checking a hand-picked subset is the failure mode this exists to avoid: it silently stops
    covering whatever is added to the table later. Anything the generator cannot produce is
    reported rather than skipped quietly. The band is the loosest the test file asserts at, which
    is the 5e-3 the five log-based features carry.
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
    print("  verified %d pinned goldens against this run" % checked)
    if unknown:
        print("  not produced by this generator: %s" % ", ".join(sorted(unknown)))
        ok = False
    return ok


def main():
    values = run()
    # same emitter as the PyRadiomics generator: per-angle feature and _AVE twin share the golden,
    # and both names must appear literally for the coverage report to credit them
    # the package exposes no __version__, so the pin comes from the installed distribution
    print("// mirp %s, recipe glcm.ibsi_identity (by_slice + 2d_average + no discretisation),"
          " dense 8x8 phantom" % metadata.version("mirp"))
    print("// generated by tests/vetting/oracles/gen_glcm_mirp.py")
    emitted = {}
    for column in sorted(MIRP_TO_NYXUS):
        key = column + SUFFIX
        if key not in values:
            print("// MISSING from this mirp build: %s" % key)
            continue
        for feature in MIRP_TO_NYXUS[column]:
            names = [feature] if feature == "GLCM_HOM2" else [feature, feature + "_AVE"]
            for n in names:
                emitted[n] = values[key]
                print('    {"%s", %.17g},   // %s' % (n, values[key], column))
    print("\n// raw:", json.dumps(values, sort_keys=True))

    print("\n=== re-verify every pinned golden in test_2d_glcm_mirp.h ===")
    here = os.path.dirname(os.path.abspath(__file__))
    ok = verify_pins(os.path.join(here, "..", "..", "test_2d_glcm_mirp.h"),
                     "glcm_2d_mirp_ref_vals", emitted)
    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- do not paste goldens"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
