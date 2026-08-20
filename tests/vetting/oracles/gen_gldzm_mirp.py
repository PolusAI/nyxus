"""OFFLINE mirp oracle for the 2D GLDZM features, on the IBSI digital phantom.

    python tests/vetting/oracles/gen_gldzm_mirp.py        (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_2d_gldzm_mirp.h -- both
the per-slice table and the four-slice means -- exiting non-zero on any mismatch or on any pin this
generator cannot produce.

Recipe `gldzm.ibsi_phantom_2d`: the four IBSI digital-phantom slices, each featurised on its own.
Each slice is its own ROI and yields its own scalar; their mean is what the IBSI "2D, averaged"
aggregation publishes. Both quantities are pinned, because a mean is weaker than the four values
behind it -- errors in two slices that cancel leave it unmoved, and on this family the disagreement
that mattered lived in one slice. mirp runs `by_slice=True` with `base_discretisation_method="none"`
(the phantom is already discrete 1..6), which is what Nyxus computes in IBSI mode.

Agreement is 8.9e-16 worst case: mirp and Nyxus build the same 8-connected zones over the same
distance-to-border metric, so this is pinned at the SPEC 7 "exact" tier rather than a cross-tool
band.

NOT covered here: GLDZM_GLM and GLDZM_ZDM. mirp exposes no grey-level-mean or zone-distance-mean
column -- neither is an IBSI GLDZM feature -- so those two stay regression rows in
test_2d_gldzm_regression.h.

Provenance: tool=mirp (version printed by this script); numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_gldzm_mirp.py. Run offline; CI never invokes it.
"""
import importlib.util
import logging
import os
import re
from importlib import metadata

import numpy as np

logging.disable(logging.INFO)          # mirp logs at INFO onto stdout, into the golden table

import mirp

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
TEST_H = os.path.join(TESTS, "test_2d_gldzm_mirp.h")

RELTOL = 1e-9          # SPEC 7 exact tier; measured worst residual is 8.9e-16

# Nyxus feature -> mirp column stem. mirp writes "zone" where Nyxus writes "zone level", and
# abbreviates emphasis as lge/hge against Nyxus' LGLZE/HGLZE, so map by meaning, not by spelling.
MIRP_COLUMN = {
    "GLDZM_SDE": "dzm_sde",              # small distance emphasis
    "GLDZM_LDE": "dzm_lde",              # large distance emphasis
    "GLDZM_LGLZE": "dzm_lgze",           # low grey level zone emphasis
    "GLDZM_HGLZE": "dzm_hgze",           # high grey level zone emphasis
    "GLDZM_SDLGLE": "dzm_sdlge",
    "GLDZM_SDHGLE": "dzm_sdhge",
    "GLDZM_LDLGLE": "dzm_ldlge",
    "GLDZM_LDHGLE": "dzm_ldhge",
    "GLDZM_GLNU": "dzm_glnu",
    "GLDZM_GLNUN": "dzm_glnu_norm",
    "GLDZM_ZDNU": "dzm_zdnu",
    "GLDZM_ZDNUN": "dzm_zdnu_norm",
    "GLDZM_ZP": "dzm_z_perc",            # zone percentage
    "GLDZM_GLV": "dzm_gl_var",
    "GLDZM_ZDV": "dzm_zd_var",
    "GLDZM_ZDE": "dzm_zd_entr",
}


def phantom_slices():
    """The IBSI phantom, read out of tests/test_data.h by the shared helper."""
    spec = importlib.util.spec_from_file_location(
        "ibsi_phantom", os.path.join(HERE, "ibsi_phantom.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.phantom_slices()


def parse_pins(txt, table):
    # the opening brace may sit on the declaration line or the next one, so match across whitespace
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found in {os.path.basename(TEST_H)}")
    body = txt[m.end():].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def run():
    """-> {feature: [value per phantom slice, in z order]}."""
    per_slice = {k: [] for k in MIRP_COLUMN}
    for intensity, mask in phantom_slices():
        image = np.asarray(intensity, dtype=float)[None, :, :]
        roi = (np.asarray(mask) > 0).astype(int)[None, :, :]
        res = mirp.extract_features(image=image, mask=roi, by_slice=True,
                                    base_discretisation_method="none")
        df = res[0] if isinstance(res, list) else res
        for feature, stem in MIRP_COLUMN.items():
            col = [c for c in df.columns if c == stem + "_2d"]
            if not col:
                raise RuntimeError(f"mirp produced no column for {feature} ({stem})")
            per_slice[feature].append(float(df.iloc[0][col[0]]))
    return per_slice


def verify(pins, got, label):
    """Compares one pinned table against this run. -> (verified, failed, unproducible)."""
    print("")
    print(f"# verifying {len(pins)} pinned {label} goldens against this run")
    nok = nfail = nmiss = 0
    for name in sorted(pins):
        want = pins[name]
        if name not in got:
            print(f"  MISSING {name}: pinned {want!r} but mirp produces no such value")
            nmiss += 1
            continue
        have = got[name]
        rel = abs(have - want) / max(abs(want), 1e-12)
        if rel <= RELTOL:
            print(f"  OK   {name}: mirp={have!r} pinned={want!r} rel={rel:.3g}")
            nok += 1
        else:
            print(f"  FAIL {name}: mirp={have!r} pinned={want!r} rel={rel:.3g}")
            nfail += 1
    return nok, nfail, nmiss


def main():
    try:
        version = metadata.version("mirp")       # mirp exposes no __version__
    except metadata.PackageNotFoundError:
        version = "unknown"

    slices = run()
    means = {k: float(np.mean(v)) for k, v in slices.items()}
    by_slice = {f"{k}_z{z + 1}": v for k, vs in slices.items() for z, v in enumerate(vs)}

    print(f"# mirp {version}, numpy {np.__version__}, recipe gldzm.ibsi_phantom_2d")
    print("# paste-ready goldens (mean over the four IBSI phantom slices)")
    for name in sorted(means):
        print(f'\t{{"{name}", {means[name]!r}}},')
    print("# paste-ready per-slice goldens")
    for name in sorted(by_slice):
        print(f'\t{{"{name}", {by_slice[name]!r}}},')

    if not os.path.exists(TEST_H):
        print("")
        print(f"{os.path.basename(TEST_H)} does not exist yet - nothing to verify")
        return 0

    # Both tables the header pins, not just the means: a per-slice golden nothing re-derives is
    # exactly the kind of pin that stops being checked the moment it is written.
    text = open(TEST_H, encoding="utf-8", errors="replace").read()
    nok, nfail, nmiss = verify(parse_pins(text, "gldzm_2d_mirp_ref_vals"), means, "mean")
    a, b, c = verify(parse_pins(text, "gldzm_2d_mirp_slice_ref_vals"), by_slice, "per-slice")
    nok, nfail, nmiss = nok + a, nfail + b, nmiss + c

    print("")
    print(f"{nok} verified, {nfail} failed, {nmiss} unproducible")
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
