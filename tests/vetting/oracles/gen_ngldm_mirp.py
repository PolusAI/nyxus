"""OFFLINE mirp oracle for the 2D NGLDM features, on the IBSI digital phantom.

    python tests/vetting/oracles/gen_ngldm_mirp.py        (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_2d_ngldm_mirp.h, exiting
non-zero on any mismatch or on any pin this generator cannot produce.

Recipe `ngldm.ibsi_phantom_2d`: the four IBSI digital-phantom slices, each featurised on its own and
the per-feature values averaged over the four, which is what test_2d_ngldm_ibsi.h's helper does.
mirp runs `by_slice=True` with `base_discretisation_method="none"` (the phantom is already discrete
1..6), `ngldm_distance=1` and `ngldm_difference_level=0` -- the alpha=0, d=1 coarseness the IBSI
NGLDM definition uses, and what Nyxus computes in IBSI mode.

Agreement is 2.9e-16 worst case: mirp and Nyxus implement the same definition over the same
neighbourhood, so this is pinned at the SPEC 7 "exact" tier rather than a cross-tool band. The IBSI
consensus values in test_2d_ngldm_ibsi.h are the same quantities quoted to three significant
figures, which is why that file asserts at rel=1e-2 while this one asserts at 1e-9.

NOT covered here: NGLDM_GLM and NGLDM_DCM. mirp exposes no grey-level-mean or
dependence-count-mean column -- they are not IBSI NGLDM features -- so those two stay regression
rows in test_2d_ngldm_regression.h.

Provenance: tool=mirp (version printed by this script); numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_ngldm_mirp.py. Run offline; CI never invokes it.
"""
import importlib.util
import logging
import os
import re
import sys
from importlib import metadata

import numpy as np

logging.disable(logging.INFO)          # mirp logs at INFO onto stdout, into the golden table

import mirp

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
TEST_H = os.path.join(TESTS, "test_2d_ngldm_mirp.h")

RELTOL = 1e-9          # SPEC 7 exact tier; measured worst residual is 2.9e-16

# Nyxus feature -> mirp column stem (mirp suffixes the discretisation/config, e.g. _d1_a0.0_2d)
MIRP_COLUMN = {
    "NGLDM_LDE": "ngl_lde", "NGLDM_HDE": "ngl_hde",
    "NGLDM_LGLCE": "ngl_lgce", "NGLDM_HGLCE": "ngl_hgce",
    "NGLDM_LDLGLE": "ngl_ldlge", "NGLDM_LDHGLE": "ngl_ldhge",
    "NGLDM_HDLGLE": "ngl_hdlge", "NGLDM_HDHGLE": "ngl_hdhge",
    "NGLDM_GLNU": "ngl_glnu", "NGLDM_GLNUN": "ngl_glnu_norm",
    "NGLDM_DCNU": "ngl_dcnu", "NGLDM_DCNUN": "ngl_dcnu_norm",
    "NGLDM_GLV": "ngl_gl_var", "NGLDM_DCP": "ngl_dc_perc",
    "NGLDM_DCV": "ngl_dc_var", "NGLDM_DCENT": "ngl_dc_entr",
    "NGLDM_DCENE": "ngl_dc_energy",
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
            re.findall(r'\{\s*"([A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def run():
    """-> {feature: value averaged over the four phantom slices}."""
    per_slice = {k: [] for k in MIRP_COLUMN}
    for intensity, mask in phantom_slices():
        image = np.asarray(intensity, dtype=float)[None, :, :]
        roi = (np.asarray(mask) > 0).astype(int)[None, :, :]
        res = mirp.extract_features(image=image, mask=roi, by_slice=True,
                                    base_discretisation_method="none",
                                    ngldm_distance=1.0, ngldm_difference_level=0.0)
        df = res[0] if isinstance(res, list) else res
        for feature, stem in MIRP_COLUMN.items():
            col = [c for c in df.columns if c.startswith(stem + "_d1")]
            if not col:
                raise RuntimeError(f"mirp produced no column for {feature} ({stem})")
            per_slice[feature].append(float(df.iloc[0][col[0]]))
    return {k: float(np.mean(v)) for k, v in per_slice.items()}


def main():
    try:
        version = metadata.version("mirp")       # mirp exposes no __version__
    except metadata.PackageNotFoundError:
        version = "unknown"

    got = run()

    print(f"# mirp {version}, numpy {np.__version__}, recipe ngldm.ibsi_phantom_2d")
    print("# paste-ready goldens (mean over the four IBSI phantom slices)")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},')

    if not os.path.exists(TEST_H):
        print(f"\n{os.path.basename(TEST_H)} does not exist yet - nothing to verify")
        return 0

    pins = parse_pins(open(TEST_H, encoding="utf-8", errors="replace").read(),
                      "ngldm_2d_mirp_ref_vals")
    print(f"\n# verifying {len(pins)} pinned goldens against this run")
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

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible")
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
