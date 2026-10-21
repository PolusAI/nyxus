"""OFFLINE mirp oracle for the 2D NGLDM features, on the IBSI digital phantom and bench_square12_graded.

    python tests/vetting/oracles/gen_ngldm_mirp.py           (from the repository root)
    python tests/vetting/oracles/gen_ngldm_mirp.py --check   (verify only, no paste-ready tables)

Prints the paste-ready goldens of both recipes AND re-verifies every golden pinned in
test_2d_ngldm_mirp.h, exiting non-zero on any mismatch or on any pin this generator cannot produce.

Recipe `ngldm.ibsi_phantom_2d`: the four IBSI digital-phantom slices, each featurised on its own and
the per-feature values averaged over the four, which is what test_2d_ngldm_common.h's fixture does.
mirp runs `by_slice=True` with `base_discretisation_method="none"` (the phantom is already discrete
1..6), `ngldm_distance=1` and `ngldm_difference_level=0` -- the alpha=0, d=1 coarseness the IBSI
NGLDM definition uses, and what Nyxus computes in IBSI mode.

Agreement is 2.9e-16 worst case: mirp and Nyxus implement the same definition over the same
neighbourhood, so this is pinned at the SPEC 7 "exact" tier rather than a cross-tool band. The IBSI
consensus values in test_2d_ngldm_ibsi.h are the same quantities quoted to three significant
figures, which is why that file asserts at rel=1e-2 while this one asserts at 1e-9.

Recipe `ngldm.mirp_fbn20_negdepth`, table ngldm_2d_mirp_negdepth_ref_vals: the negative coarse grey
depth. GREYDEPTH=-20 bins the ROI into 20 PyRadiomics-style bins over its own [min, max], which is
IBSI's fixed bin number, floor(20 (x - min) / (max - min)) + 1, and exactly what mirp's
`base_discretisation_method="fixed_bin_number"`, n_bins=20, computes. So mirp discretises here
itself: nothing of Nyxus' binning is reproduced on the mirp side, and the comparison covers the
binning as well as the matrix and the feature formulas. The fixture is bench_square12_graded, built
by square12_graded.py (shared with gen_gldzm_mirp.py), `by_slice=True`, distance 1, difference
level 0. 14 of its 144 pixels sit exactly on a bin edge at 20 bins, and the 5 at 181 are the ones a
division by the bin width sends one bin low; square12_graded.py has the arithmetic. Every run also
checks that test_main_nyxus.h still builds the square from the same formula. NGLDM_DCP is left out
of this table: both tools return 1 whenever every pixel has a same-level neighbour. The pins are
printed at full precision, so they are held at rel 1e-12 -- a staleness check, not a vetting band.

NOT covered here: NGLDM_GLM and NGLDM_DCM. mirp exposes no grey-level-mean or
dependence-count-mean column -- they are not IBSI NGLDM features -- so those two stay regression
rows in test_2d_ngldm_regression.h.

Provenance: tool=mirp (version printed by this script); numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_ngldm_mirp.py, with tests/vetting/oracles/square12_graded.py.
Run offline; CI never invokes it.
"""
import argparse
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
NEGDEPTH_RELTOL = 1e-12   # the negdepth pins are printed at %.17g; a staleness check, not a band
NEGDEPTH_TABLE = "ngldm_2d_mirp_negdepth_ref_vals"

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


def _helper(name):
    """A shared fixture module from this directory, loaded by path."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def phantom_slices():
    """The IBSI phantom, read out of tests/test_data.h by the shared helper."""
    return _helper("ibsi_phantom").phantom_slices()


SQUARE = _helper("square12_graded")
NEGDEPTH_SCOPE = {k: v for k, v in MIRP_COLUMN.items() if k != "NGLDM_DCP"}


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


def verify_negdepth(pins, got):
    """-> (verified, failed, unproducible) for the negdepth table: scope and values both."""
    print(f"\n# verifying {len(pins)} goldens pinned in {NEGDEPTH_TABLE} against this run, "
          f"at rel<={NEGDEPTH_RELTOL:g}")
    nok = nfail = nmiss = 0
    for name in sorted(set(pins) - set(got)):
        print(f"  MISSING {name}: pinned {pins[name]!r} but is outside this table's scope")
        nmiss += 1
    for name in sorted(set(got) - set(pins)):
        print(f"  FAIL {name}: mirp computes it in this table's scope and the header does not pin it")
        nfail += 1
    for name in sorted(set(pins) & set(got)):
        have, want = got[name][0], pins[name]
        rel = abs(want - have) / max(abs(have), 1e-300)
        if rel <= NEGDEPTH_RELTOL:
            print(f"  OK   {name}: mirp={have!r} pinned={want!r} rel={rel:.3g}")
            nok += 1
        else:
            print(f"  FAIL {name}: mirp={have!r} pinned={want!r} rel={rel:.3g}")
            nfail += 1
    return nok, nfail, nmiss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the pins without printing the paste-ready tables")
    args = ap.parse_args()

    try:
        version = metadata.version("mirp")       # mirp exposes no __version__
    except metadata.PackageNotFoundError:
        version = "unknown"

    got = run()
    negdepth = SQUARE.mirp_fbn("ngldm", NEGDEPTH_SCOPE)

    print(f"# mirp {version}, numpy {np.__version__}, "
          f"recipes ngldm.ibsi_phantom_2d and ngldm.mirp_fbn20_negdepth")
    if not args.check:
        print("# paste-ready goldens (mean over the four IBSI phantom slices)")
        for name in sorted(got):
            print(f'\t{{"{name}", {got[name]!r}}},')
        print(f"\n# paste-ready {NEGDEPTH_TABLE} (bench_square12_graded, fixed_bin_number n=20)")
        print(f"static const ref_vals_map<double> {NEGDEPTH_TABLE}{{")
        items = list(negdepth.items())
        for i, (name, (v, col)) in enumerate(items):
            sep = "," if i < len(items) - 1 else ""
            print(f'\t\t{{ "{name}",\t{v!r} }}{sep}\t\t// {col}')
        print("};")

    if not os.path.exists(TEST_H):
        print(f"\n{os.path.basename(TEST_H)} does not exist yet - nothing to verify")
        return 0

    text = open(TEST_H, encoding="utf-8", errors="replace").read()
    SQUARE.check_square_in_header()
    print("\n# test_main_nyxus.h builds bench_square12_graded from the same formula")
    neg_ok, neg_fail, neg_miss = verify_negdepth(parse_pins(text, NEGDEPTH_TABLE), negdepth)

    pins = parse_pins(text, "ngldm_2d_mirp_ref_vals")
    print(f"\n# verifying {len(pins)} pinned goldens against this run")
    nok, nfail, nmiss = neg_ok, neg_fail, neg_miss
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
