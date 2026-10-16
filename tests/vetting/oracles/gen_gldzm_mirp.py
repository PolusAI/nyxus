"""OFFLINE mirp oracle for the 2D GLDZM features, on the IBSI digital phantom and bench_square12_graded.

    python tests/vetting/oracles/gen_gldzm_mirp.py           (from the repository root)
    python tests/vetting/oracles/gen_gldzm_mirp.py --check   (verify only, no paste-ready tables)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_2d_gldzm_mirp.h -- the
per-slice table, the four-slice means and the negative-grey-depth table -- exiting non-zero on any
mismatch or on any pin this generator cannot produce.

Recipe `gldzm.ibsi_phantom_2d`: the four IBSI digital-phantom slices, each featurised on its own.
Each slice is its own ROI and yields its own scalar; their mean is what the IBSI "2D, averaged"
aggregation publishes. Both quantities are pinned, because a mean is weaker than the four values
behind it -- errors in two slices that cancel leave it unmoved, and on this family the disagreement
that mattered lived in one slice. mirp runs `by_slice=True` with `base_discretisation_method="none"`
(the phantom is already discrete 1..6), which is what Nyxus computes in IBSI mode.

mirp and Nyxus build the same 8-connected zones over the same distance-to-border metric, so nothing
but float summation order separates them: worst absolute residual 1.3e-15, worst relative 7.0e-16,
over the 80 comparisons. That is why the header pins at SPEC 7's "exact" tier -- an ABSOLUTE 1e-9
band -- rather than at a cross-tool relative one.

Recipe `gldzm.mirp_fbn20_negdepth`, table gldzm_2d_mirp_negdepth_ref_vals: the negative coarse grey
depth. GREYDEPTH=-20 bins the ROI into 20 PyRadiomics-style bins over its own [min, max], which is
IBSI's fixed bin number, floor(20 (x - min) / (max - min)) + 1, and exactly what mirp's
`base_discretisation_method="fixed_bin_number"`, n_bins=20, computes. So mirp discretises here
itself: nothing of Nyxus' binning is reproduced on the mirp side, and the comparison covers the
binning as well as the zones and the feature formulas. The fixture is bench_square12_graded, built
by square12_graded.py (shared with gen_ngldm_mirp.py), `by_slice=True`. 14 of its 144 pixels sit
exactly on a bin edge at 20 bins, and the 5 at 181 are the ones a division by the bin width sends
one bin low; square12_graded.py has the arithmetic. Every run also checks that test_main_nyxus.h
still builds the square from the same formula. The pins are printed at full precision, so they are
held at rel 1e-12 -- a staleness check, not a vetting band.

That table pins only the six features that read no zone's border distance -- GLNU, GLNUN, ZP, GLV,
LGLZE, HGLZE. The square fills its bounding box, so its border distance cannot tell the 2D distance
map from IBSI's, and pinning the distance features on it would claim more than it shows.

NOT covered here: GLDZM_GLM and GLDZM_ZDM. mirp exposes no grey-level-mean or zone-distance-mean
column -- neither is an IBSI GLDZM feature -- so those two stay regression rows in
test_2d_gldzm_regression.h.

Provenance: tool=mirp (version printed by this script); numpy; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_gldzm_mirp.py, with tests/vetting/oracles/square12_graded.py.
Run offline; CI never invokes it.
"""
import argparse
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

RELTOL = 1e-9          # this script's own golden-vs-mirp check, not the test tier; current run: rel = 0
NEGDEPTH_RELTOL = 1e-12   # the negdepth pins are printed at %.17g; a staleness check, not a band
NEGDEPTH_TABLE = "gldzm_2d_mirp_negdepth_ref_vals"

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
NEGDEPTH_SCOPE = {k: MIRP_COLUMN[k] for k in
                  ("GLDZM_GLNU", "GLDZM_GLNUN", "GLDZM_ZP", "GLDZM_GLV", "GLDZM_LGLZE", "GLDZM_HGLZE")}


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


def verify_negdepth(pins, got):
    """-> (verified, failed, unproducible) for the negdepth table: scope and values both."""
    print("")
    print(f"# verifying {len(pins)} goldens pinned in {NEGDEPTH_TABLE} against this run, "
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

    slices = run()
    means = {k: float(np.mean(v)) for k, v in slices.items()}
    by_slice = {f"{k}_z{z + 1}": v for k, vs in slices.items() for z, v in enumerate(vs)}
    negdepth = SQUARE.mirp_fbn("gldzm", NEGDEPTH_SCOPE)

    print(f"# mirp {version}, numpy {np.__version__}, "
          f"recipes gldzm.ibsi_phantom_2d and gldzm.mirp_fbn20_negdepth")
    if not args.check:
        print("# paste-ready goldens (mean over the four IBSI phantom slices)")
        for name in sorted(means):
            print(f'\t{{"{name}", {means[name]!r}}},')
        print("# paste-ready per-slice goldens")
        for name in sorted(by_slice):
            print(f'\t{{"{name}", {by_slice[name]!r}}},')
        print("")
        print(f"# paste-ready {NEGDEPTH_TABLE} (bench_square12_graded, fixed_bin_number n=20)")
        print(f"static const ref_vals_map<double> {NEGDEPTH_TABLE}{{")
        items = list(negdepth.items())
        for i, (name, (v, col)) in enumerate(items):
            sep = "," if i < len(items) - 1 else ""
            print(f'\t\t{{ "{name}",\t{v!r} }}{sep}\t\t// {col}')
        print("};")

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

    SQUARE.check_square_in_header()
    print("")
    print("# test_main_nyxus.h builds bench_square12_graded from the same formula")
    a, b, c = verify_negdepth(parse_pins(text, NEGDEPTH_TABLE), negdepth)
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
