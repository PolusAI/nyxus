"""OFFLINE MIRP oracle for the negative coarse grey depth: NGLDM and GLDZM, 2D and 3D, at 20 bins.

    python tests/vetting/oracles/gen_negative_greydepth_mirp.py           (print paste-ready pins)
    python tests/vetting/oracles/gen_negative_greydepth_mirp.py --check   (re-verify the pins)

A negative grey depth -N bins every ROI into N PyRadiomics-style bins over its own [min, max], which
is IBSI's fixed bin number, floor(N (x - min) / (max - min)) + 1, and exactly what MIRP's
`base_discretisation_method="fixed_bin_number"` computes. So MIRP discretises here itself: nothing
of Nyxus' binning is reproduced on the MIRP side, and the comparison covers the binning as well as
the matrix and the feature formulas. Four recipes, one per table:

- `ngldm3d.mirp_fbn20_negdepth`, `gldzm3d.mirp_fbn20_negdepth` -- the segmented phantom
  (tests/data/nifti/phantoms/ut_inten.nii + ut_mask57.nii, label 57), `by_slice=False`. The phantom
  is a float NIfTI and Nyxus reads it as integer grey levels shifted by the volume minimum; MIRP is
  handed those same integers (`nyxus_raw_levels()`), which is the loader's step and not the binning.
- `ngldm.mirp_fbn20_negdepth`, `gldzm.mirp_fbn20_negdepth` -- `bench_square12_graded`, a 12x12 ROI in
  a 14x14 frame at levels 100 + 3 ((7x + 11y + xy) mod 37), 36 of them over 100-208, `by_slice=True`.
  It is defined twice, here and in tests/test_main_nyxus.h, and `--check` compares the two.

The 2D square puts five of its 144 pixels exactly on a bin edge at 20 bins (n (x - min) a multiple of
the range), which is what separates the IBSI formula from one that divides by a rounded bin width.
The 3D phantom has no such pixel, so the square is the fixture that pins the edge.

Feature scope. 3NGLDM_DCP / NGLDM_DCP are left out as in the family generators (both tools return 1
whenever every voxel has a same-level neighbour), and 3NGLDM_GLM / 3NGLDM_DCM have no MIRP column.
2D GLDZM pins only its six features that do not read a zone's border distance -- GLNU, GLNUN, ZP,
GLV, LGLZE, HGLZE. The square fills its bounding box, so its border distance cannot tell the 2D
distance map from IBSI's, and pinning the distance features on it would claim more than it shows.

Provenance: tool=mirp (version printed at run time); env=nyxus_mirp or any env with mirp + numpy;
generator=tests/vetting/oracles/gen_negative_greydepth_mirp.py. Run offline; CI never invokes it.
"""
import argparse
import importlib.util
import logging
import os
import re
import sys
from importlib import metadata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
NBINS = 20
LABEL = 57
RELTOL = 1e-12   # the pins are printed at %.17g; this is a staleness check, not a vetting band


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NG3, GZ3, NG2, GZ2 = (_load(n) for n in
                      ("gen_ngldm3d_mirp", "gen_gldzm3d_mirp", "gen_ngldm_mirp", "gen_gldzm_mirp"))

GLDZM_2D_PINNED = ("GLDZM_GLNU", "GLDZM_GLNUN", "GLDZM_ZP", "GLDZM_GLV", "GLDZM_LGLZE", "GLDZM_HGLZE")

# table name -> (header it lives in, Nyxus feature -> MIRP column stem)
TABLES = {
    "ngldm_3d_mirp_negdepth_ref_vals": (
        "test_3d_ngldm_mirp.h", {k: v for k, v in NG3.MIRP.items() if k != "3NGLDM_DCP"}),
    "gldzm_3d_mirp_negdepth_ref_vals": ("test_3d_gldzm_mirp.h", dict(GZ3.MIRP)),
    "ngldm_2d_mirp_negdepth_ref_vals": (
        "test_2d_ngldm_mirp.h", {k: v for k, v in NG2.MIRP_COLUMN.items() if k != "NGLDM_DCP"}),
    "gldzm_2d_mirp_negdepth_ref_vals": (
        "test_2d_gldzm_mirp.h", {k: GZ2.MIRP_COLUMN[k] for k in GLDZM_2D_PINNED}),
}

SQUARE_SIDE, SQUARE_LO, SQUARE_HI = 14, 1, 12


def square12_graded():
    """-> (intensity, mask) of bench_square12_graded, as tests/test_main_nyxus.h builds it."""
    img = np.zeros((SQUARE_SIDE, SQUARE_SIDE), np.float64)
    msk = np.zeros((SQUARE_SIDE, SQUARE_SIDE), np.int32)
    for y in range(SQUARE_SIDE):
        for x in range(SQUARE_SIDE):
            if SQUARE_LO <= x <= SQUARE_HI and SQUARE_LO <= y <= SQUARE_HI:
                xx, yy = x - SQUARE_LO, y - SQUARE_LO
                img[y, x] = 100 + 3 * ((7 * xx + 11 * yy + xx * yy) % 37)
                msk[y, x] = 1
    return img, msk


def check_square_in_header():
    """The C++ fixture must be the array this generator feeds MIRP: compare the formula's text."""
    src = open(os.path.join(TESTS, "test_main_nyxus.h"), encoding="utf-8").read()
    m = re.search(r"make_2d_square12_graded_phantom.*?\{(.*?)\n    \}", src, re.S)
    if not m:
        raise RuntimeError("no make_2d_square12_graded_phantom in test_main_nyxus.h")
    body = re.sub(r"\s+", "", m.group(1))
    for frag in ("side=14", "lo=1", "hi=12", "100+3*((7*xx+11*yy+xx*yy)%37)"):
        if frag not in body:
            raise RuntimeError(f"test_main_nyxus.h's square no longer carries {frag!r}")


def run():
    import mirp
    logging.disable(logging.INFO)

    def features(image, mask, spacing, by_slice, family, mapping):
        res = mirp.extract_features(
            image=image, mask=mask, image_spacing=spacing, by_slice=by_slice,
            base_feature_families=family, base_discretisation_method="fixed_bin_number",
            base_discretisation_n_bins=NBINS)
        row = (res[0] if isinstance(res, list) else res).iloc[0]
        dim = "2d" if by_slice else "3d"
        out = {}
        for nyx, stem in mapping.items():
            pat = re.compile(re.escape(stem) + r"_(d1_a0\.0_)?" + dim + f"_fbn_n{NBINS}$")
            cols = [c for c in row.index if pat.match(c)]
            if len(cols) != 1:
                raise RuntimeError(f"{nyx}: expected one MIRP column for {stem}, found {cols}")
            out[nyx] = (float(row[cols[0]]), cols[0])
        return out

    inten, spacing = NG3.read_nifti(NG3.INTEN)
    mask_vol, _ = NG3.read_nifti(NG3.MASK)
    m3 = (mask_vol == LABEL).astype(np.int32)
    levels3 = NG3.nyxus_raw_levels(inten).astype(np.float64)

    img2, m2 = square12_graded()
    img2, m2 = img2[np.newaxis], m2[np.newaxis]
    one = (1.0, 1.0, 1.0)

    got = {}
    for table, (_, mapping) in TABLES.items():
        if table.startswith("ngldm_3d"):
            got[table] = features(levels3, m3, spacing, False, "ngldm", mapping)
        elif table.startswith("gldzm_3d"):
            got[table] = features(levels3, m3, spacing, False, "gldzm", mapping)
        elif table.startswith("ngldm_2d"):
            got[table] = features(img2, m2, one, True, "ngldm", mapping)
        else:
            got[table] = features(img2, m2, one, True, "gldzm", mapping)
    return got


def parse_pins(header, table):
    src = open(os.path.join(TESTS, header), encoding="utf-8", errors="replace").read()
    m = re.search(r"ref_vals_map<double>\s+" + re.escape(table) + r"\s*\{(.*?)\n\}", src, re.S)
    if not m:
        raise RuntimeError(f"no {table} initialiser in {header}")
    return {n: float(v) for n, v in re.findall(r'\{\s*"([0-9A-Z_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', m.group(1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    print(f"# mirp {metadata.version('mirp')}, numpy {np.__version__}, fixed_bin_number n={NBINS}")
    got = run()

    if not args.check:
        for table, vals in got.items():
            print(f"\nstatic const ref_vals_map<double> {table}{{")
            items = list(vals.items())
            for i, (name, (v, col)) in enumerate(items):
                sep = "," if i < len(items) - 1 else ""
                print(f'\t\t{{ "{name}",\t{v!r} }}{sep}\t\t// {col}')
            print("};")
        return 0

    check_square_in_header()
    problems = []
    for table, (header, mapping) in TABLES.items():
        pins = parse_pins(header, table)
        if set(pins) != set(mapping):
            problems.append(f"{table}: pinned {sorted(set(pins) ^ set(mapping))} differ from the scope")
        for name, (v, _) in got[table].items():
            if name in pins and abs(pins[name] - v) > RELTOL * max(abs(v), 1e-300):
                problems.append(f"{table}: {name} pinned {pins[name]!r}, mirp {v!r}")
    for p in problems:
        print("MISMATCH:", p)
    print("ALL CHECKS PASSED" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
