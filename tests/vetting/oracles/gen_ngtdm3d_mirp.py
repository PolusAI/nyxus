"""OFFLINE MIRP oracle for the 3D NGTDM family, on ROIs that mix voxels with and without a neighbour.

    python tests/vetting/oracles/gen_ngtdm3d_mirp.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in ../../test_3d_ngtdm_mirp.h and
../../python/test_3d_ngtdm_mirp.py, exiting non-zero on any mismatch, on any pin it cannot produce,
and on any value it produces that nothing pins.

WHY MIRP AND NOT PYRADIOMICS. A voxel with no ROI voxel within the neighbourhood radius has no
neighbourhood mean. IBSI leaves such a voxel out of the NGTDM: it is in no row and counts towards
neither n_i, N_v nor N_g,p. MIRP follows IBSI (it drops the voxel, and `n_p` is the number of levels
left with a voxel), and so does Nyxus. PyRadiomics keeps the voxel as a row with s_i = 0, which
moves all five features of any ROI holding one. So a ROI that mixes the two kinds of voxel has to be
held to MIRP; gen_ngtdm3d_pyradiomics.py keeps its fixtures free of neighbourless voxels for the same
reason. On a ROI with no neighboured voxel at all MIRP returns NaN, which is what Nyxus' soft-NaN
value stands for; mirp_empty_check() measures that on label 58 of the ball phantom.

Recipe `ngtdm3d.mirp_fbn_mixed`: label 59 of the NGTDM ball phantom
(tests/data/nifti/compat_int/compat_int_ngtdm_3d_ball.nii + compat_seg/compat_seg_ngtdm_3d_ball.nii)
-- a chain of three voxels at levels 3, 0, 3 and a lone voxel at level 2 that nothing of the ROI
reaches. On the Nyxus side GREYDEPTH=100, IBSI=false, NGTDM_GREYDEPTH=0 (no binning),
NGTDM_RADIUS=1. MIRP: by_slice=False, distance 1, native 1x1x1 spacing,
base_discretisation_method="fixed_bin_number" with n = (ROI max - ROI min + 1) bins.

THE BIN COUNT REPRODUCES THE RAW LEVELS. Over integer levels min..max, fixed_bin_number with
n = max - min + 1 bins maps x to floor(n (x - min) / (max - min)) + 1, clipped to n, which is
x - min + 1 for every x in range -- the same levels Nyxus' zero-min correction produces when the ROI's
lowest level is 0. fbn_identity_check() asserts that over each ROI rather than leaving it argued, and
the generator also runs MIRP with the discretisation switched off on those levels and requires the
two runs to agree.

The out-of-core pins (`NGTDM_3D_MIXED_MIRP` in tests/python/test_3d_ngtdm_mirp.py) are the same recipe
on make_ooc_mixed() below: the out-of-core ellipsoid of gen_ngtdm3d_pyradiomics.py plus one voxel at a
corner outside its reach, at level 12, which the ellipsoid (levels 0..11) does not carry.

TWO REFERENCES. reference_ngtdm() is a plain-numpy NGTDM from the IBSI definition, with no MIRP in its
path, and every pinned value is produced by both. The label-59 matrix pinned in the C++ header is the
numpy one, in Nyxus' layout (one row per level of the ROI, the lone voxel's row empty), and the five
MIRP features are recomputed from that pinned table as a cross-check.

NIFTI READING WITHOUT A NIFTI LIBRARY: the mirp env has neither SimpleITK nor nibabel, so the header
is parsed directly, as gen_ngldm3d_mirp.py does.

Provenance of the run behind the pins -- the printed header names whatever mirp is actually
installed, so a run under another version says so rather than repeating this line:
tool=mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3, Python 3.11); env=nyxus_mirp (conda-forge:
`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`);
generator=tests/vetting/oracles/gen_ngtdm3d_mirp.py. Run offline; CI never invokes it.
"""
import logging
import math
import os
import re
import sys
from importlib import metadata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
BALL_INTEN = os.path.join(DATA, "compat_int", "compat_int_ngtdm_3d_ball.nii")
BALL_MASK = os.path.join(DATA, "compat_seg", "compat_seg_ngtdm_3d_ball.nii")
MIRP_H = os.path.join(TESTS, "test_3d_ngtdm_mirp.h")
MIRP_PY = os.path.join(TESTS, "python", "test_3d_ngtdm_mirp.py")

EMPTY_LABEL = 58        # no voxel has a neighbour
MIXED_LABEL = 59        # some voxels have one, one does not
RELTOL = 1e-12          # both references are exact up to float summation order

SCALAR_TABLE = "ngtdm_3d_mirp_mixed_ref_vals"
MATRIX_TABLE = "ngtdm_3d_mirp_mixed_matrix_ref_vals"
OOC_PINS = "NGTDM_3D_MIXED_MIRP"

# Nyxus feature -> MIRP NGTDM column stem. MIRP suffixes every column with the discretisation
# (`_3d_fbn_n4`); the stem is matched and the suffix asserted, so a changed bin count cannot read a
# column of another config.
MIRP = {
    "3NGTDM_BUSYNESS": "ngt_busyness",
    "3NGTDM_COARSENESS": "ngt_coarseness",
    "3NGTDM_COMPLEXITY": "ngt_complexity",
    "3NGTDM_CONTRAST": "ngt_contrast",
    "3NGTDM_STRENGTH": "ngt_strength",
}

NIFTI_DTYPE = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
               512: np.uint16, 768: np.uint32}


def read_nifti(path):
    """-> array shaped (z, y, x). Uncompressed single-file NIfTI-1 only."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if int(np.frombuffer(raw, np.int32, 1, 0)[0]) != 348 or raw[344:347] != b"n+1":
        raise RuntimeError(f"{path} is not an uncompressed single-file NIfTI-1")
    dim = np.frombuffer(raw, np.int16, 8, 40)
    datatype = int(np.frombuffer(raw, np.int16, 1, 70)[0])
    vox_offset = int(np.frombuffer(raw, np.float32, 1, 108)[0])
    if datatype not in NIFTI_DTYPE:
        raise RuntimeError(f"{path}: unsupported NIfTI datatype {datatype}")
    nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
    return np.frombuffer(raw, NIFTI_DTYPE[datatype], nx * ny * nz, vox_offset).reshape((nz, ny, nx))


def make_ooc_mixed():
    """-> (intensity uint16, mask bool), (z, y, x) = 24x60x60: the out-of-core mixed volume.

    gen_ngtdm3d_pyradiomics.py's make_ooc_ellipsoid() -- every voxel of it has a neighbour -- plus the
    corner voxel (0,0,0) at level 12. The ellipsoid's nearest voxel is well beyond radius 1 of the
    corner, and its levels are 0..11, so level 12 is the corner's alone. Must stay the volume
    tests/python/test_3d_ngtdm_mirp.py::_make_mixed_volume_pair builds.
    """
    shape = (24, 60, 60)
    z, y, x = np.indices(shape)
    mask = ((z - 11.5) / 11.5) ** 2 + ((y - 29.5) / 28.0) ** 2 + ((x - 29.5) / 28.0) ** 2 <= 1.0
    inten = ((3 * x + 5 * y + 7 * z + x * y * z) % 12).astype(np.uint16)
    inten[0, 0, 0] = 12
    mask[0, 0, 0] = True
    return inten, mask


def neighbour_sums(levels, mask, delta=1):
    """-> (sum of in-ROI neighbour levels, number of in-ROI neighbours) per voxel, Chebyshev `delta`."""
    lv = np.where(mask, levels, 0).astype(np.float64)
    m = mask.astype(np.int64)
    pad = [(delta, delta)] * 3
    lp, mp = np.pad(lv, pad), np.pad(m, pad)
    nz, ny, nx = levels.shape
    tot = np.zeros(levels.shape)
    cnt = np.zeros(levels.shape, dtype=np.int64)
    for dz in range(-delta, delta + 1):
        for dy in range(-delta, delta + 1):
            for dx in range(-delta, delta + 1):
                if dz == dy == dx == 0:
                    continue
                sl = (slice(delta + dz, delta + dz + nz), slice(delta + dy, delta + dy + ny),
                      slice(delta + dx, delta + dx + nx))
                tot += lp[sl]
                cnt += mp[sl]
    return tot, cnt


def reference_ngtdm(levels, mask, delta=1):
    """-> [(level, n_i, p_i, s_i)] with one row per level the ROI carries, in Nyxus' layout: a level
    whose every voxel lacks a neighbour keeps its row, with n_i = 0. No MIRP in this path."""
    tot, cnt = neighbour_sums(levels, mask, delta)
    valid = mask & (cnt > 0)
    nvp = int(valid.sum())
    rows = []
    for i in sorted(set(levels[mask].tolist())):
        sel = valid & (levels == i)
        n = int(sel.sum())
        s = float(np.sum(np.abs(i - tot[sel] / cnt[sel]))) if n else 0.0
        rows.append((int(i), n, n / nvp if nvp else 0.0, s))
    return rows


def reference_features(rows):
    """-> {nyxus feature: value} from an NGTDM table by the IBSI formulas, over its non-empty rows."""
    rows = [r for r in rows if r[1] > 0]
    i = np.array([r[0] for r in rows], dtype=float)
    p = np.array([r[2] for r in rows], dtype=float)
    s = np.array([r[3] for r in rows], dtype=float)
    nvp = float(sum(r[1] for r in rows))
    ngp = len(rows)
    d2 = (i[:, None] - i[None, :]) ** 2
    ps = p * s
    return {
        "3NGTDM_COARSENESS": 1.0 / float(np.sum(ps)),
        "3NGTDM_CONTRAST": (float(np.sum(p[:, None] * p[None, :] * d2)) / (ngp * (ngp - 1))
                            * float(np.sum(s)) / nvp),
        "3NGTDM_BUSYNESS": (float(np.sum(ps))
                            / float(np.sum(np.abs(i[:, None] * p[:, None] - i[None, :] * p[None, :])))),
        "3NGTDM_COMPLEXITY": float(np.sum(np.abs(i[:, None] - i[None, :]) * (ps[:, None] + ps[None, :])
                                          / (p[:, None] + p[None, :]))) / nvp,
        "3NGTDM_STRENGTH": float(np.sum((p[:, None] + p[None, :]) * d2)) / float(np.sum(s)),
    }


def lone_count(mask, delta=1):
    _, cnt = neighbour_sums(mask.astype(np.int64), mask, delta)
    return int((mask & (cnt == 0)).sum())


def fbn_bins(inten, mask):
    """-> the bin count that makes fixed_bin_number reproduce the ROI's raw levels, min..max -> 1..n."""
    v = inten[mask].astype(np.int64)
    return int(v.max() - v.min() + 1)


def fbn_identity_check(name, inten, mask):
    """MIRP's fixed_bin_number formula at fbn_bins() is x - min + 1 on every ROI level. -> failures."""
    v = inten[mask].astype(np.int64)
    lo, hi, n = int(v.min()), int(v.max()), fbn_bins(inten, mask)
    binned = np.minimum(np.floor(n * (v - lo) / (hi - lo)) + 1, n).astype(np.int64)
    if not np.array_equal(binned, v - lo + 1):
        print(f"  FAIL {name}: fixed_bin_number n={n} does not reproduce the raw levels")
        return 1
    print(f"  OK   {name}: fixed_bin_number n={n} maps the ROI's levels {lo}..{hi} to {1}..{n}")
    return 0


def mirp_features(image, mask, **discretisation):
    """-> ({nyxus feature: value}, the column suffix MIRP used)."""
    import mirp
    res = mirp.extract_features(
        image=image.astype(np.float64), mask=mask.astype(np.int32), image_spacing=(1.0, 1.0, 1.0),
        by_slice=False, base_feature_families="ngtdm", **discretisation)
    df = res[0] if isinstance(res, list) else res
    row = df.iloc[0]
    out, suffixes = {}, set()
    for nyx, stem in MIRP.items():
        cols = [c for c in df.columns if c.startswith(stem + "_")]
        if len(cols) != 1:
            raise RuntimeError(f"MIRP produced {cols} for {nyx}")
        out[nyx] = float(row[cols[0]])
        suffixes.add(cols[0][len(stem):])
    if len(suffixes) != 1:
        raise RuntimeError(f"MIRP columns carry different suffixes: {suffixes}")
    return out, suffixes.pop()


def oracle_run(name, inten, mask):
    """MIRP twice (fbn, and no discretisation over the lifted levels) and the numpy reference, all of
    which must agree. -> ({feature: value}, [(level, n_i, p_i, s_i)], failure count)."""
    bad = fbn_identity_check(name, inten, mask)
    n = fbn_bins(inten, mask)
    fbn, suffix = mirp_features(inten, mask, base_discretisation_method="fixed_bin_number",
                                base_discretisation_n_bins=n)
    if suffix != f"_3d_fbn_n{n}":
        print(f"  FAIL {name}: MIRP column suffix {suffix!r}, expected _3d_fbn_n{n}")
        bad += 1
    lifted = (inten.astype(np.int64) - int(inten[mask].min()) + 1)
    none, _ = mirp_features(np.where(mask, lifted, 0), mask, base_discretisation_method="none")
    rows = reference_ngtdm(lifted, mask)
    ref = reference_features(rows)
    worst_none = max(rel(fbn[k], none[k]) for k in MIRP)
    worst_ref = max(rel(fbn[k], ref[k]) for k in MIRP)
    print(f"# {name}: {int(mask.sum())} voxels, {lone_count(mask)} without a neighbour; "
          f"levels {sorted(set(lifted[mask].tolist()))}")
    print(f"  mirp fbn n={n} vs mirp without discretisation: worst rel {worst_none:.3g}")
    print(f"  mirp fbn n={n} vs the numpy reference:          worst rel {worst_ref:.3g}")
    if worst_none > RELTOL or worst_ref > RELTOL:
        print(f"  FAIL {name}: the references disagree")
        bad += 1
    return fbn, rows, bad


def mirp_empty_check(inten, labels):
    """MIRP reports NaN for every feature of label 58, none of whose voxels has a neighbour. -> failures."""
    mask = labels == EMPTY_LABEL
    got, _ = mirp_features(inten, mask, base_discretisation_method="fixed_bin_number",
                           base_discretisation_n_bins=fbn_bins(inten, mask))
    print(f"\n# label {EMPTY_LABEL} ({int(mask.sum())} voxels, {lone_count(mask)} without a neighbour): "
          f"mirp {got}")
    if lone_count(mask) != int(mask.sum()) or not all(math.isnan(v) for v in got.values()):
        print(f"  FAIL label {EMPTY_LABEL}: MIRP no longer reports NaN on a ROI with no neighboured voxel")
        return 1
    return 0


def rel(have, want):
    return abs(have - want) / max(abs(want), 1e-300)


def parse_scalar_pins(txt, table):
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found")
    body = re.sub(r"//[^\n]*", "", txt[m.end():].split("};", 1)[0])
    return {n: float(v) for n, v in re.findall(r'\{\s*"(3NGTDM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def parse_matrix_pins(txt, table):
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found")
    body = re.sub(r"//[^\n]*", "", txt[m.end():].split("};", 1)[0])
    out = []
    for row in re.finditer(r"\{([^{}]*)\}", body):
        parts = [p.strip() for p in row.group(1).split(",") if p.strip()]
        if len(parts) != 4:
            raise RuntimeError(f"{table}: row {row.group(1)!r} has {len(parts)} fields, expected 4")
        out.append((int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])))
    return out


def parse_py_pins(txt, name):
    m = re.search(re.escape(name) + r"\s*=\s*\{", txt)
    if not m:
        raise RuntimeError(f"{name} not found in {MIRP_PY}")
    body = re.sub(r"#[^\n]*", "", txt[m.end():].split("}", 1)[0])
    return {n: float(v) for n, v in re.findall(r'"(3NGTDM_[A-Z0-9_]+)"\s*:\s*([-0-9.eE+]+)', body)}


def verify_scalars(what, pins, got):
    bad = 0
    print(f"\n# verifying the {len(pins)} pins of {what} against this run, rel<={RELTOL:g}")
    if sorted(pins) != sorted(got):
        print(f"  FAIL {what} pins {sorted(pins)}, MIRP produces {sorted(got)}")
        bad += 1
    for name in sorted(set(pins) & set(got)):
        r = rel(pins[name], got[name])
        print(f"  {'OK  ' if r <= RELTOL else 'FAIL'} {name}: pin={pins[name]!r} mirp={got[name]!r} rel={r:.3g}")
        bad += r > RELTOL
    return bad


def main():
    for p in (BALL_INTEN, BALL_MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1
    logging.disable(logging.INFO)
    try:
        version = metadata.version("mirp")
    except metadata.PackageNotFoundError:
        version = "unknown"
    print(f"# mirp {version}, numpy {np.__version__}, by_slice=False, distance=1")

    inten = read_nifti(BALL_INTEN).astype(np.int64)
    labels = read_nifti(BALL_MASK)
    mixed = labels == MIXED_LABEL

    feats, rows, bad = oracle_run(f"ball label {MIXED_LABEL}", inten, mixed)
    ooc_inten, ooc_mask = make_ooc_mixed()
    ooc_feats, _, b = oracle_run("out-of-core mixed volume", ooc_inten.astype(np.int64), ooc_mask)
    bad += b
    bad += mirp_empty_check(inten, labels)

    print(f"\n# paste-ready goldens: {SCALAR_TABLE}")
    for name in sorted(feats):
        print(f'\t{{"{name}", {feats[name]!r}}},'.ljust(56) + f"// {MIRP[name]}_3d_fbn_n{fbn_bins(inten, mixed)}")
    print(f"\n# paste-ready goldens: {MATRIX_TABLE}   {{ i, n_i, p_i, s_i }}")
    for lev, n, p, s in rows:
        print(f"\t{{ {lev}, {n}, {p!r}, {s!r} }},")
    print(f"\n# paste-ready goldens: {OOC_PINS} ({os.path.relpath(MIRP_PY, TESTS)})")
    for name in sorted(ooc_feats):
        print(f'    "{name}": {ooc_feats[name]!r},')

    txt_h = open(MIRP_H, encoding="utf-8", errors="replace").read()
    bad += verify_scalars(SCALAR_TABLE, parse_scalar_pins(txt_h, SCALAR_TABLE), feats)

    pinned = parse_matrix_pins(txt_h, MATRIX_TABLE)
    print(f"\n# verifying the {len(pinned)} rows of {MATRIX_TABLE} against the numpy reference")
    if [r[:2] for r in pinned] != [r[:2] for r in rows]:
        print(f"  FAIL levels/counts: pinned {[r[:2] for r in pinned]}, reference {[r[:2] for r in rows]}")
        bad += 1
    else:
        for (lev, _, pp, ps), (_, _, p, s) in zip(pinned, rows):
            for what, have, want in (("p_i", p, pp), ("s_i", s, ps)):
                if abs(have - want) > RELTOL * max(abs(want), 1.0):
                    print(f"  FAIL i={lev} {what}: pinned {want!r}, reference {have!r}")
                    bad += 1
    # the pinned table determines all five features; recomputing them is what ties it to MIRP
    derived = reference_features(pinned)
    print(f"# the five MIRP values recomputed from the pinned {MATRIX_TABLE}: "
          f"worst rel {max(rel(derived[k], feats[k]) for k in MIRP):.3g}")
    for k in MIRP:
        if rel(derived[k], feats[k]) > RELTOL:
            print(f"  FAIL {k}: from the pinned matrix {derived[k]!r}, MIRP {feats[k]!r}")
            bad += 1
    if not any(r[1] == 0 for r in pinned):
        print(f"  FAIL {MATRIX_TABLE} has no empty row, so it cannot show Ngp != the number of levels")
        bad += 1

    txt_py = open(MIRP_PY, encoding="utf-8", errors="replace").read()
    bad += verify_scalars(OOC_PINS, parse_py_pins(txt_py, OOC_PINS), ooc_feats)

    print("\nALL CHECKS PASSED" if not bad else f"\n{bad} MISMATCH(ES)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
