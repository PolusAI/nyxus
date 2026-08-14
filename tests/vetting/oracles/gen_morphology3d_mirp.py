"""OFFLINE MIRP oracle for the 3D morphology PCA axis features, on the segmented phantom.

    python tests/vetting/oracles/gen_morphology3d_mirp.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_morphology_mirp.h,
exiting non-zero on any mismatch, on any pin it cannot produce, and on any structural identity the
family carries by construction that the oracle's own output violates.

Recipe `morphology3d.mirp_ibsi`: the segmented phantom (tests/data/nifti/phantoms/ut_inten.nii +
ut_mask57.nii, label 57) at its native 1x1x1 spacing, `by_slice=False` (a true 3D ROI, not a stack of
2D slices) and `base_discretisation_method="none"` -- morphology is computed from the mask geometry,
so no grey-level binning applies and asking for one would be meaningless. On the Nyxus side that is
D3_SurfaceFeature with IBSI=true and PIXELSIZEUM=100, which is what test_3d_morphology_common.h sets.

MIRP's `morph_*` block is the IBSI section 3.1 morphology family. Only the five PCA axis quantities are
pinned here. The rest of the block is deliberately not: MIRP's `morph_area_mesh` is a marching-cubes
mesh area while Nyxus' 3AREA counts exposed voxel faces (46739 against 59992, ~28%), and the five
features derived from area -- 3AREA_2_VOLUME, 3COMPACTNESS1/2, 3SPHERICITY,
3SPHERICAL_DISPROPORTION -- inherit that convention difference. They stay regression-only until the
convention question is settled; see ../audit/morphology_3d_mirp_vetting_report.md.

NIFTI READING WITHOUT A NIFTI LIBRARY: the mirp env has neither SimpleITK nor nibabel, and adding one
would make this a two-env, two-step generator. The phantoms are uncompressed single-file NIfTI-1
(magic "n+1"), so the header is parsed directly below -- 20 lines of numpy against a frozen on-disk
format, and the generator stays runnable in one env.

Provenance: tool=mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3, Python 3.11); env=nyxus_mirp (conda-forge:
`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`);
generator=tests/vetting/oracles/gen_morphology3d_mirp.py. Run offline; CI never invokes it.
"""
import logging
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
PHANTOMS = os.path.join(TESTS, "data", "nifti", "phantoms")
INTEN = os.path.join(PHANTOMS, "ut_inten.nii")
MASK = os.path.join(PHANTOMS, "ut_mask57.nii")
TEST_H = os.path.join(TESTS, "test_3d_morphology_mirp.h")

RELTOL = 1e-9          # matches the C++ assertions (agrees_gt frac_tolerance = 1e9)
LABEL = 57

# Nyxus feature -> MIRP feature. MIRP names the PCA axes by role; Nyxus names them by size rank, and
# the two agree only because MAJOR is the largest eigenvalue. That correspondence is exactly what the
# axis-mislabel defect broke, so the identities below re-check it every run.
MIRP = {
    "3MAJOR_AXIS_LEN": "morph_pca_maj_axis",
    "3MINOR_AXIS_LEN": "morph_pca_min_axis",
    "3LEAST_AXIS_LEN": "morph_pca_least_axis",
    "3ELONGATION": "morph_pca_elongation",
    "3FLATNESS": "morph_pca_flatness",
}

# Reported but NOT pinned: the MIRP quantities that correspond to the family's other features. They
# are a second opinion on the three rows whose registry oracle is `matlab` -- MATLAB itself cannot be
# re-run here (no licence, and Octave's image package has no regionprops3) -- and they are the
# measurement behind the area-convention gap that keeps five features regression-only.
CROSSCHECK = [
    ("morph_vol_approx", "3VOXEL_VOLUME, and MATLAB regionprops3 Volume"),
    ("morph_volume", "mesh volume; Nyxus 3MESH_VOLUME is aliased to the convex-hull volume instead"),
    ("morph_vol_dens_conv_hull", "volume / convex-hull volume -> back out the hull volume"),
    ("morph_area_mesh", "3AREA, but marching-cubes mesh area vs Nyxus' exposed-voxel-face count"),
    ("morph_sphericity", "3SPHERICITY (inherits the area convention)"),
    ("morph_av", "3AREA_2_VOLUME (inherits the area convention)"),
]

# NIfTI-1 datatype code -> numpy dtype, for the codes these phantoms use.
NIFTI_DTYPE = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}


def read_nifti(path):
    """-> (array shaped (z,y,x), spacing (z,y,x)). Uncompressed single-file NIfTI-1 only."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if int(np.frombuffer(raw, np.int32, 1, 0)[0]) != 348 or raw[344:347] != b"n+1":
        raise RuntimeError(f"{path} is not an uncompressed single-file NIfTI-1")
    dim = np.frombuffer(raw, np.int16, 8, 40)
    datatype = int(np.frombuffer(raw, np.int16, 1, 70)[0])
    pixdim = np.frombuffer(raw, np.float32, 8, 76)
    vox_offset = int(np.frombuffer(raw, np.float32, 1, 108)[0])
    if datatype not in NIFTI_DTYPE:
        raise RuntimeError(f"{path}: unsupported NIfTI datatype {datatype}")
    nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
    vol = np.frombuffer(raw, NIFTI_DTYPE[datatype], nx * ny * nz, vox_offset)
    # NIfTI stores x fastest; MIRP wants (z, y, x).
    vol = vol.reshape((nz, ny, nx))
    spacing_zyx = (float(pixdim[3]), float(pixdim[2]), float(pixdim[1]))
    return vol, spacing_zyx


def parse_pins(txt, table):
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found")
    body = txt[m.end():].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"(3[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def run():
    import mirp
    # MIRP configures its own logger during the run, so a level on the root logger does not stop it
    # interleaving INFO lines with the golden table.
    logging.disable(logging.INFO)

    inten, spacing = read_nifti(INTEN)
    mask_vol, mask_spacing = read_nifti(MASK)
    if spacing != mask_spacing:
        raise RuntimeError(f"spacing mismatch: intensity {spacing} vs mask {mask_spacing}")
    mask = (mask_vol == LABEL).astype(np.int32)
    if not mask.any():
        raise RuntimeError(f"label {LABEL} is empty in {MASK}")
    print(f"# volume {inten.shape}, roi voxels {int(mask.sum())}, spacing zyx {spacing}")

    res = mirp.extract_features(
        image=inten.astype(np.float64), mask=mask,
        image_spacing=spacing,
        by_slice=False,
        base_feature_families="morphology",
        base_discretisation_method="none",
    )
    df = res[0] if isinstance(res, list) else res
    row = df.iloc[0]
    out = {}
    for nyx, mrp in MIRP.items():
        if mrp not in df.columns:
            raise RuntimeError(f"MIRP produced no {mrp} (for {nyx})")
        out[nyx] = float(row[mrp])
    extra = {c: float(row[c]) for c, _ in CROSSCHECK if c in df.columns}
    return out, extra


def check_identities(got):
    """The structural relations the axis features carry by construction.

    These are not decoration: the defect this family carried assigned the eigenvalues in the wrong
    order, which produced LEAST > MAJOR and FLATNESS > 1 -- both impossible, and both caught here.
    """
    maj, mino, least = got["3MAJOR_AXIS_LEN"], got["3MINOR_AXIS_LEN"], got["3LEAST_AXIS_LEN"]
    elong, flat = got["3ELONGATION"], got["3FLATNESS"]
    problems = []
    if not (maj >= mino >= least > 0):
        problems.append(f"axis order broken: MAJOR={maj!r} MINOR={mino!r} LEAST={least!r}")
    for name, v in (("3ELONGATION", elong), ("3FLATNESS", flat)):
        if not 0.0 <= v <= 1.0:
            problems.append(f"{name} = {v!r} is outside [0,1]")
    for name, v, ratio in (("3ELONGATION", elong, mino / maj), ("3FLATNESS", flat, least / maj)):
        if abs(v - ratio) > 1e-9 * max(abs(ratio), 1e-12):
            problems.append(f"{name} = {v!r} != its defining ratio {ratio!r}")
    for p in problems:
        print(f"  IDENTITY VIOLATED {p}")
    print(f"# {'all' if not problems else 'NOT all'} structural identities hold "
          f"(MAJOR>=MINOR>=LEAST>0, ELONGATION/FLATNESS in [0,1] and equal to their ratios)")
    return len(problems)


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1

    got, extra = run()

    import mirp
    print(f"# mirp {getattr(mirp, '__version__', None) or __import__('importlib.metadata', fromlist=['version']).version('mirp')}, "
          f"numpy {np.__version__}, label={LABEL}, by_slice=False, discretisation=none")
    print("# paste-ready goldens")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// {MIRP[name]}")

    print()
    nbad = check_identities(got)

    print("\n# cross-check only, nothing below is pinned")
    for key, why in CROSSCHECK:
        if key in extra:
            print(f"  {key} = {extra[key]!r}    # {why}")
    if "morph_volume" in extra and extra.get("morph_vol_dens_conv_hull"):
        hull = extra["morph_volume"] / extra["morph_vol_dens_conv_hull"]
        print(f"  (derived) convex-hull volume = {hull!r}")

    if not os.path.exists(TEST_H):
        print(f"\n# {os.path.basename(TEST_H)} does not exist yet -- nothing to verify")
        return 1 if nbad else 0

    pins = parse_pins(open(TEST_H, encoding="utf-8", errors="replace").read(),
                      "morphology_3d_mirp_ref_vals")
    print(f"\n# verifying {len(pins)} pinned goldens against this run")
    nok = nfail = nmiss = 0
    for name in sorted(pins):
        want = pins[name]
        if name not in got:
            print(f"  MISSING {name}: pinned {want!r} but MIRP reports no counterpart")
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

    unpinned = sorted(set(got) - set(pins))
    for name in unpinned:
        print(f"  UNPINNED {name}: MIRP reports {got[name]!r} and nothing pins it")

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible, {len(unpinned)} unpinned")
    if nfail or nmiss or nbad or unpinned:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
