"""OFFLINE MIRP oracle for the 3D NGLDM features, on the segmented phantom.

    python tests/vetting/oracles/gen_ngldm3d_mirp.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_ngldm_mirp.h, exiting
non-zero on any mismatch or on any pin it cannot produce.

Recipe `ngldm3d.mirp_fbn64`: the segmented phantom (tests/data/nifti/phantoms/ut_inten.nii +
ut_mask57.nii, label 57) at native 1x1x1 spacing, `by_slice=False`, fixed_bin_number with 64 bins,
distance 1, difference level (alpha) 0 -- the IBSI NGLDM coarseness. On the Nyxus side that is
GREYDEPTH=64 and IBSI=false, which is what test_3d_ngldm_regression.h sets, so the two are binned
alike and the comparison is config-matched.

READ THIS BEFORE PINNING ANYTHING FROM THIS RUN: Nyxus' 3D NGLDM disagrees with MIRP on 16 of the 17
comparable features, several by an order of magnitude, and two concrete causes are visible in
src/nyx/features/3d_ngldm.cpp -- see ../audit/ngldm_3d_mirp_vetting_report.md. Only 3NGLDM_DCP
matches, and it matches at the degenerate value 1.0. Treat this generator as the measurement behind
that report, not as a licence to promote rows.

3NGLDM_GLM (grey level mean) and 3NGLDM_DCM (dependence count mean) have no MIRP counterpart -- its
NGLDM emits no gl_mean / dc_mean column -- so they cannot be vetted here at all.

NIFTI READING WITHOUT A NIFTI LIBRARY: the mirp env has neither SimpleITK nor nibabel. The phantoms
are uncompressed single-file NIfTI-1 (magic "n+1"), so the header is parsed directly below and the
generator stays single-env. Same approach as gen_morphology3d_mirp.py.

Provenance: tool=mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3, Python 3.11); env=nyxus_mirp (conda-forge:
`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`);
generator=tests/vetting/oracles/gen_ngldm3d_mirp.py. Run offline; CI never invokes it.
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
TEST_H = os.path.join(TESTS, "test_3d_ngldm_mirp.h")

RELTOL = 1e-9
LABEL = 57
NBINS = 64

# Nyxus feature -> MIRP NGLDM column stem. MIRP suffixes every column with the neighbourhood and
# discretisation it was computed at (`_d1_a0.0_3d_fbn_n64`), so match on the stem and assert the
# suffix separately -- otherwise a changed bin count silently reads a column from another config.
MIRP = {
    "3NGLDM_LDE": "ngl_lde",
    "3NGLDM_HDE": "ngl_hde",
    "3NGLDM_LGLCE": "ngl_lgce",
    "3NGLDM_HGLCE": "ngl_hgce",
    "3NGLDM_LDLGLE": "ngl_ldlge",
    "3NGLDM_LDHGLE": "ngl_ldhge",
    "3NGLDM_HDLGLE": "ngl_hdlge",
    "3NGLDM_HDHGLE": "ngl_hdhge",
    "3NGLDM_GLNU": "ngl_glnu",
    "3NGLDM_GLNUN": "ngl_glnu_norm",
    "3NGLDM_DCNU": "ngl_dcnu",
    "3NGLDM_DCNUN": "ngl_dcnu_norm",
    "3NGLDM_DCP": "ngl_dc_perc",
    "3NGLDM_GLV": "ngl_gl_var",
    "3NGLDM_DCV": "ngl_dc_var",
    "3NGLDM_DCENT": "ngl_dc_entr",
    "3NGLDM_DCENE": "ngl_dc_energy",
}

SUFFIX = f"_d1_a0.0_3d_fbn_n{NBINS}"

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
    vol = np.frombuffer(raw, NIFTI_DTYPE[datatype], nx * ny * nz, vox_offset).reshape((nz, ny, nx))
    return vol, (float(pixdim[3]), float(pixdim[2]), float(pixdim[1]))


def parse_pins(txt, table):
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found")
    body = txt[m.end():].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"(3NGLDM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def run():
    import mirp
    logging.disable(logging.INFO)

    inten, spacing = read_nifti(INTEN)
    mask_vol, _ = read_nifti(MASK)
    mask = (mask_vol == LABEL).astype(np.int32)
    print(f"# volume {inten.shape}, roi voxels {int(mask.sum())}, spacing zyx {spacing}")

    res = mirp.extract_features(
        image=inten.astype(np.float64), mask=mask,
        image_spacing=spacing,
        by_slice=False,
        base_feature_families="ngldm",
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=NBINS,
    )
    df = res[0] if isinstance(res, list) else res
    row = df.iloc[0]
    out = {}
    for nyx, stem in MIRP.items():
        col = stem + SUFFIX
        if col not in df.columns:
            raise RuntimeError(f"MIRP produced no {col} (for {nyx}); columns present: "
                               f"{sorted(c for c in df.columns if c.startswith('ngl_'))}")
        out[nyx] = float(row[col])
    return out


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1

    got = run()

    import mirp
    print(f"# mirp 2.6.0, numpy {np.__version__}, label={LABEL}, by_slice=False, "
          f"fixed_bin_number n={NBINS}, distance=1, alpha=0")
    print("# paste-ready goldens")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// {MIRP[name]}{SUFFIX}")
    print("\n# no MIRP counterpart, cannot be vetted here: 3NGLDM_GLM, 3NGLDM_DCM")

    if not os.path.exists(TEST_H):
        print(f"\n# {os.path.basename(TEST_H)} does not exist yet -- nothing to verify")
        return 0

    pins = parse_pins(open(TEST_H, encoding="utf-8", errors="replace").read(),
                      "ngldm_3d_mirp_ref_vals")
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

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible")
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
