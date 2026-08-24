"""OFFLINE MIRP oracle for the 3D GLDZM features, on the segmented phantom.

    python tests/vetting/oracles/gen_gldzm3d_mirp.py     (from the repository root)

This generator pins nothing, because no 3D GLDZM row is vetted. Its job is to make the measurement
behind tests/vetting/audit/gldzm_3d_mirp_vetting_report.md reproducible in one command, and to
establish -- rather than assert -- what the correct answer is.

It does three things:

  1. Runs MIRP, which implements IBSI GLDZM. PyRadiomics has no GLDZM at all, so MIRP is the only
     mainstream oracle for this family.
  2. Recomputes the same 16 features from scratch with an independent implementation: zones are the
     26-connected components of each discretised grey level, and the distance to the ROI border is
     a city-block distance transform. This is a different route to the same definition, not a
     re-derivation of Nyxus' steps (SPEC 5.2), and it reproduces MIRP -- which is what licenses the
     claim that the definition is reachable and that Nyxus is not computing it.
  3. Recomputes them once more with Nyxus' straight-ray distance in place of the distance
     transform, to show that the ray metric is NOT the source of Nyxus' disagreement: it moves the
     features by well under a percent.

Recipe `gldzm3d.mirp_fbn64`: the segmented phantom (tests/data/nifti/phantoms/ut_inten.nii +
ut_mask57.nii, label 57) at native 1x1x1 spacing, `by_slice=False`, fixed_bin_number with 64 bins.
On the Nyxus side that is GREYDEPTH=64 with IBSI=false, which is what test_3d_gldzm_regression.h
sets, so the two are binned alike and the comparison is config-matched.

Provenance: tool=mirp 2.6.0 (numpy 2.4.6, scipy 1.17.1, Python 3.11); env=nyxus_mirp (conda-forge:
`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy scipy`);
generator=tests/vetting/oracles/gen_gldzm3d_mirp.py. Run offline; CI never invokes it.
"""
import logging
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
PHANTOMS = os.path.join(TESTS, "data", "nifti", "phantoms")
INTEN = os.path.join(PHANTOMS, "ut_inten.nii")
MASK = os.path.join(PHANTOMS, "ut_mask57.nii")

LABEL = 57
NBINS = 64
SUFFIX = f"_3d_fbn_n{NBINS}"
RELTOL = 1e-9

NIFTI_DTYPE = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}

# Nyxus feature -> MIRP GLDZM column stem.
MIRP = {
    "3GLDZM_SDE": "dzm_sde",
    "3GLDZM_LDE": "dzm_lde",
    "3GLDZM_LGLZE": "dzm_lgze",
    "3GLDZM_HGLZE": "dzm_hgze",
    "3GLDZM_SDLGLE": "dzm_sdlge",
    "3GLDZM_SDHGLE": "dzm_sdhge",
    "3GLDZM_LDLGLE": "dzm_ldlge",
    "3GLDZM_LDHGLE": "dzm_ldhge",
    "3GLDZM_GLNU": "dzm_glnu",
    "3GLDZM_GLNUN": "dzm_glnu_norm",
    "3GLDZM_ZDNU": "dzm_zdnu",
    "3GLDZM_ZDNUN": "dzm_zdnu_norm",
    "3GLDZM_ZP": "dzm_z_perc",
    "3GLDZM_GLV": "dzm_gl_var",
    "3GLDZM_ZDV": "dzm_zd_var",
    "3GLDZM_ZDE": "dzm_zd_entr",
}

# No MIRP counterpart and no IBSI definition: MIRP emits no dzm_gl_mean or dzm_zd_mean.
NO_COUNTERPART = ("3GLDZM_GLM", "3GLDZM_ZDM")


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


def discretise(inten, mask):
    """fixed_bin_number: levels 1..NBINS over the ROI's own range."""
    v = inten[mask].astype(np.float64)
    lo, hi = v.min(), v.max()
    b = np.floor((v - lo) / (hi - lo) * NBINS).astype(np.int32) + 1
    b[b > NBINS] = NBINS
    out = np.zeros(inten.shape, np.int32)
    out[mask] = b
    return out


def ray_distance(mask):
    """Nyxus' metric: walk each of the 6 axis rays until it leaves the ROI, take the shortest."""
    d = np.full(mask.shape, 1 << 30, np.int64)
    for axis in range(3):
        for direction in (1, -1):
            m = np.moveaxis(mask, axis, 0)
            if direction == -1:
                m = m[::-1]
            run = np.zeros(m.shape[1:], np.int64)
            acc = np.empty(m.shape, np.int64)
            for i in range(m.shape[0]):
                run = np.where(m[i], run + 1, 0)
                acc[i] = run
            if direction == -1:
                acc = acc[::-1]
            d = np.minimum(d, np.moveaxis(acc, 0, axis))
    return d


def gldzm_features(binned, distance, n_roi):
    """IBSI GLDZM from 26-connected zones and the given distance-to-border map."""
    struct = np.ones((3, 3, 3), dtype=bool)          # 26-connectivity
    zones = []
    for g in range(1, NBINS + 1):
        sel = binned == g
        if not sel.any():
            continue
        lab, n = ndimage.label(sel, structure=struct)
        if n == 0:
            continue
        idx = range(1, n + 1)
        mins = ndimage.minimum(distance, lab, index=idx)
        for dmin in np.atleast_1d(mins):
            zones.append((g, int(dmin)))

    Nd = max(z[1] for z in zones)
    P = np.zeros((NBINS, Nd), dtype=np.float64)
    for g, dmin in zones:
        P[g - 1, dmin - 1] += 1.0

    Ns = P.sum()
    p = P / Ns
    g = np.arange(1, NBINS + 1)
    d = np.arange(1, Nd + 1)
    gi, di = g[:, None], d[None, :]
    Mg, Md = p.sum(axis=1), p.sum(axis=0)
    mu_g, mu_d = (Mg * g).sum(), (Md * d).sum()
    nz = p[p > 0]
    return len(zones), {
        "dzm_sde": (Md / d ** 2).sum(),
        "dzm_lde": (Md * d ** 2).sum(),
        "dzm_lgze": (Mg / g ** 2).sum(),
        "dzm_hgze": (Mg * g ** 2).sum(),
        "dzm_sdlge": (p / (gi ** 2 * di ** 2)).sum(),
        "dzm_sdhge": (p * gi ** 2 / di ** 2).sum(),
        "dzm_ldlge": (p * di ** 2 / gi ** 2).sum(),
        "dzm_ldhge": (p * gi ** 2 * di ** 2).sum(),
        "dzm_glnu": (P.sum(axis=1) ** 2).sum() / Ns,
        "dzm_glnu_norm": (Mg ** 2).sum(),
        "dzm_zdnu": (P.sum(axis=0) ** 2).sum() / Ns,
        "dzm_zdnu_norm": (Md ** 2).sum(),
        "dzm_z_perc": Ns / n_roi,
        "dzm_gl_var": (Mg * (g - mu_g) ** 2).sum(),
        "dzm_zd_var": (Md * (d - mu_d) ** 2).sum(),
        "dzm_zd_entr": -(nz * np.log2(nz)).sum(),
    }


def run_mirp(inten, mask, spacing):
    import mirp
    logging.disable(logging.INFO)
    res = mirp.extract_features(
        image=inten.astype(np.float64), mask=mask.astype(np.int32),
        image_spacing=spacing,
        by_slice=False,
        base_feature_families="gldzm",
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=NBINS,
    )
    df = res[0] if isinstance(res, list) else res
    row = df.iloc[0]
    out = {}
    for stem in set(MIRP.values()):
        col = stem + SUFFIX
        if col not in df.columns:
            raise RuntimeError(f"MIRP produced no {col}; dzm columns present: "
                               f"{sorted(c for c in df.columns if c.startswith('dzm_'))}")
        out[stem] = float(row[col])
    return out


def compare(title, ref, got, tol):
    print(f"\n=== {title} ===")
    print("%-16s %20s %20s %12s" % ("mirp column", "mirp", "computed", "rel"))
    worst, bad = 0.0, 0
    for stem in sorted(ref):
        a, b = ref[stem], got[stem]
        denom = max(abs(a), abs(b))
        rel = 0.0 if denom == 0 else abs(a - b) / denom
        worst = max(worst, rel)
        flag = "" if rel <= tol else "   <-- differs"
        if rel > tol:
            bad += 1
        print("%-16s %20.12g %20.12g %12.3e%s" % (stem, a, b, rel, flag))
    print("worst rel = %.3e over %d features" % (worst, len(ref)))
    return worst, bad


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1

    inten, spacing = read_nifti(INTEN)
    mask = read_nifti(MASK)[0] == LABEL
    n_roi = int(mask.sum())
    print(f"# volume {inten.shape}, roi voxels {n_roi}, spacing zyx {spacing}")
    print(f"# mirp 2.6.0, label={LABEL}, by_slice=False, fixed_bin_number n={NBINS}")

    binned = discretise(inten, mask)
    dt = ndimage.distance_transform_cdt(mask, metric="taxicab").astype(np.int64)
    rd = ray_distance(mask)
    print("# distance to border, over ROI voxels: "
          f"transform mean {dt[mask].mean():.4f} max {dt[mask].max()}, "
          f"straight-ray mean {rd[mask].mean():.4f} max {rd[mask].max()}")

    oracle = run_mirp(inten, mask, spacing)

    n_dt, f_dt = gldzm_features(binned, dt, n_roi)
    worst_dt, bad_dt = compare(
        f"26-connected zones + city-block distance transform ({n_dt} zones)", oracle, f_dt, RELTOL)

    n_rd, f_rd = gldzm_features(binned, rd, n_roi)
    compare(f"the same, with Nyxus' straight-ray distance ({n_rd} zones)", oracle, f_rd, 5e-2)

    print("\n# no MIRP or IBSI counterpart, cannot be vetted here: " + ", ".join(NO_COUNTERPART))

    if bad_dt:
        print(f"\nFAILED: the independent implementation does not reproduce MIRP "
              f"({bad_dt} feature(s) above rel={RELTOL})")
        return 1
    print(f"\nALL CHECKS PASSED: the independent implementation reproduces MIRP to "
          f"rel={worst_dt:.1e}, so the IBSI GLDZM definition is reachable on this fixture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
