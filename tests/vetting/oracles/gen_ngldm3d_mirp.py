"""OFFLINE MIRP oracle for the 3D NGLDM features, on the segmented phantom.

    python tests/vetting/oracles/gen_ngldm3d_mirp.py     (from the repository root)

Runs MIRP twice and re-verifies both comparison tables in the vetting report, exiting non-zero on
any mismatch, on any quoted feature it cannot produce, and on any feature it produces that the report
does not quote.

- `ngldm3d.mirp_fbn64` -- MIRP discretises the ROI itself, `fixed_bin_number` n=64. The two sides
  land on different grey levels (below), so this table measures the discretisation gap.
- `ngldm3d.mirp_samelevels` -- MIRP is handed the grey levels Nyxus bins to and told
  `base_discretisation_method="none"`, so both compute the NGLDM over identical levels and the table
  measures the NGLDM itself. Nyxus agrees here to machine precision.

Nothing in the tree PINS these numbers: the family is regression-only, so the artifact this generator
feeds is ../audit/ngldm_3d_mirp_vetting_report.md rather than a header. That is what it verifies. A
generator that has nothing to check exits 0 without checking anything, and its "ALL CHECKS PASSED"
then means only that it ran.

The fixture for both runs is the segmented phantom (tests/data/nifti/phantoms/ut_inten.nii +
ut_mask57.nii, label 57) at native 1x1x1 spacing, `by_slice=False`, distance 1, difference level
(alpha) 0 -- the IBSI NGLDM coarseness. The Nyxus side is GREYDEPTH=64, IBSI=false, which is what
test_3d_ngldm_regression.h sets.

The grey levels the two tools reach are NOT the same, which is why there are two runs. MIRP's
fixed_bin_number spreads this ROI over levels 1-64. Nyxus bins with `to_grayscale(i, 0, ROI max, 64)`
over a volume whose minimum has been shifted to 0, so the ROI -- which occupies the upper two thirds
of that shifted range -- lands on levels 21-64, 44 of them distinct. `nyxus_grey_levels()` below
reproduces that mapping; the agreement of the samelevels run is what confirms the reproduction.

READ THIS BEFORE PINNING ANYTHING FROM THE fbn64 RUN: it is not a config-matched comparison. The
matched one is samelevels.

3NGLDM_GLM (grey level mean) and 3NGLDM_DCM (dependence count mean) have no MIRP counterpart -- its
NGLDM emits no gl_mean / dc_mean column -- so they cannot be vetted here at all.

NIFTI READING WITHOUT A NIFTI LIBRARY: the mirp env has neither SimpleITK nor nibabel. The phantoms
are uncompressed single-file NIfTI-1 (magic "n+1"), so the header is parsed directly below and the
generator stays single-env. Same approach as gen_morphology3d_mirp.py.

Provenance of the run behind the vetting report -- the printed header names whatever mirp is
actually installed, so a run under another version says so rather than repeating this line:
tool=mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3, Python 3.11); env=nyxus_mirp (conda-forge:
`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`);
generator=tests/vetting/oracles/gen_ngldm3d_mirp.py. Run offline; CI never invokes it.
"""
import logging
import os
import re
import sys
from importlib import metadata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
PHANTOMS = os.path.join(TESTS, "data", "nifti", "phantoms")
INTEN = os.path.join(PHANTOMS, "ut_inten.nii")
MASK = os.path.join(PHANTOMS, "ut_mask57.nii")
REPORT = os.path.join(TESTS, "vetting", "audit", "ngldm_3d_mirp_vetting_report.md")

# The report quotes the comparison to six significant figures, so that is the precision the
# re-verification can hold it to. It is a staleness check on a published table, not a vetting band.
RELTOL = 1e-5
LABEL = 57
NBINS = 64

# Nyxus feature -> MIRP NGLDM column stem. MIRP suffixes every column with the neighbourhood and
# discretisation it was computed at (`_d1_a0.0_3d_fbn_n64` when it discretises, `_d1_a0.0_3d` when it
# does not), so match on the stem and assert the suffix separately -- otherwise a changed bin count
# silently reads a column from another config.
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

SUFFIX_FBN = f"_d1_a0.0_3d_fbn_n{NBINS}"
SUFFIX_SAMELEVELS = "_d1_a0.0_3d"   # MIRP drops the discretisation stem when there is none

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


def parse_report(txt, heading):
    """-> {feature: MIRP value} from the comparison table under `heading`.

    Rows look like | `3NGLDM_GLNU` | 5636.02 | 4350.27 | 1.30x | -- Nyxus, MIRP, ratio. Only the
    MIRP column is this generator's to verify; the Nyxus column is the program's own output and the
    ratio is derived from the two. The report carries one such table per run, so the section is
    located first and the search stops at the next heading of the same level.
    """
    m = re.search(r"^" + re.escape(heading) + r"\s*$", txt, re.M)
    if not m:
        raise RuntimeError(f"no {heading!r} section in " + os.path.basename(REPORT))
    rest = txt[m.end():]
    nxt = re.search(r"^##\s", rest, re.M)
    section = rest[: nxt.start()] if nxt else rest
    rows = re.findall(
        r"^\|\s*`(3NGLDM_[A-Z0-9_]+)`\s*\|\s*[-0-9.eE+]+\s*\|\s*([-0-9.eE+]+)\s*\|",
        section, re.M)
    if not rows:
        raise RuntimeError(f"no comparison rows under {heading!r} in " + os.path.basename(REPORT))
    return {n: float(v) for n, v in rows}


def nyxus_grey_levels(inten, mask):
    """-> the volume of grey levels Nyxus' GREYDEPTH=64 binning produces, off-ROI voxels at 0.

    Two steps, both Nyxus-side and both visible in its output: the loader shifts every voxel by the
    volume minimum, and `to_grayscale(i, 0, ROI max, 64)` truncates i / (ROI max) * 64. Handing the
    result to MIRP with the discretisation switched off is what makes the samelevels run comparable.
    """
    shifted = (inten - inten.min()).astype(np.uint32)
    roi_max = float(shifted[mask].max())
    levels = (shifted.astype(np.float64) / roi_max * NBINS).astype(np.uint32)
    return np.where(mask, levels, 0).astype(np.float64)


def run():
    """-> ((fbn64 features, samelevels features), the grey-level span of the Nyxus binning)."""
    import mirp
    logging.disable(logging.INFO)

    inten, spacing = read_nifti(INTEN)
    mask_vol, _ = read_nifti(MASK)
    m = mask_vol == LABEL
    mask = m.astype(np.int32)
    print(f"# volume {inten.shape}, roi voxels {int(mask.sum())}, spacing zyx {spacing}")

    levels = nyxus_grey_levels(inten, m)
    lv_roi = levels[m]
    span = (int(lv_roi.min()), int(lv_roi.max()), int(np.unique(lv_roi).size))
    print(f"# nyxus grey levels over the roi: {span[0]}-{span[1]}, {span[2]} distinct; "
          f"mirp fixed_bin_number n={NBINS} spreads the same roi over 1-{NBINS}")

    def features(image, suffix, **discretisation):
        res = mirp.extract_features(
            image=image, mask=mask, image_spacing=spacing,
            by_slice=False, base_feature_families="ngldm", **discretisation)
        df = res[0] if isinstance(res, list) else res
        row = df.iloc[0]
        out = {}
        for nyx, stem in MIRP.items():
            col = stem + suffix
            if col not in df.columns:
                raise RuntimeError(f"MIRP produced no {col} (for {nyx}); columns present: "
                                   f"{sorted(c for c in df.columns if c.startswith('ngl_'))}")
            out[nyx] = float(row[col])
        return out

    fbn = features(inten.astype(np.float64), SUFFIX_FBN,
                   base_discretisation_method="fixed_bin_number",
                   base_discretisation_n_bins=NBINS)
    same = features(levels, SUFFIX_SAMELEVELS, base_discretisation_method="none")
    return (fbn, same), span


def verify(got, heading, txt):
    """-> (n verified, n failed, n unproducible, [unquoted]) for one of the report's tables."""
    quoted = parse_report(txt, heading)
    print(f"\n# verifying the {len(quoted)} MIRP values quoted under {heading!r} "
          f"against this run, at rel<={RELTOL:g}")
    nok = nfail = nmiss = 0
    for name in sorted(quoted):
        want = quoted[name]
        if name not in got:
            print(f"  MISSING {name}: report quotes {want!r} but MIRP reports no counterpart")
            nmiss += 1
            continue
        have = got[name]
        rel = abs(have - want) / max(abs(want), 1e-12)
        if rel <= RELTOL:
            print(f"  OK   {name}: mirp={have!r} report={want!r} rel={rel:.3g}")
            nok += 1
        else:
            print(f"  FAIL {name}: mirp={have!r} report={want!r} rel={rel:.3g}")
            nfail += 1

    # the reverse direction: a feature this run produces that the report says nothing about
    unquoted = sorted(set(got) - set(quoted))
    for name in unquoted:
        print(f"  UNQUOTED {name}: MIRP reports {got[name]!r} and the report does not quote it")
    return nok, nfail, nmiss, unquoted


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1

    (fbn, same), span = run()

    try:
        version = metadata.version("mirp")       # mirp exposes no __version__
    except metadata.PackageNotFoundError:
        version = "unknown"

    # the installed mirp, read from the distribution rather than written in: this line is the
    # provenance of the values printed below it, so a run under another version has to say so.
    print(f"# mirp {version}, numpy {np.__version__}, label={LABEL}, by_slice=False, "
          f"distance=1, alpha=0")
    for title, got, suffix in (("ngldm3d.mirp_fbn64", fbn, SUFFIX_FBN),
                               ("ngldm3d.mirp_samelevels", same, SUFFIX_SAMELEVELS)):
        print(f"\n# {title}")
        for name in sorted(got):
            print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// {MIRP[name]}{suffix}")
    print("\n# no MIRP counterpart at either recipe: 3NGLDM_GLM, 3NGLDM_DCM")

    if not os.path.exists(REPORT):
        print(f"\n# {os.path.basename(REPORT)} is missing -- there is nothing this run can be "
              f"checked against, which is a failure, not a pass")
        return 1

    txt = open(REPORT, encoding="utf-8", errors="replace").read()
    nok = nfail = nmiss = 0
    unquoted = []
    for heading, got in (("## Result at `ngldm3d.mirp_fbn64` -- MIRP discretises", fbn),
                         ("## Result at `ngldm3d.mirp_samelevels` -- the same grey levels", same)):
        a, b, c, d = verify(got, heading, txt)
        nok += a; nfail += b; nmiss += c; unquoted += d

    # the grey-level span is the report's explanation of the fbn64 gap, so it is checked too
    span_txt = f"{span[0]}-{span[1]}, {span[2]} distinct"
    if span_txt not in txt:
        print(f"\nFAIL grey-level span: this run measures {span_txt}, which the report does not state")
        nfail += 1
    else:
        print(f"\nOK   grey-level span: report states {span_txt}")

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible, {len(unquoted)} unquoted")
    if nfail or nmiss or unquoted:
        print("SOME CHECKS FAILED -- the report is out of step with the tool")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
