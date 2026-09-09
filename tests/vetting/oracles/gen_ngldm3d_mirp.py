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

It feeds two artifacts and verifies both. ../../test_3d_ngldm_mirp.h pins the samelevels values as
oracle goldens, and ../audit/ngldm_3d_mirp_vetting_report.md publishes both comparison tables. Every
column of both tables is checked, not just MIRP's: the Nyxus column against the goldens pinned in the
C++ headers, and the derived column recomputed from the two. A generator that verifies only the half
it produced would let the other half drift while still printing "ALL CHECKS PASSED".

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
NGLDM emits no gl_mean / dc_mean column -- so they cannot be vetted here at all. 3NGLDM_DCP is
computed by both and deliberately not pinned: Nyxus hard-codes it to 1 and MIRP returns 1 on any
input where every voxel has a same-level neighbour, so the check below allows exactly that one
omission and fails on any other.

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
REGRESSION_H = os.path.join(TESTS, "test_3d_ngldm_regression.h")
MIRP_H = os.path.join(TESTS, "test_3d_ngldm_mirp.h")

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
    """-> {feature: (nyxus, reference, derived)} from the comparison table under `heading`.

    Rows are | `3NGLDM_GLNU` | 6480.48 | 4350.27 | 1.49x | -- Nyxus, MIRP, and a derived column that
    is a ratio in one table and a relative difference in the other. ALL THREE are returned, because
    all three are claims: the reference column is checked against a fresh MIRP run, the Nyxus column
    against the pins checked into the C++ headers, and the derived column is recomputed from the two.
    Verifying only the reference column would let every Nyxus number and every ratio in the report
    drift while this generator still printed success.

    The report carries one such table per run, so the section is located first and the search stops
    at the next heading of the same level.
    """
    m = re.search(r"^" + re.escape(heading) + r"\s*$", txt, re.M)
    if not m:
        raise RuntimeError(f"no {heading!r} section in " + os.path.basename(REPORT))
    rest = txt[m.end():]
    nxt = re.search(r"^##\s", rest, re.M)
    section = rest[: nxt.start()] if nxt else rest
    rows = re.findall(
        r"^\|\s*`(3NGLDM_[A-Z0-9_]+)`\s*\|\s*([-0-9.eE+]+)\s*\|\s*([-0-9.eE+]+)\s*\|"
        r"\s*([-0-9.eE+]+)x?\s*\|",
        section, re.M)
    if not rows:
        raise RuntimeError(f"no comparison rows under {heading!r} in " + os.path.basename(REPORT))
    return {n: (float(a), float(b), float(c)) for n, a, b, c in rows}


def parse_pins(path, table):
    """-> {feature: value} from a `ref_vals_map<double> <table>` initialiser in a C++ header.

    These are the checked-in goldens -- what Nyxus is actually asserted against -- so they are the
    source the report's Nyxus column has to agree with. Reading them here is what ties the prose to
    the tests rather than to a run nobody can reproduce.
    """
    src = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"ref_vals_map<double>\s+" + re.escape(table) + r"\s*\{(.*?)\n\}", src, re.S)
    if not m:
        raise RuntimeError(f"no {table} initialiser in {os.path.basename(path)}")
    pins = re.findall(r'\{\s*"(3NGLDM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', m.group(1))
    if not pins:
        raise RuntimeError(f"{table} in {os.path.basename(path)} holds no 3NGLDM pins")
    return {n: float(v) for n, v in pins}


def nyxus_raw_levels(inten):
    """-> the volume Nyxus sees after the loader's shift, which IS the grey level when IBSI=true.

    Widen before subtracting: NIfTI datatype 4 is int16, and a volume spanning more than 32767 would
    wrap in the file's own dtype and stop reproducing what Nyxus sees.
    """
    return (inten.astype(np.int64) - int(inten.min())).astype(np.uint32)


def nyxus_grey_levels(inten, mask):
    """-> the volume of grey levels Nyxus' GREYDEPTH=64 binning produces, off-ROI voxels at 0.

    Two steps, both Nyxus-side and both visible in its output: the loader shifts every voxel by the
    volume minimum, and `to_grayscale(i, 0, ROI max, 64)` truncates i / (ROI max) * 64. Handing the
    result to MIRP with the discretisation switched off is what makes the samelevels run comparable.
    """
    shifted = nyxus_raw_levels(inten)
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

    # The IBSI=true config point. IBSI reaches to_grayscale as disable_binning, so Nyxus does not bin
    # at all there and the raw loader-shifted intensity is the grey level. Nothing is reproduced for
    # MIRP beyond that shift -- there is no binning step to replicate -- which is why this recipe's
    # rows are not scope-narrowed the way mirp_samelevels' are.
    raw = np.where(m, nyxus_raw_levels(inten), 0).astype(np.float64)
    ibsi = features(raw, SUFFIX_SAMELEVELS, base_discretisation_method="none")
    print(f"# ibsi=true grey levels over the roi: {int(np.unique(raw[m]).size)} distinct raw values")
    return (fbn, same, ibsi), span


def verify(got, heading, txt, pins, derived, bound):
    """-> (n verified, n failed, n unproducible, [unquoted]) for one of the report's tables.

    Three checks per row, because a comparison table makes three claims:
      1. the MIRP column against this run,
      2. the Nyxus column against the goldens checked into the C++ headers,
      3. the derived column recomputed from the two by `derived`.
    A row passes only if all three do.
    """
    quoted = parse_report(txt, heading)
    print(f"\n# verifying the {len(quoted)} rows quoted under {heading!r} -- MIRP column against "
          f"this run, Nyxus column against the C++ pins, derived column recomputed; rel<={RELTOL:g}")
    nok = nfail = nmiss = 0
    for name in sorted(quoted):
        want_ny, want_mirp, want_d = quoted[name]
        if name not in got:
            print(f"  MISSING {name}: report quotes {want_mirp!r} but MIRP reports no counterpart")
            nmiss += 1
            continue
        if name not in pins:
            print(f"  MISSING {name}: report quotes a Nyxus value but no C++ pin backs it")
            nmiss += 1
            continue
        have_mirp, have_ny = got[name], pins[name]
        rel_mirp = abs(have_mirp - want_mirp) / max(abs(want_mirp), 1e-12)
        rel_ny = abs(have_ny - want_ny) / max(abs(want_ny), 1e-12)
        have_d = derived(have_ny, have_mirp)
        bad = [n for n, r in (("mirp", rel_mirp), ("nyxus", rel_ny)) if r > RELTOL]
        # The two derived columns are different kinds of claim and are checked differently. The
        # fbn64 ratio is a value, quoted at six significant figures and verified as one. The
        # samelevels column is a residual whose magnitude is float noise -- comparing 8.7e-16
        # against 9e-16 as a value would be checking the report's rounding, not the tool -- so it
        # is quoted as a bound rounded UP and verified as one: the run must not exceed it.
        if bound:
            if have_d > want_d:
                bad.append("derived")
        elif abs(have_d - want_d) / max(abs(want_d), 1e-12) > RELTOL:
            bad.append("derived")
        if not bad:
            print(f"  OK   {name}: mirp={have_mirp!r} nyxus={have_ny!r} "
                  f"derived={have_d:.6g} {'<=' if bound else '=='} {want_d!r}")
            nok += 1
        else:
            print(f"  FAIL {name} [{','.join(bad)}]: mirp={have_mirp!r} vs {want_mirp!r} "
                  f"(rel {rel_mirp:.3g}); nyxus={have_ny!r} vs {want_ny!r} (rel {rel_ny:.3g}); "
                  f"derived={have_d:.6g} vs {want_d!r}")
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

    (fbn, same, ibsi), span = run()

    try:
        version = metadata.version("mirp")       # mirp exposes no __version__
    except metadata.PackageNotFoundError:
        version = "unknown"

    # the installed mirp, read from the distribution rather than written in: this line is the
    # provenance of the values printed below it, so a run under another version has to say so.
    print(f"# mirp {version}, numpy {np.__version__}, label={LABEL}, by_slice=False, "
          f"distance=1, alpha=0")
    for title, got, suffix in (("ngldm3d.mirp_fbn64", fbn, SUFFIX_FBN),
                               ("ngldm3d.mirp_samelevels", same, SUFFIX_SAMELEVELS),
                               ("ngldm3d.mirp_ibsi_rawlevels", ibsi, SUFFIX_SAMELEVELS)):
        print(f"\n# {title}")
        for name in sorted(got):
            print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// {MIRP[name]}{suffix}")
    print("\n# no MIRP counterpart at either recipe: 3NGLDM_GLM, 3NGLDM_DCM")

    if not os.path.exists(REPORT):
        print(f"\n# {os.path.basename(REPORT)} is missing -- there is nothing this run can be "
              f"checked against, which is a failure, not a pass")
        return 1

    txt = open(REPORT, encoding="utf-8", errors="replace").read()

    # The Nyxus half of both tables is one run at GREYDEPTH=64 / IBSI=false, so both are backed by
    # the regression pins. The oracle header's own pins are checked against this MIRP run separately
    # below -- they are goldens, not a report column.
    # The fbn64 and samelevels tables share one Nyxus run at GREYDEPTH=64 / IBSI=false; three of its
    # nineteen values are pinned in the regression header and the other sixteen in the oracle header,
    # so the Nyxus column is backed by whichever table owns the feature. The ibsi table is a
    # different Nyxus run and is backed by its own oracle pins -- checked below in their own pass, so
    # here the Nyxus column is compared against those same pins.
    pins = dict(parse_pins(REGRESSION_H, "ngldm_3d_regression_ref_vals"))
    pins.update(parse_pins(MIRP_H, "ngldm_3d_mirp_ref_vals"))
    ibsi_pins = dict(parse_pins(MIRP_H, "ngldm_3d_mirp_ibsi_ref_vals"))
    # 3NGLDM_DCP is quoted in every table but pinned only once: Nyxus hard-codes f_DCP = 1, so its
    # value does not depend on the config point and the regression pin backs it at both.
    ibsi_pins["3NGLDM_DCP"] = parse_pins(REGRESSION_H, "ngldm_3d_regression_ref_vals")["3NGLDM_DCP"]

    nok = nfail = nmiss = 0
    unquoted = []
    for heading, got, derived, bound in (
            ("## Result at `ngldm3d.mirp_fbn64` -- MIRP discretises", fbn,
             lambda ny, ref: ny / ref, False),                  # a ratio: verified as a value
            ("## Result at `ngldm3d.mirp_samelevels` -- the same grey levels", same,
             lambda ny, ref: abs(ny - ref) / max(abs(ref), 1e-12), True),    # a residual: a bound
            ("## Result at `ngldm3d.mirp_ibsi_rawlevels` -- the IBSI=true config point", ibsi,
             lambda ny, ref: abs(ny - ref) / max(abs(ref), 1e-12), True)):
        table_pins = ibsi_pins if "ibsi_rawlevels" in heading else pins
        a, b, c, d = verify(got, heading, txt, table_pins, derived, bound)
        nok += a; nfail += b; nmiss += c; unquoted += d

    # The oracle headers' goldens ARE this run's MIRP values -- that is what makes
    # test_3d_ngldm_mirp.h an assertion against MIRP rather than against a number someone typed.
    # One table per config point, each checked against the run that produced it.
    for table, got in (("ngldm_3d_mirp_ref_vals", same),
                       ("ngldm_3d_mirp_ibsi_ref_vals", ibsi)):
        oracle_pins = parse_pins(MIRP_H, table)
        print(f"\n# verifying the {len(oracle_pins)} goldens pinned in {table} against this run, "
              f"at rel<={RELTOL:g}")
        for name in sorted(oracle_pins):
            if name not in got:
                print(f"  MISSING {name}: pinned as a MIRP golden but MIRP reports no counterpart")
                nmiss += 1
                continue
            rel = abs(oracle_pins[name] - got[name]) / max(abs(got[name]), 1e-12)
            if rel <= RELTOL:
                print(f"  OK   {name}: pin={oracle_pins[name]!r} mirp={got[name]!r} rel={rel:.3g}")
                nok += 1
            else:
                print(f"  FAIL {name}: pin={oracle_pins[name]!r} mirp={got[name]!r} rel={rel:.3g}")
                nfail += 1
        # a feature MIRP can vet that the header does not pin has to be a deliberate omission
        for name in sorted(set(got) - set(oracle_pins)):
            if name != "3NGLDM_DCP":
                print(f"  FAIL {name}: MIRP computes it and {table} does not pin it")
                nfail += 1

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
