"""OFFLINE MIRP oracle for the 3D GLDZM features.

    python tests/vetting/oracles/gen_gldzm3d_mirp.py     (from the repository root)

Runs MIRP three times and re-verifies everything the tree quotes from it, exiting non-zero on any
mismatch, on any quoted feature it cannot produce, and on any feature it produces that nothing
quotes.

EVERY NUMBER THE REPORT PUBLISHES IS CHECKED, not only the MIRP column: each table's Nyxus column
is verified against its source in the tree (the oracle pins for the vetted cell, the regression pins
for the other two) and each table's ratio or relative-residual column is recomputed from the row's
own two values and compared as the string the report writes. A corrupted Nyxus value or a corrupted
ratio fails the run.

  1. `gldzm3d.mirp_compat_phantom` -- the VETTED cell, and the only one whose numbers are pinned.
     The GLDZM compatibility phantom this file also builds, read by both tools as grey levels 1..8
     directly: MIRP with `base_discretisation_method="none"`, Nyxus at `IBSI=true`, which switches
     the family's binning off. Verified against the pins in tests/test_3d_gldzm_mirp.h.
  2. `gldzm3d.mirp_fbn64` -- the drift guard's configuration on the segmented phantom, with MIRP
     discretising the ROI itself at `fixed_bin_number` n=64. The two sides land on different grey
     levels (below), so this table measures the discretisation gap and NOT the GLDZM.
  3. `gldzm3d.mirp_samelevels` -- the same fixture with MIRP handed the grey levels Nyxus bins to,
     so the GLDZM itself is what is compared there too.

Runs 2 and 3 pin nothing. The artifact they feed is ../audit/gldzm_3d_mirp_vetting_report.md, and
that is what this generator checks them against.

MIRP is the only mainstream oracle for this family: PyRadiomics implements no GLDZM at all.

3GLDZM_GLM (grey level mean) and 3GLDZM_ZDM (zone distance mean) have no MIRP counterpart at any
recipe -- its GLDZM emits no dzm_gl_mean / dzm_zd_mean column and IBSI defines neither -- so they
cannot be vetted here and stay drift guards.

BOTH OF THE FAMILY'S FIXTURES ARE BUILT HERE. `--write-phantom` writes the MIRP compatibility
phantom (compat_int/compat_int_gldzm_3d.nii + compat_seg/compat_seg_gldzm_3d.nii) and the zero-level
phantom (phantoms/gldzm_zerolevel_{inten,mask}.nii); every other run rebuilds both in memory and
fails if the checked-in files differ from them, so neither fixture can drift away from the rule that
describes it. The rules, and what each of their choices separates, are in tests/test_3d_gldzm_common.h
beside the phantoms' accessors.

The zero-level phantom carries no MIRP goldens and never will: MIRP's GLDZM has the same problem
with a grey level 0 in its input that Nyxus does, so it cannot be the judge of what to do with one.
It is a mechanics fixture, and test_3d_gldzm_mechanics.h derives its expected values by hand.

NIFTI WITHOUT A NIFTI LIBRARY: the mirp env has neither SimpleITK nor nibabel. The phantoms are
uncompressed single-file NIfTI-1 (magic "n+1"), so the header is parsed and written directly below
and the generator stays single-env. Same approach as gen_ngldm3d_mirp.py.

Provenance of the run behind the pins and the report -- the printed header names whatever mirp is
actually installed, so a run under another version says so rather than repeating this line:
tool=mirp 2.6.0 (numpy 2.4.6, Python 3.11); env=nyxus_mirp (conda-forge:
`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy scipy`);
generator=tests/vetting/oracles/gen_gldzm3d_mirp.py. Run offline; CI never invokes it.
"""
import logging
import os
import re
import sys
from importlib import metadata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
PHANTOMS = os.path.join(DATA, "phantoms")
INTEN = os.path.join(PHANTOMS, "ut_inten.nii")
MASK = os.path.join(PHANTOMS, "ut_mask57.nii")
COMPAT_INTEN = os.path.join(DATA, "compat_int", "compat_int_gldzm_3d.nii")
COMPAT_MASK = os.path.join(DATA, "compat_seg", "compat_seg_gldzm_3d.nii")
ZERO_INTEN = os.path.join(PHANTOMS, "gldzm_zerolevel_inten.nii")
ZERO_MASK = os.path.join(PHANTOMS, "gldzm_zerolevel_mask.nii")
HEADER = os.path.join(TESTS, "test_3d_gldzm_mirp.h")
REGRESSION_HEADER = os.path.join(TESTS, "test_3d_gldzm_regression.h")
REPORT = os.path.join(TESTS, "vetting", "audit", "gldzm_3d_mirp_vetting_report.md")

LABEL = 57
NBINS = 64

# The header pins MIRP's values in full, so they are held to the exact tier the assertions use.
PIN_RELTOL = 1e-12
# The report quotes its comparison to six significant figures, so that is the precision the
# re-verification can hold it to. It is a staleness check on a published table, not a vetting band.
REPORT_RELTOL = 1e-5

# Nyxus feature -> MIRP GLDZM column stem. MIRP suffixes every column with the dimensionality and
# the discretisation it was computed at (`_3d_fbn_n64` when it discretises, `_3d` when it does not),
# so match on the stem and assert the suffix separately -- otherwise a changed bin count silently
# reads a column from another config.
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

SUFFIX_NONE = "_3d"                     # MIRP drops the discretisation stem when there is none
SUFFIX_FBN = "_3d_fbn_n%d" % NBINS

NIFTI_DTYPE = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
               768: np.uint32}

# --- the compatibility phantom -------------------------------------------------------------------

VOL = 16        # volume side
MARGIN = 2      # background voxels on every side
CUBE = 12       # the ROI cube's side, an exact number of bricks
BRICK = 2
NOTCH = 3       # bricks cut out of the cube's low corner, in each axis
IMPLANT = (3, 3, 3)


def build_compat_phantom():
    """-> (intensities, mask), both shaped (z,y,x).

    A 12x12x12 cube of 2x2x2 bricks with a 6x6x6 corner cut out of it, inside a two-voxel background
    margin. A brick's grey level is 1 + 4*(bz%2) + 2*(by%2) + (bx%2) over its brick coordinates,
    except the brick at (3,3,3), which carries 1 instead of the 8 that rule gives it.
    """
    inten = np.zeros((VOL, VOL, VOL), np.float32)
    mask = np.zeros((VOL, VOL, VOL), np.uint32)
    for bz in range(CUBE // BRICK):
        for by in range(CUBE // BRICK):
            for bx in range(CUBE // BRICK):
                if bz < NOTCH and by < NOTCH and bx < NOTCH:
                    continue
                level = 1 + (bz % 2) * 4 + (by % 2) * 2 + (bx % 2)
                if (bz, by, bx) == IMPLANT:
                    level = 1
                z, y, x = (MARGIN + b * BRICK for b in (bz, by, bx))
                inten[z:z + BRICK, y:y + BRICK, x:x + BRICK] = level
                mask[z:z + BRICK, y:y + BRICK, x:x + BRICK] = LABEL
    return inten, mask


# --- the zero-level fixture ----------------------------------------------------------------------

ZERO_VOL = 8        # volume side
ZERO_MARGIN = 2     # background voxels on every side
ZERO_SIDE = 4       # the ROI cube's side
ZERO_HIGH = 5       # the nonzero raw intensity inside the ROI


def build_zero_level_phantom():
    """-> (intensities, mask), both shaped (z,y,x).

    A 4x4x4 ROI inside a two-voxel background margin, whose raw intensities are 0 on the voxels with
    an even x+y+z and 5 on the rest. Its whole point is the zeros: they are ROI voxels, and the two
    binning schemes that do not remap a zero -- IBSI (which bins nothing) and radiomics -- hand them
    to the GLDZM as grey level 0, which is not a valid level. It is what separates a family that
    lifts them onto a valid level from one that drops them, and MIRP cannot be the judge because it
    has the same problem with a level 0 in its input.

    Every number it produces is derivable by hand: each parity class is one 26-connected zone (two
    voxels of one class always touch at least at a corner), every zone reaches the ROI surface so
    both sit at distance 1, and the levels after the lift are 1 and 6.
    """
    inten = np.zeros((ZERO_VOL, ZERO_VOL, ZERO_VOL), np.float32)
    mask = np.zeros((ZERO_VOL, ZERO_VOL, ZERO_VOL), np.uint32)
    for z in range(ZERO_MARGIN, ZERO_MARGIN + ZERO_SIDE):
        for y in range(ZERO_MARGIN, ZERO_MARGIN + ZERO_SIDE):
            for x in range(ZERO_MARGIN, ZERO_MARGIN + ZERO_SIDE):
                mask[z, y, x] = LABEL
                inten[z, y, x] = 0 if (x + y + z) % 2 == 0 else ZERO_HIGH
    return inten, mask


def read_nifti(path):
    """-> (array shaped (z,y,x), spacing (z,y,x)). Uncompressed single-file NIfTI-1 only."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if int(np.frombuffer(raw, np.int32, 1, 0)[0]) != 348 or raw[344:347] != b"n+1":
        raise RuntimeError(path + " is not an uncompressed single-file NIfTI-1")
    dim = np.frombuffer(raw, np.int16, 8, 40)
    datatype = int(np.frombuffer(raw, np.int16, 1, 70)[0])
    pixdim = np.frombuffer(raw, np.float32, 8, 76)
    vox_offset = int(np.frombuffer(raw, np.float32, 1, 108)[0])
    if datatype not in NIFTI_DTYPE:
        raise RuntimeError("%s: unsupported NIfTI datatype %d" % (path, datatype))
    nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
    vol = np.frombuffer(raw, NIFTI_DTYPE[datatype], nx * ny * nz, vox_offset).reshape((nz, ny, nx))
    return vol, (float(pixdim[3]), float(pixdim[2]), float(pixdim[1]))


def write_nifti(path, vol, datatype):
    """Writes 'vol' as an uncompressed single-file NIfTI-1 at 1x1x1 spacing."""
    nz, ny, nx = vol.shape
    hdr = bytearray(352)
    hdr[0:4] = np.array([348], np.int32).tobytes()
    hdr[40:56] = np.array([3, nx, ny, nz, 1, 1, 1, 1], np.int16).tobytes()
    hdr[70:72] = np.array([datatype], np.int16).tobytes()
    hdr[72:74] = np.array([32], np.int16).tobytes()
    hdr[76:108] = np.array([0, 1, 1, 1, 1, 1, 1, 1], np.float32).tobytes()
    hdr[108:112] = np.array([352.0], np.float32).tobytes()
    hdr[148:228] = b"nyxus 3d gldzm compatibility phantom".ljust(80, b"\0")
    hdr[254:256] = np.array([1], np.int16).tobytes()               # sform_code
    hdr[280:296] = np.array([1, 0, 0, 0], np.float32).tobytes()    # srow_x
    hdr[296:312] = np.array([0, 1, 0, 0], np.float32).tobytes()    # srow_y
    hdr[312:328] = np.array([0, 0, 1, 0], np.float32).tobytes()    # srow_z
    hdr[344:348] = b"n+1\0"
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(vol.tobytes())


def check_one_phantom(name_of_builder, pairs, write):
    """Writes a phantom pair, or checks the checked-in files still hold what its rule produces."""
    bad = 0
    for path, vol, datatype in pairs:
        name = os.path.basename(path)
        if write:
            write_nifti(path, vol, datatype)
            print("# wrote " + path)
            continue
        if not os.path.exists(path):
            print("  FAIL %s is missing; rerun with --write-phantom" % name)
            bad += 1
            continue
        got = read_nifti(path)[0]
        if got.shape != vol.shape or not np.array_equal(got, vol.astype(got.dtype)):
            print("  FAIL %s does not hold what %s produces" % (name, name_of_builder))
            bad += 1
        else:
            print("  OK   %s is what %s produces" % (name, name_of_builder))
    return bad


def check_phantoms(write):
    """Both fixtures this family owns: the MIRP compatibility phantom and the zero-level one."""
    inten, mask = build_compat_phantom()
    roi = mask == LABEL
    print("# compat phantom %s, roi %d voxels, grey levels %s"
          % (inten.shape, int(roi.sum()), sorted(int(v) for v in np.unique(inten[roi]))))
    bad = check_one_phantom("build_compat_phantom()",
                            [(COMPAT_INTEN, inten, 16), (COMPAT_MASK, mask, 768)], write)

    zi, zm = build_zero_level_phantom()
    zroi = zm == LABEL
    print("# zero-level phantom %s, roi %d voxels, raw levels %s"
          % (zi.shape, int(zroi.sum()), sorted(int(v) for v in np.unique(zi[zroi]))))
    bad += check_one_phantom("build_zero_level_phantom()",
                             [(ZERO_INTEN, zi, 16), (ZERO_MASK, zm, 768)], write)
    return inten, mask, bad


# --- the grey levels Nyxus reaches on the segmented phantom --------------------------------------

def nyxus_grey_levels(inten, mask, roi_max):
    """-> the volume of grey levels Nyxus' GREYDEPTH=64 binning produces, off-ROI voxels at 0.

    Two steps, both Nyxus-side and both visible in its output: the loader shifts every voxel by the
    volume minimum, and the family's MATLAB-style binning maps i to floor(64 * i / ROI max) + 1,
    clipped into [1, 64]. Handing the result to MIRP with the discretisation switched off is what
    makes the samelevels run comparable.
    """
    # widen before subtracting: NIfTI datatype 4 is int16, and a volume spanning more than 32767
    # would wrap in the file's own dtype and stop reproducing Nyxus' binning
    shifted = inten.astype(np.int64) - int(inten.min())
    levels = np.floor(NBINS * shifted.astype(np.float64) / float(roi_max)) + 1.0
    levels = np.clip(levels, 1.0, float(NBINS))
    return np.where(mask, levels, 0.0)


# --- verification --------------------------------------------------------------------------------

def parse_pins(path, table):
    """-> {feature: pinned value} from a ref_vals_map in a test header.

    Counts braces rather than matching a non-greedy body, which would swallow the last entry's
    closing brace and silently drop it.
    """
    txt = open(path, encoding="utf-8", errors="replace").read()
    at = txt.index(table)
    at = txt.index("{", at)
    depth, end = 0, None
    for i in range(at, len(txt)):
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise RuntimeError("%s is not brace-balanced" % table)
    rows = re.findall(r'\{\s*"(3GLDZM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', txt[at:end + 1])
    if not rows:
        raise RuntimeError("no pins found in %s" % table)
    return dict((n, float(v)) for n, v in rows)


def parse_report(txt, heading):
    """-> {feature: (nyxus, mirp, third_as_written)} from the comparison table under `heading`.

    Rows look like | `3GLDZM_GLNU` | 1349.4 | 1433.45 | 0.94x |. EVERY column comes back, because
    every one of them is a numeric claim the report publishes: the MIRP column is this run's to
    verify, the Nyxus column has a source in the tree, and the third is derived from the two. A
    parser that kept only the MIRP column would let the other two be corrupted and still pass.

    The report carries one such table per run, so the section is located first and the search stops
    at the next heading of the same level.
    """
    m = re.search("^" + re.escape(heading) + r"\s*$", txt, re.M)
    if not m:
        raise RuntimeError("no %r section in %s" % (heading, os.path.basename(REPORT)))
    rest = txt[m.end():]
    nxt = re.search(r"^##\s", rest, re.M)
    section = rest[: nxt.start()] if nxt else rest
    rows = re.findall(
        r"^\|\s*`(3GLDZM_[A-Z0-9_]+)`\s*\|\s*([-0-9.eE+]+)\s*\|\s*([-0-9.eE+]+)\s*\|"
        r"\s*([-0-9.eE+x]+)\s*\|",
        section, re.M)
    if not rows:
        raise RuntimeError("no comparison rows under %r in %s"
                           % (heading, os.path.basename(REPORT)))
    return dict((n, (float(a), float(b), c.strip())) for n, a, b, c in rows)


def rel_of(a, b):
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0 else abs(a - b) / denom


def verify_table(quoted, mirp_run, nyxus_source, nyxus_tol, third_kind, what):
    """-> (n verified, n failed, n unproducible, [unquoted]) for one published comparison table.

    Three claims per row, each checked against its own source and each able to fail on its own:

      the MIRP column    against this run of MIRP, at REPORT_RELTOL
      the Nyxus column   against `nyxus_source` -- the regression pins for the segmented-phantom
                         tables, the header's MIRP pins for the vetted cell, where what the tree
                         actually asserts is agreement inside the oracle band
      the third column   recomputed from the row's own two values and re-rendered with the format
                         the report writes, so a corrupted ratio or residual fails as a string
    """
    print("\n# verifying the %d rows published in %s -- all three columns" % (len(quoted), what))
    nok = nfail = nmiss = 0
    for name in sorted(quoted):
        nyxus, mirp, third = quoted[name]

        if name not in mirp_run:
            print("  MISSING %s: %s quotes MIRP %r but MIRP reports no counterpart"
                  % (name, what, mirp))
            nmiss += 1
            continue

        bad = []
        rel_mirp = rel_of(mirp_run[name], mirp)
        if rel_mirp > REPORT_RELTOL:
            bad.append("mirp column %r against this run's %r (rel %.3g)"
                       % (mirp, mirp_run[name], rel_mirp))

        if name not in nyxus_source:
            bad.append("nyxus column %r has no source to check against" % nyxus)
        else:
            want = nyxus_source[name]
            off = abs(nyxus - want) if nyxus_tol[0] == "abs" else rel_of(nyxus, want)
            if off > nyxus_tol[1]:
                bad.append("nyxus column %r against %r (%s %.3g)"
                           % (nyxus, want, nyxus_tol[0], off))

        recomputed = ("%.1e" % rel_of(nyxus, mirp) if third_kind == "rel"
                      else "%.3g" % (nyxus / mirp))
        if recomputed != third:
            bad.append("third column %r, recomputed from this row's own values as %r"
                       % (third, recomputed))

        if bad:
            print("  FAIL %s: %s" % (name, "; ".join(bad)))
            nfail += 1
        else:
            print("  OK   %s: nyxus=%r mirp=%r %s=%s" % (name, nyxus, mirp, third_kind, third))
            nok += 1

    unquoted = sorted(set(mirp_run) - set(quoted))
    for name in unquoted:
        print("  UNQUOTED %s: MIRP reports %r and %s does not quote it"
              % (name, mirp_run[name], what))
    return nok, nfail, nmiss, unquoted


def verify(got, quoted, what, tol):
    """-> (n verified, n failed, n unproducible, [unquoted]) for one table of quoted values."""
    print("\n# verifying the %d MIRP values quoted in %s against this run, at rel<=%g"
          % (len(quoted), what, tol))
    nok = nfail = nmiss = 0
    for name in sorted(quoted):
        want = quoted[name]
        if name not in got:
            print("  MISSING %s: %s quotes %r but MIRP reports no counterpart" % (name, what, want))
            nmiss += 1
            continue
        have = got[name]
        rel = abs(have - want) / max(abs(want), 1e-12)
        if rel <= tol:
            print("  OK   %s: mirp=%r quoted=%r rel=%.3g" % (name, have, want, rel))
            nok += 1
        else:
            print("  FAIL %s: mirp=%r quoted=%r rel=%.3g" % (name, have, want, rel))
            nfail += 1

    # the reverse direction: a feature this run produces that nothing quotes
    unquoted = sorted(set(got) - set(quoted))
    for name in unquoted:
        print("  UNQUOTED %s: MIRP reports %r and %s does not quote it" % (name, got[name], what))
    return nok, nfail, nmiss, unquoted


def main():
    write = "--write-phantom" in sys.argv[1:]
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print("missing phantom: " + p)
            return 1

    compat_inten, compat_mask, bad_phantom = check_phantoms(write)
    if write:
        return 0

    import mirp
    logging.disable(logging.INFO)

    def features(image, mask, spacing, suffix, **discretisation):
        res = mirp.extract_features(
            image=image, mask=mask, image_spacing=spacing,
            by_slice=False, base_feature_families="gldzm", **discretisation)
        df = res[0] if isinstance(res, list) else res
        row = df.iloc[0]
        out = {}
        for nyx, stem in MIRP.items():
            col = stem + suffix
            if col not in df.columns:
                raise RuntimeError("MIRP produced no %s (for %s); dzm columns present: %s"
                                   % (col, nyx,
                                      sorted(c for c in df.columns if c.startswith("dzm_"))))
            out[nyx] = float(row[col])
        return out

    compat = features(compat_inten.astype(np.float64),
                      (compat_mask == LABEL).astype(np.int32),
                      (1.0, 1.0, 1.0), SUFFIX_NONE, base_discretisation_method="none")

    inten, spacing = read_nifti(INTEN)
    roi = read_nifti(MASK)[0] == LABEL
    m32 = roi.astype(np.int32)
    shifted_roi_max = int((inten.astype(np.int64) - int(inten.min()))[roi].max())
    print("# segmented phantom %s, roi %d voxels, spacing zyx %s"
          % (inten.shape, int(roi.sum()), spacing))

    levels = nyxus_grey_levels(inten, roi, shifted_roi_max)
    lv = levels[roi]
    span = (int(lv.min()), int(lv.max()), int(np.unique(lv).size))
    print("# nyxus grey levels over the roi: %d-%d, %d distinct; mirp fixed_bin_number n=%d "
          "spreads the same roi over 1-%d" % (span[0], span[1], span[2], NBINS, NBINS))

    fbn = features(inten.astype(np.float64), m32, spacing, SUFFIX_FBN,
                   base_discretisation_method="fixed_bin_number",
                   base_discretisation_n_bins=NBINS)
    same = features(levels, m32, spacing, SUFFIX_NONE, base_discretisation_method="none")

    try:
        version = metadata.version("mirp")       # mirp exposes no __version__
    except metadata.PackageNotFoundError:
        version = "unknown"

    # the installed mirp, read from the distribution rather than written in: this line is the
    # provenance of the values printed below it, so a run under another version has to say so.
    print("# mirp %s, numpy %s, label=%d, by_slice=False" % (version, np.__version__, LABEL))
    for title, got, suffix in (("gldzm3d.mirp_compat_phantom", compat, SUFFIX_NONE),
                               ("gldzm3d.mirp_fbn64", fbn, SUFFIX_FBN),
                               ("gldzm3d.mirp_samelevels", same, SUFFIX_NONE)):
        print("\n# " + title)
        for name in sorted(got):
            print(('\t{"%s", %r},' % (name, got[name])).ljust(56)
                  + "// " + MIRP[name] + suffix)
    print("\n# no MIRP counterpart at any recipe: 3GLDZM_GLM, 3GLDZM_ZDM")

    nok = nmiss = 0
    nfail = bad_phantom
    unquoted = []

    oracle_pins = parse_pins(HEADER, "gldzm_3d_mirp_ref_vals")
    regression_pins = parse_pins(REGRESSION_HEADER, "gldzm_3d_regression_ref_vals")

    a, b, c, d = verify(compat, oracle_pins, "test_3d_gldzm_mirp.h", PIN_RELTOL)
    nok += a
    nfail += b
    nmiss += c
    unquoted += d

    if not os.path.exists(REPORT):
        print("\n# %s is missing -- there is nothing the published tables can be checked against, "
              "which is a failure, not a pass" % os.path.basename(REPORT))
        return 1
    txt = open(REPORT, encoding="utf-8", errors="replace").read()

    # heading -> (this run's MIRP values, what the report's Nyxus column answers to, its tolerance,
    #             what the third column holds). The vetted cell's Nyxus column answers to the oracle
    #             pins within the band the assertions use, because agreement inside that band is the
    #             claim; the other two answer to the regression pins exactly, being the same run.
    tables = (
        ("## Result at `gldzm3d.mirp_compat_phantom` -- the vetted cell",
         compat, oracle_pins, ("abs", 1e-9), "rel"),
        ("## Result at `gldzm3d.mirp_fbn64` -- MIRP discretises",
         fbn, regression_pins, ("rel", 1e-12), "ratio"),
        ("## Result at `gldzm3d.mirp_samelevels` -- the same grey levels",
         same, regression_pins, ("rel", 1e-12), "rel"),
    )
    for heading, mirp_run, nyxus_source, nyxus_tol, third_kind in tables:
        a, b, c, d = verify_table(parse_report(txt, heading), mirp_run, nyxus_source, nyxus_tol,
                                  third_kind, heading.split("`")[1])
        nok += a
        nfail += b
        nmiss += c
        unquoted += d

    # the grey-level span is the report's explanation of the fbn64 gap, so it is checked too
    span_txt = "%d-%d, %d distinct" % span
    if span_txt not in txt:
        print("\nFAIL grey-level span: this run measures %s, which the report does not state"
              % span_txt)
        nfail += 1
    else:
        print("\nOK   grey-level span: report states " + span_txt)

    print("\n%d verified, %d failed, %d unproducible, %d unquoted"
          % (nok, nfail, nmiss, len(unquoted)))
    if nfail or nmiss or unquoted:
        print("SOME CHECKS FAILED -- the tree is out of step with the tool")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
