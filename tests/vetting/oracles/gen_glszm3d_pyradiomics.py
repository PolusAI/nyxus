"""OFFLINE PyRadiomics oracle for the 3D GLSZM features, on the compat phantom.

    python tests/vetting/oracles/gen_glszm3d_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_glszm_pyradiomics.h --
the sixteen scalars, the size-zone matrix of the phantom, and the size-zone matrix of the small
hand-worked volume -- exiting non-zero on any mismatch, on any pin it cannot produce, on a bound
violation, or on a cross-table disagreement.

TWO RECIPES. `glszm3d.pyradiomics_bincount20` is the phantom run below. `glszm3d.pyradiomics_ibsi_gapped`
is the second: a 3x3x3 literal carrying grey levels 1, 3 and 5, read at binWidth=1, which is the only
place either side is asked what an absent grey level does to the row index. Nyxus reaches that point
through IBSI=true, where calculate() forces its binning to 0; PyRadiomics reaches it through a bin
width that leaves the levels alone. Both report Ng = 5 for three occupied levels.

Recipe `glszm3d.pyradiomics_bincount20`: the compat phantom
(tests/data/nifti/compat_int/compat_int_mri.nii + compat_seg/compat_seg_liver.nii, label 1) with
binCount=20, no resampling, weightingNorm=None, imageType=Original -- the settings the test header's
provenance block records. On the Nyxus side that is GREYDEPTH=100, IBSI=false, GLSZM_GREYDEPTH=-20
(negative activates radiomics binCount-based binning, so the magnitude is the bin count).

ONE REFERENCE, NOT TWO. Unlike GLCM and GLRLM this family has no direction set: a size-zone matrix
counts connected components, and a component has no orientation. PyRadiomics' public API therefore
reports exactly the quantity Nyxus stores, and there is no per-angle table to intercept.

THE MATRIX IS PINNED, NOT ONLY THE SCALARS. All sixteen features are contractions of one P(i, j)
table, so a compensating pair of errors inside it survives every scalar assertion. PyRadiomics builds
P_glszm before any feature formula runs, so intercepting it reimplements nothing. `recompute()` then
closes the loop in the other direction: it evaluates all sixteen IBSI definitions from that same
matrix and compares them against the sixteen values the public extractor reported, which is what
makes it impossible to edit one table here and not the other.

CONNECTIVITY: both sides walk the full 26-voxel neighbourhood in 3D. PyRadiomics' C extension uses
it by construction; Nyxus' D3_GLSZM_feature::gather_size_zones spells out the 8 in-slice, 8 upper, 8
lower and 2 strictly-vertical offsets. The 4x4x3 hand-worked volume below is the direct check of
that, and it is built so that only the 26-neighbourhood produces its table: its zones include a
strictly vertical run, a z-edge join, a z-corner join that no 18-neighbourhood would make, and a pair
of same-level voxels two slices apart that must stay two zones. Counting them under 18-, 6- and
2D-8-connectivity gives 10, 13 and 13 zones against 26-connectivity's 9. Its predecessor, one
populated slice between two empty ones, gave the same nine zones under 26-, 18- AND 2D-8-connectivity.

Provenance: tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs
Python <= 3.9); generator=tests/vetting/oracles/gen_glszm3d_pyradiomics.py. Run offline; CI never
invokes it.
"""
import math
import os
import re

import numpy
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, glszm, imageoperations

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
INTEN = os.path.join(DATA, "compat_int", "compat_int_mri.nii")
MASK = os.path.join(DATA, "compat_seg", "compat_seg_liver.nii")
TEST_H = os.path.join(TESTS, "test_3d_glszm_pyradiomics.h")

RELTOL = 1e-9          # matches the C++ assertions (agrees_gt frac_tolerance = 1e9)
ZE_RELTOL = 1e-3       # the band the C++ assertions give 3GLSZM_ZE alone (Nyxus' fast_log10)
LABEL = 1
BINCOUNT = 20

# Nyxus feature -> PyRadiomics GLSZM feature. All sixteen map one-to-one, so this table is complete:
# the family has no identity-vetted leftovers the way 3D GLCM does.
PYRAD = {
    "3GLSZM_GLN": "GrayLevelNonUniformity",
    "3GLSZM_GLNN": "GrayLevelNonUniformityNormalized",
    "3GLSZM_GLV": "GrayLevelVariance",
    "3GLSZM_HGLZE": "HighGrayLevelZoneEmphasis",
    "3GLSZM_LAE": "LargeAreaEmphasis",
    "3GLSZM_LAHGLE": "LargeAreaHighGrayLevelEmphasis",
    "3GLSZM_LALGLE": "LargeAreaLowGrayLevelEmphasis",
    "3GLSZM_LGLZE": "LowGrayLevelZoneEmphasis",
    "3GLSZM_SAE": "SmallAreaEmphasis",
    "3GLSZM_SAHGLE": "SmallAreaHighGrayLevelEmphasis",
    "3GLSZM_SALGLE": "SmallAreaLowGrayLevelEmphasis",
    "3GLSZM_SZN": "SizeZoneNonUniformity",
    "3GLSZM_SZNN": "SizeZoneNonUniformityNormalized",
    "3GLSZM_ZE": "ZoneEntropy",
    "3GLSZM_ZP": "ZonePercentage",
    "3GLSZM_ZV": "ZoneVariance",
}

# Mathematical bounds the family's features carry by construction. Checked against the oracle's own
# output, so a misconfigured run is caught before anything is pinned.
BOUNDS = {
    "3GLSZM_SAE": (0.0, 1.0),
    "3GLSZM_LAE": (1.0, None),
    "3GLSZM_ZP": (0.0, 1.0),
    "3GLSZM_GLNN": (0.0, 1.0),
    "3GLSZM_SZNN": (0.0, 1.0),
    "3GLSZM_ZE": (0.0, None),
    "3GLSZM_GLV": (0.0, None),
    "3GLSZM_ZV": (0.0, None),
}

# The 4x4x3 volume the header's small matrix assertion runs on, written as PyRadiomics wants it,
# (z, y, x). Every slice is populated and the zones cross them; the header lists the nine zones and
# what each one is there to separate.
DOC_VOLUME = numpy.array([
    [[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 4]],
    [[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 1, 0], [0, 3, 0, 1]],
    [[1, 0, 4, 0], [3, 3, 3, 4], [0, 0, 0, 0], [2, 0, 0, 4]],
], dtype=numpy.int16)

# The 3x3x3 volume behind recipe `glszm3d.pyradiomics_ibsi_gapped`, (z, y, x). Levels 1, 3 and 5 with
# 2 and 4 absent, so the position of a level in the occupied-level list and the level itself are
# different numbers -- which is the only thing that separates the IBSI row-index branch from the one
# beside it.
GAPPED_VOLUME = numpy.array([
    [[1, 0, 3], [0, 0, 0], [5, 0, 5]],
    [[1, 0, 0], [0, 3, 0], [0, 0, 0]],
    [[0, 0, 3], [0, 0, 0], [5, 0, 1]],
], dtype=numpy.int16)


def nonzero_cells(f):
    """-> [(grey level, zone size, count)] over a calculated RadiomicsGLSZM, row-major.

    A COLUMN INDEX IS NOT A ZONE SIZE. _calculateCoefficients() deletes every column whose zone size
    no zone in the ROI has, and carries the surviving sizes in 'jvector'; on the compat phantom that
    turns 634 columns into 46 whose sizes run 1..16, 18..29, 31, 32, 34, 36, 44, ..., 634. Reading
    the size off the index instead would relabel almost every large zone -- which is what the
    cross-table check in recompute() exists to catch, and did.

    Nyxus keeps the matrix dense (column index = zone size - 1, width = the largest zone size), so
    the two agree cell for cell once the size is taken from jvector.
    """
    P = f.P_glszm[0]
    sizes = f.coefficients["jvector"]
    levels = f.coefficients["ivector"]
    return [(int(levels[i]), int(sizes[j]), int(P[i, j]))
            for i, j in numpy.argwhere(P > 0)]


def phantom_matrix():
    """-> (cells, Ng, Ns, Nz, Np) of the compat phantom's size-zone matrix.

    Ns is the largest zone size, which is the width Nyxus allocates -- not the count of distinct
    sizes PyRadiomics is left holding.
    """
    img = sitk.ReadImage(INTEN)
    msk = sitk.ReadImage(MASK)
    bb, _ = imageoperations.checkMask(img, msk, label=LABEL)
    img, msk = imageoperations.cropToTumorMask(img, msk, bb)

    f = glszm.RadiomicsGLSZM(img, msk, binCount=BINCOUNT, label=LABEL, weightingNorm=None,
                             interpolator=sitk.sitkBSpline, resampledPixelSpacing=None,
                             force2D=False)
    f._initCalculation()
    cells = nonzero_cells(f)
    return (cells,
            int(numpy.asarray(f.coefficients["Ng"]).ravel()[0]),
            max(c[1] for c in cells),
            int(f.P_glszm.sum()),
            int(f.coefficients["Np"][0]))


def doc_matrix():
    """-> (cells, Ng, Ns, Np) of DOC_VOLUME's size-zone matrix at binWidth=1.

    binWidth=1 on levels 1..4 discretises to 1..4, which is what Nyxus' no-binning path reads
    straight off the volume, so the two sides index the same rows.
    """
    img = sitk.GetImageFromArray(DOC_VOLUME)
    msk = sitk.GetImageFromArray((DOC_VOLUME > 0).astype(numpy.uint8))
    f = glszm.RadiomicsGLSZM(img, msk, binWidth=1, label=1, weightingNorm=None,
                             resampledPixelSpacing=None, force2D=False)
    f._initCalculation()
    cells = nonzero_cells(f)
    return (cells,
            int(numpy.asarray(f.coefficients["Ng"]).ravel()[0]),
            max(c[1] for c in cells),
            int(f.coefficients["Np"][0]))


def gapped_matrix():
    """-> (cells, Ng, Ns, Np) of GAPPED_VOLUME's size-zone matrix at binWidth=1."""
    img = sitk.GetImageFromArray(GAPPED_VOLUME)
    msk = sitk.GetImageFromArray((GAPPED_VOLUME > 0).astype(numpy.uint8))
    f = glszm.RadiomicsGLSZM(img, msk, binWidth=1, label=1, weightingNorm=None,
                             resampledPixelSpacing=None, force2D=False)
    f._initCalculation()
    cells = nonzero_cells(f)
    return (cells,
            int(numpy.asarray(f.coefficients["Ng"]).ravel()[0]),
            max(c[1] for c in cells),
            int(f.coefficients["Np"][0]))


def gapped_run():
    """-> {nyxus feature: PyRadiomics value} on GAPPED_VOLUME, through the public extractor."""
    radiomics.logger.setLevel(40)
    ex = featureextractor.RadiomicsFeatureExtractor(
        binWidth=1, label=1, weightingNorm=None, resampledPixelSpacing=None, force2D=False)
    ex.disableAllFeatures()
    ex.enableFeatureClassByName("glszm")
    img = sitk.GetImageFromArray(GAPPED_VOLUME)
    msk = sitk.GetImageFromArray((GAPPED_VOLUME > 0).astype(numpy.uint8))
    res = ex.execute(img, msk)
    return {nyx: float(res[f"original_glszm_{pyr}"]) for nyx, pyr in PYRAD.items()}


def recompute(cells, Ng, Ns, Nz, Np):
    """-> the sixteen features evaluated from the size-zone matrix alone.

    The IBSI definitions, written out over the pinned cells. Nothing here reads PyRadiomics, so
    comparing this against run() is a cross-table check in the sense of revet.txt section 9: the
    matrix and the scalars can only agree if they describe the same run.
    """
    si = [0.0] * (Ng + 1)                    # per grey level
    sj = [0.0] * (Ns + 1)                    # per zone size
    for i, j, c in cells:
        si[i] += c
        sj[j] += c

    tot = float(Nz)
    mu_i = sum(c * i for i, _, c in cells) / tot
    mu_j = sum(c * j for _, j, c in cells) / tot
    eps = numpy.spacing(1)
    return {
        "3GLSZM_SAE": sum(c / (j * j) for _, j, c in cells) / tot,
        "3GLSZM_LAE": sum(c * (j * j) for _, j, c in cells) / tot,
        "3GLSZM_GLN": sum(v * v for v in si) / tot,
        "3GLSZM_GLNN": sum(v * v for v in si) / (tot * tot),
        "3GLSZM_SZN": sum(v * v for v in sj) / tot,
        "3GLSZM_SZNN": sum(v * v for v in sj) / (tot * tot),
        "3GLSZM_ZP": tot / float(Np),
        "3GLSZM_GLV": sum(c / tot * (i - mu_i) ** 2 for i, _, c in cells),
        "3GLSZM_ZV": sum(c / tot * (j - mu_j) ** 2 for _, j, c in cells),
        "3GLSZM_ZE": -sum(c / tot * math.log(c / tot + eps, 2) for _, _, c in cells),
        "3GLSZM_LGLZE": sum(c / (i * i) for i, _, c in cells) / tot,
        "3GLSZM_HGLZE": sum(c * (i * i) for i, _, c in cells) / tot,
        "3GLSZM_SALGLE": sum(c / (i * i * j * j) for i, j, c in cells) / tot,
        "3GLSZM_SAHGLE": sum(c * (i * i) / (j * j) for i, j, c in cells) / tot,
        "3GLSZM_LALGLE": sum(c * (j * j) / (i * i) for i, j, c in cells) / tot,
        "3GLSZM_LAHGLE": sum(c * (i * i) * (j * j) for i, j, c in cells) / tot,
    }


def run():
    """-> {nyxus feature: PyRadiomics value} through the public extractor."""
    radiomics.logger.setLevel(40)             # keep progress chatter out of the golden table
    settings = {
        "binCount": BINCOUNT,
        "label": LABEL,
        "interpolator": sitk.sitkBSpline,
        "resampledPixelSpacing": None,
        "weightingNorm": None,
        "force2D": False,
    }
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.disableAllFeatures()
    ex.enableFeatureClassByName("glszm")
    res = ex.execute(INTEN, MASK)
    out = {}
    for nyx, pyr in PYRAD.items():
        key = f"original_glszm_{pyr}"
        if key not in res:
            raise RuntimeError(f"PyRadiomics produced no {key} (for {nyx})")
        out[nyx] = float(res[key])
    return out


def _table_body(txt, table):
    """-> the text between the table's opening brace and its matching close.

    Counts braces rather than stopping at the first '};'. A non-greedy regex swallows the last
    entry's closing brace when the entries are themselves braced, which is how a header parser can
    silently drop a pin and still report success.
    """
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found in {os.path.basename(TEST_H)}")
    depth, i = 1, m.end()
    while depth:
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
        i += 1
    return re.sub(r"//[^\n]*", "", txt[m.end():i - 1])     # a commented-out golden is not a pin


def parse_pins(txt, table):
    body = _table_body(txt, table)
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"(3GLSZM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def parse_cells(txt, table):
    body = _table_body(txt, table)
    return [(int(a), int(b), int(c)) for a, b, c in
            re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]


def parse_int(txt, name):
    m = re.search(re.escape(name) + r"\s*=\s*(\d+)", txt)
    if not m:
        raise RuntimeError(f"constant {name} not found in {os.path.basename(TEST_H)}")
    return int(m.group(1))


def compare(label, got, want, reltol):
    rel = abs(got - want) / max(abs(want), 1e-12)
    if rel <= reltol:
        return 0
    print(f"  FAIL {label}: oracle={got!r} pinned={want!r} rel={rel:.3g} (band {reltol:g})")
    return 1


def check_bounds(got):
    bad = 0
    for name, (lo, hi) in sorted(BOUNDS.items()):
        v = got[name]
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print(f"  OUT OF RANGE {name} = {v!r}, expected [{lo}, {hi}]")
            bad += 1
    print(f"# {len(BOUNDS) - bad}/{len(BOUNDS)} bounded features in range")
    return bad


def check_identities(got, Nz):
    """The relations the family carries independently of any oracle."""
    bad = 0
    bad += compare("identity GLNN == GLN / Nz", got["3GLSZM_GLN"] / Nz, got["3GLSZM_GLNN"], 1e-12)
    bad += compare("identity SZNN == SZN / Nz", got["3GLSZM_SZN"] / Nz, got["3GLSZM_SZNN"], 1e-12)
    if not got["3GLSZM_SAE"] <= got["3GLSZM_LAE"]:
        print("  FAIL identity SAE <= LAE")
        bad += 1
    if not got["3GLSZM_LGLZE"] <= got["3GLSZM_HGLZE"]:
        print("  FAIL identity LGLZE <= HGLZE")
        bad += 1
    print(f"  {4 - bad}/4 identities hold")
    return bad


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1

    got = run()
    cells, Ng, Ns, Nz, Np = phantom_matrix()
    doc_cells, doc_Ng, doc_Ns, doc_Np = doc_matrix()
    gap = gapped_run()
    gap_cells, gap_Ng, gap_Ns, gap_Np = gapped_matrix()

    print(f"# pyradiomics {radiomics.__version__}, SimpleITK {sitk.__version__}, "
          f"binCount={BINCOUNT}, label={LABEL}")
    print(f"# size-zone matrix: Ng={Ng} Ns={Ns} Nz={Nz} Np={Np}, {len(cells)} non-empty cells")
    print("# paste-ready goldens")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// original_glszm_{PYRAD[name]}")

    print("\n# paste-ready non-empty cells of the phantom's size-zone matrix, {level, size, count}")
    for lev, size, cnt in cells:
        print(f"\t{{ {lev}, {size}, {cnt} }},")

    print(f"\n# paste-ready non-empty cells of the 4x4x3 volume's matrix "
          f"(Ng={doc_Ng}, Ns={doc_Ns}), {{level, size, count}}")
    for lev, size, cnt in doc_cells:
        print(f"\t{{ {lev}, {size}, {cnt} }},")

    print("\n# paste-ready goldens of the gapped-level volume (recipe glszm3d.pyradiomics_ibsi_gapped)")
    for name in sorted(gap):
        print(f'\t{{"{name}", {gap[name]!r}}},'.ljust(56) + f"// original_glszm_{PYRAD[name]}")
    print(f"\n# paste-ready non-empty cells of the gapped volume's matrix "
          f"(Ng={gap_Ng}, Ns={gap_Ns}, Np={gap_Np}), {{level, size, count}}")
    for lev, size, cnt in gap_cells:
        print(f"\t{{ {lev}, {size}, {cnt} }},")

    print()
    nbad = check_bounds(got)
    print("# identities")
    nbad += check_identities(got, Nz)

    print("\n# cross-table: the sixteen features recomputed from the matrix alone")
    ncross = 0
    again = recompute(cells, Ng, Ns, Nz, Np)
    for name in sorted(got):
        # ZE is a sum over logarithms and the two paths accumulate it in different orders
        ncross += compare(f"cross-table {name}", again[name], got[name],
                          1e-9 if name != "3GLSZM_ZE" else 1e-12)
    print(f"  {len(got) - ncross}/{len(got)} features reproduce from the pinned matrix")

    txt_h = open(TEST_H, encoding="utf-8", errors="replace").read()

    nok = nfail = nmiss = 0
    missing_pin = []
    for table, oracle in (("glszm_3d_pyradiomics_ref_vals", got),
                          ("glszm_3d_pyradiomics_gapped_ref_vals", gap)):
        pins = parse_pins(txt_h, table)
        print(f"\n# verifying {len(pins)} pinned goldens of {table} against this run")
        for name in sorted(pins):
            want = pins[name]
            if name not in oracle:
                print(f"  MISSING {name}: pinned {want!r} but PyRadiomics reports no counterpart")
                nmiss += 1
                continue
            band = ZE_RELTOL if name == "3GLSZM_ZE" else RELTOL
            if compare(name, oracle[name], want, band):
                nfail += 1
            else:
                print(f"  OK   {name}: pyradiomics={oracle[name]!r} pinned={want!r}")
                nok += 1

        # the reverse check: something the oracle produces that the header pins nothing for
        for name in sorted(set(oracle) - set(pins)):
            print(f"  UNPINNED {name}: PyRadiomics reports {oracle[name]!r} and nothing pins it")
            missing_pin.append(f"{table}:{name}")

    nmat = 0
    for table, want_cells, want_dims in (
            ("glszm_3d_pyradiomics_matrix_ref_vals", cells,
             (("glszm_3d_pyradiomics_matrix_ng", Ng), ("glszm_3d_pyradiomics_matrix_ns", Ns),
              ("glszm_3d_pyradiomics_matrix_nz", Nz), ("glszm_3d_pyradiomics_matrix_np", Np))),
            ("glszm_3d_pyradiomics_smallmatrix_ref_vals", doc_cells,
             (("glszm_3d_pyradiomics_smallmatrix_ng", doc_Ng),
              ("glszm_3d_pyradiomics_smallmatrix_ns", doc_Ns),
              ("glszm_3d_pyradiomics_smallmatrix_nz",
               sum(c[2] for c in doc_cells)),
              ("glszm_3d_pyradiomics_smallmatrix_np", doc_Np))),
            ("glszm_3d_pyradiomics_gappedmatrix_ref_vals", gap_cells,
             (("glszm_3d_pyradiomics_gappedmatrix_ng", gap_Ng),
              ("glszm_3d_pyradiomics_gappedmatrix_ns", gap_Ns),
              ("glszm_3d_pyradiomics_gappedmatrix_nz",
               sum(c[2] for c in gap_cells)),
              ("glszm_3d_pyradiomics_gappedmatrix_np", gap_Np)))):
        pinned_cells = parse_cells(txt_h, table)
        print(f"\n# verifying {len(pinned_cells)} pinned cells of {table} against this run")
        if pinned_cells != want_cells:
            for k in range(max(len(pinned_cells), len(want_cells))):
                p = pinned_cells[k] if k < len(pinned_cells) else None
                w = want_cells[k] if k < len(want_cells) else None
                if p != w:
                    print(f"  FAIL cell {k}: pinned={p} pyradiomics={w}")
                    nmat += 1
        else:
            print(f"  all {len(want_cells)} cells reproduce")
        for cname, cval in want_dims:
            pinned = parse_int(txt_h, cname)
            if pinned != cval:
                print(f"  FAIL {cname}: pinned={pinned} pyradiomics={cval}")
                nmat += 1
            else:
                print(f"  OK   {cname} = {cval}")

    print(f"\n{nok} verified, {nfail} failed, {nmiss} unproducible, {len(missing_pin)} unpinned, "
          f"{nbad} out of range or identity-violating, {ncross} cross-table mismatch(es), "
          f"{nmat} matrix mismatch(es)")
    if nfail or nmiss or nbad or ncross or nmat or missing_pin:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
