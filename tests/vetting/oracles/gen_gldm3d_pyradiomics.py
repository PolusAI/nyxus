"""OFFLINE PyRadiomics oracle for the 3D GLDM features, on the compat phantom.

    python tests/vetting/oracles/gen_gldm3d_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_gldm_pyradiomics.h --
the fourteen scalars, the dependence matrix of the phantom, and the dependence matrix of the small
hand-worked volume -- exiting non-zero on any mismatch, on any pin it cannot produce, on a bound
violation, or on a cross-table disagreement.

Recipe `gldm3d.pyradiomics_bincount20`: the compat phantom
(tests/data/nifti/compat_int/compat_int_mri.nii + compat_seg/compat_seg_liver.nii, label 1) with
binCount=20, no resampling, weightingNorm=None, imageType=Original -- the settings the test header's
provenance block records. On the Nyxus side that is GREYDEPTH=100, IBSI=false, GLDM_GREYDEPTH=-20
(negative activates radiomics binCount-based binning, so the magnitude is the bin count).

ONE REFERENCE, NOT TWO. This family has no direction set: a dependence count sums over the whole
26-voxel neighbourhood at once, so there is no per-angle table to intercept and PyRadiomics' public
API reports exactly the quantity Nyxus stores.

THE MATRIX IS PINNED, NOT ONLY THE SCALARS. All fourteen features are contractions of one P(i, j)
table, so a compensating pair of errors inside it survives every scalar assertion. PyRadiomics builds
P_gldm before any feature formula runs, so intercepting it reimplements nothing. `recompute()` then
closes the loop in the other direction: it evaluates all fourteen definitions from that same matrix
and compares them against the fourteen values the public extractor reported, which is what makes it
impossible to edit one table here and not the other.

THE NEIGHBOURHOOD RULE IS RE-DERIVED, NOT ASSERTED IN PROSE. `independent_matrix()` rebuilds the
dependence matrix straight from the definition -- 26 offsets at Chebyshev distance 1, a neighbour
counts when it is inside the mask and equal to the centre, and the count starts at 1 -- and the run
requires it to reproduce PyRadiomics' `P_gldm` cell for cell on both volumes. That is revet.txt step
3's "run both and compare all three" applied to the matrix rather than only to the scalars: the tool's
own C extension, the definition written out, and the fourteen published scalars all have to describe
one matrix. It is also what turns the two facts this family rests on into checks instead of comments
-- that both sides walk all 26 neighbours (PyRadiomics through `distances=[1]`, Nyxus through
`shifts[]`), and that dependence is offset by one on both sides (PyRadiomics puts a voxel with no
dependent neighbour in column j=1, Nyxus starts `nd = 1`), so the two index the same column.

Nz == Np IS AN ASSERTION, NOT A NOTE. PyRadiomics allows incomplete zones, so every ROI voxel owns
exactly one dependence zone and Nz equals the voxel count by construction. Nyxus reaches the same
number a different way -- it skips background-valued voxels of the ROI cube and counts the rest -- so
the two agreeing is evidence that Nyxus is not counting voxels outside the ROI into the matrix. Np is
counted off the mask on both volumes, never off the matrix, or the check would be a tautology.

THE MATRIX AND THE SCALARS COME FROM TWO DIFFERENT ENTRY POINTS. `run()` goes through the public
`RadiomicsFeatureExtractor`; `phantom_matrix()` reproduces its preprocessing (`checkMask` then
`cropToTumorMask`) to reach the class directly, because the extractor does not hand the matrix back.
Nothing guarantees those two stay in step -- the cross-table check is what does, by requiring the
fourteen scalars to fall out of the intercepted matrix.

Provenance of the run behind the checked-in vetting report: pyradiomics 3.0.1, SimpleITK 2.3.1,
numpy 1.23.5, Python 3.8, env=nyxus_oracle (conda, needs Python <= 3.9). That is what those goldens
were produced under, not a claim about the installed version -- the run prints whatever versions it
actually finds. generator=tests/vetting/oracles/gen_gldm3d_pyradiomics.py. Run offline; CI never
invokes it.
"""
import math
import os
import re

import numpy
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, gldm, imageoperations

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
INTEN = os.path.join(DATA, "compat_int", "compat_int_mri.nii")
MASK = os.path.join(DATA, "compat_seg", "compat_seg_liver.nii")
TEST_H = os.path.join(TESTS, "test_3d_gldm_pyradiomics.h")

# The band this script checks a PINNED DIGIT against a fresh run at, which is not the band the C++
# assertions compare Nyxus against PyRadiomics at. The pins here are PyRadiomics' own output printed
# at repr() precision, so they have to round-trip essentially exactly; anything looser would let a
# transcription slip through. The Nyxus-side band, including whatever 3GLDM_DE needs for fast_log10,
# is set from measurement in the test header and has nothing to do with this constant.
PIN_ROUNDTRIP_RELTOL = 1e-12
LABEL = 1
BINCOUNT = 20

# Nyxus feature -> PyRadiomics GLDM feature. All fourteen map one-to-one. PyRadiomics also exposes
# GrayLevelNonUniformityNormalized and DependencePercentage, both deprecated there because Nz == Np
# makes them degenerate; Nyxus implements neither, so nothing is left unmapped in either direction.
PYRAD = {
    "3GLDM_DE": "DependenceEntropy",
    "3GLDM_DN": "DependenceNonUniformity",
    "3GLDM_DNN": "DependenceNonUniformityNormalized",
    "3GLDM_DV": "DependenceVariance",
    "3GLDM_GLN": "GrayLevelNonUniformity",
    "3GLDM_GLV": "GrayLevelVariance",
    "3GLDM_HGLE": "HighGrayLevelEmphasis",
    "3GLDM_LDE": "LargeDependenceEmphasis",
    "3GLDM_LDHGLE": "LargeDependenceHighGrayLevelEmphasis",
    "3GLDM_LDLGLE": "LargeDependenceLowGrayLevelEmphasis",
    "3GLDM_LGLE": "LowGrayLevelEmphasis",
    "3GLDM_SDE": "SmallDependenceEmphasis",
    "3GLDM_SDHGLE": "SmallDependenceHighGrayLevelEmphasis",
    "3GLDM_SDLGLE": "SmallDependenceLowGrayLevelEmphasis",
}

# Mathematical bounds the family's features carry by construction. Checked against the oracle's own
# output, so a misconfigured run is caught before anything is pinned. Ten of the fourteen are a
# weighted mean of a quantity bounded on one side -- i and j are both >= 1, so a mean of 1/i^2,
# 1/j^2 or 1/(i^2 j^2) cannot exceed 1 and a mean of their reciprocals cannot fall below it. The
# Nz-dependent ceilings on GLN and DN are filled in by check_bounds() once Nz is known.
BOUNDS = {
    "3GLDM_SDE": (0.0, 1.0),
    "3GLDM_LDE": (1.0, None),
    "3GLDM_LGLE": (0.0, 1.0),
    "3GLDM_HGLE": (1.0, None),
    "3GLDM_SDLGLE": (0.0, 1.0),
    "3GLDM_LDHGLE": (1.0, None),
    "3GLDM_DNN": (0.0, 1.0),
    "3GLDM_DE": (0.0, None),
    "3GLDM_GLV": (0.0, None),
    "3GLDM_DV": (0.0, None),
}

# The 4x4x3 volume the header's small matrix assertion runs on: two identical populated slices with
# an all-zero one below them, written as PyRadiomics wants it, (z, y, x). Identical slices make the
# vertical coupling hand-checkable -- see the module docstring.
DOC_VOLUME = numpy.array([
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    [[1, 2, 3, 4], [1, 3, 4, 4], [3, 2, 2, 2], [4, 1, 4, 1]],
    [[1, 2, 3, 4], [1, 3, 4, 4], [3, 2, 2, 2], [4, 1, 4, 1]],
], dtype=numpy.int16)


def nonzero_cells(f):
    """-> [(grey level, dependence, count)] over a calculated RadiomicsGLDM, row-major.

    NEITHER INDEX IS ITS OWN VALUE. _calculateMatrix() deletes every row whose grey level is absent
    from the ROI and every column whose dependence no voxel has, carrying the survivors in 'ivector'
    and 'jvector'. Reading either off the index would relabel cells silently -- which is what the
    cross-table check in recompute() exists to catch.

    Nyxus keeps the matrix dense (row = index into its sorted unique intensities 'I', column =
    dependence - 1), so the two agree cell for cell once both axes are taken from the vectors.
    """
    P = f.P_gldm[0]
    levels = f.coefficients["ivector"]
    deps = f.coefficients["jvector"]
    return [(int(levels[i]), int(deps[j]), int(P[i, j]))
            for i, j in numpy.argwhere(P > 0)]


def independent_matrix(image, mask):
    """-> [(grey level, dependence, count)] built from the definition, without PyRadiomics.

    The whole rule, written out: a neighbour is one of the 26 offsets at Chebyshev distance 1; it is
    dependent when it lies inside the mask and its discretised level equals the centre's (alpha = 0);
    and the count starts at 1, so a voxel with no dependent neighbour lands at j = 1. Comparing this
    against PyRadiomics' C extension is what makes the neighbourhood, the cutoff and the offset
    checked facts rather than comments -- all three are exactly what Nyxus' `shifts[]` loop does.
    """
    counts = {}
    nz, ny, nx = image.shape
    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)]
    assert len(offsets) == 26, "the neighbourhood is 26 voxels"
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    continue
                here = image[z, y, x]
                dep = 1
                for dz, dy, dx in offsets:
                    az, ay, ax = z + dz, y + dy, x + dx
                    if not (0 <= az < nz and 0 <= ay < ny and 0 <= ax < nx):
                        continue
                    if mask[az, ay, ax] and image[az, ay, ax] == here:
                        dep += 1
                key = (int(here), dep)
                counts[key] = counts.get(key, 0) + 1
    return sorted((lev, dep, c) for (lev, dep), c in counts.items())


def phantom_matrix():
    """-> (cells, Ng, Nd, Nz, Np) of the compat phantom's dependence matrix.

    Ng is the number of grey levels PRESENT in the ROI, which is what Nyxus' 'I' holds -- not the
    bin count. Nd is the largest dependence observed, which is what Nyxus trims its matrix to
    (3d_gldm.cpp, `if (greyInfo) Nd = max_Nd;`).
    """
    img = sitk.ReadImage(INTEN)
    msk = sitk.ReadImage(MASK)
    bb, _ = imageoperations.checkMask(img, msk, label=LABEL)
    img, msk = imageoperations.cropToTumorMask(img, msk, bb)

    f = gldm.RadiomicsGLDM(img, msk, binCount=BINCOUNT, label=LABEL,
                           interpolator=sitk.sitkBSpline, resampledPixelSpacing=None,
                           weightingNorm=None, force2D=False)
    f._initCalculation()
    cells = nonzero_cells(f)
    # Np is counted off the mask rather than read from a coefficient. GLDM publishes no 'Np' the way
    # GLSZM does, and taking it from the matrix would make the Nz == Np identity a tautology.
    return (cells,
            len(f.coefficients["ivector"]),
            max(c[1] for c in cells),
            int(f.P_gldm.sum()),
            int(numpy.count_nonzero(f.maskArray)),
            independent_matrix(f.imageArray, f.maskArray))


def doc_matrix():
    """-> (cells, Ng, Nd) of DOC_VOLUME's dependence matrix at binWidth=1.

    binWidth=1 on levels 1..4 discretises to 1..4, which is what Nyxus' no-binning path reads
    straight off the volume, so the two sides index the same rows.
    """
    img = sitk.GetImageFromArray(DOC_VOLUME)
    msk = sitk.GetImageFromArray((DOC_VOLUME > 0).astype(numpy.uint8))
    f = gldm.RadiomicsGLDM(img, msk, binWidth=1, label=1, resampledPixelSpacing=None,
                           weightingNorm=None, force2D=False)
    f._initCalculation()
    cells = nonzero_cells(f)
    # Nz off the mask here too, for the same reason as on the phantom: summing the cells beside it
    # would pin a number derived from the table it is meant to corroborate.
    return (cells, len(f.coefficients["ivector"]), max(c[1] for c in cells),
            int(numpy.count_nonzero(f.maskArray)),
            independent_matrix(f.imageArray, f.maskArray))


def recompute(cells, Nz):
    """-> the fourteen features evaluated from the dependence matrix alone.

    The definitions transcribed from radiomics/gldm.py, written out over the pinned cells. Nothing
    here reads PyRadiomics, so comparing this against run() is a cross-table check in the sense of
    revet.txt section 9: the matrix and the scalars can only agree if they describe the same run.
    """
    pg, pd = {}, {}                          # marginals over grey level and over dependence
    for i, j, c in cells:
        pg[i] = pg.get(i, 0) + c
        pd[j] = pd.get(j, 0) + c

    tot = float(Nz)
    mu_i = sum(v * i for i, v in pg.items()) / tot
    mu_j = sum(v * j for j, v in pd.items()) / tot
    eps = numpy.spacing(1)
    return {
        "3GLDM_SDE": sum(v / (j * j) for j, v in pd.items()) / tot,
        "3GLDM_LDE": sum(v * (j * j) for j, v in pd.items()) / tot,
        "3GLDM_GLN": sum(v * v for v in pg.values()) / tot,
        "3GLDM_DN": sum(v * v for v in pd.values()) / tot,
        "3GLDM_DNN": sum(v * v for v in pd.values()) / (tot * tot),
        "3GLDM_GLV": sum(v / tot * (i - mu_i) ** 2 for i, v in pg.items()),
        "3GLDM_DV": sum(v / tot * (j - mu_j) ** 2 for j, v in pd.items()),
        "3GLDM_DE": -sum(c / tot * math.log(c / tot + eps, 2) for _, _, c in cells),
        "3GLDM_LGLE": sum(v / (i * i) for i, v in pg.items()) / tot,
        "3GLDM_HGLE": sum(v * (i * i) for i, v in pg.items()) / tot,
        "3GLDM_SDLGLE": sum(c / (i * i * j * j) for i, j, c in cells) / tot,
        "3GLDM_SDHGLE": sum(c * (i * i) / (j * j) for i, j, c in cells) / tot,
        "3GLDM_LDLGLE": sum(c * (j * j) / (i * i) for i, j, c in cells) / tot,
        "3GLDM_LDHGLE": sum(c * (i * i) * (j * j) for i, j, c in cells) / tot,
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
    ex.enableFeatureClassByName("gldm")
    res = ex.execute(INTEN, MASK)
    out = {}
    for nyx, pyr in PYRAD.items():
        key = f"original_gldm_{pyr}"
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
            re.findall(r'\{\s*"(3GLDM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


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


def check_bounds(got, Nz, ncells):
    """Every bound the family carries, checked against the oracle's own output."""
    bounds = dict(BOUNDS)
    # A sum of squared marginals over Nz cannot exceed Nz, because no marginal exceeds it.
    bounds["3GLDM_GLN"] = (1.0, float(Nz))
    bounds["3GLDM_DN"] = (1.0, float(Nz))
    # A discrete entropy cannot exceed the log of the number of outcomes it is taken over.
    bounds["3GLDM_DE"] = (0.0, math.log2(ncells))
    bad = 0
    for name, (lo, hi) in sorted(bounds.items()):
        v = got[name]
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print(f"  OUT OF RANGE {name} = {v!r}, expected [{lo}, {hi}]")
            bad += 1
    print(f"# {len(bounds) - bad}/{len(bounds)} bounded features in range")
    return bad


def check_identities(got, Nz, Np):
    """The relations the family carries independently of any oracle.

    The four inequalities are all statements about one joint distribution: i and j are >= 1, so
    weighting by 1/j^2 can only shrink a mean and weighting by j^2 can only grow it, whichever grey
    weighting is already applied. That makes them cross-checks over the whole matrix rather than
    per-marginal sanity, which is what the two equalities cover.
    """
    checks = [
        ("DNN == DN / Nz", lambda: compare("identity DNN == DN / Nz",
                                           got["3GLDM_DN"] / Nz, got["3GLDM_DNN"], 1e-12) == 0),
        ("Nz == Np", lambda: Nz == Np),
        ("SDE <= LDE", lambda: got["3GLDM_SDE"] <= got["3GLDM_LDE"]),
        ("LGLE <= HGLE", lambda: got["3GLDM_LGLE"] <= got["3GLDM_HGLE"]),
        ("SDLGLE <= SDE", lambda: got["3GLDM_SDLGLE"] <= got["3GLDM_SDE"]),
        ("SDLGLE <= LGLE", lambda: got["3GLDM_SDLGLE"] <= got["3GLDM_LGLE"]),
        ("LDHGLE >= LDE", lambda: got["3GLDM_LDHGLE"] >= got["3GLDM_LDE"]),
        ("LDHGLE >= HGLE", lambda: got["3GLDM_LDHGLE"] >= got["3GLDM_HGLE"]),
    ]
    bad = 0
    for label, test in checks:
        if not test():
            print(f"  FAIL identity {label}"
                  + (f": Nz={Nz} Np={Np}" if label == "Nz == Np" else ""))
            bad += 1
    print(f"  {len(checks) - bad}/{len(checks)} identities hold")
    return bad


def check_independent(label, pyrad_cells, indep_cells):
    """PyRadiomics' C matrix against the definition written out. Cell for cell, both directions."""
    if pyrad_cells == indep_cells:
        print(f"  {label}: all {len(pyrad_cells)} cells reproduce from the definition")
        return 0
    bad = 0
    for k in range(max(len(pyrad_cells), len(indep_cells))):
        p = pyrad_cells[k] if k < len(pyrad_cells) else None
        w = indep_cells[k] if k < len(indep_cells) else None
        if p != w:
            print(f"  FAIL {label} cell {k}: pyradiomics={p} definition={w}")
            bad += 1
    return bad


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print(f"missing phantom: {p}")
            return 1

    got = run()
    cells, Ng, Nd, Nz, Np, indep = phantom_matrix()
    doc_cells, doc_Ng, doc_Nd, doc_Np, doc_indep = doc_matrix()

    print(f"# pyradiomics {radiomics.__version__}, SimpleITK {sitk.__version__}, "
          f"numpy {numpy.__version__}, binCount={BINCOUNT}, label={LABEL}, "
          f"distances=[1], gldm_a=0")
    print(f"# dependence matrix: Ng={Ng} Nd={Nd} Nz={Nz} Np={Np}, {len(cells)} non-empty cells")
    print("# paste-ready goldens")
    for name in sorted(got):
        print(f'\t{{"{name}", {got[name]!r}}},'.ljust(56) + f"// original_gldm_{PYRAD[name]}")

    print("\n# paste-ready non-empty cells of the phantom's dependence matrix, {level, dep, count}")
    for lev, dep, cnt in cells:
        print(f"\t{{ {lev}, {dep}, {cnt} }},")

    print(f"\n# paste-ready non-empty cells of the 4x4x3 volume's matrix "
          f"(Ng={doc_Ng}, Nd={doc_Nd}), {{level, dep, count}}")
    for lev, dep, cnt in doc_cells:
        print(f"\t{{ {lev}, {dep}, {cnt} }},")

    print()
    nbad = check_bounds(got, Nz, len(cells))
    print("# identities")
    nbad += check_identities(got, Nz, Np)

    print("\n# the dependence rule re-derived from the definition, against PyRadiomics' own matrix")
    nindep = check_independent("phantom", cells, indep)
    nindep += check_independent("4x4x3 volume", doc_cells, doc_indep)

    print("\n# cross-table: the fourteen features recomputed from the matrix alone")
    ncross = 0
    again = recompute(cells, Nz)
    for name in sorted(got):
        # DE is a sum over logarithms and the two paths accumulate it in different orders
        ncross += compare(f"cross-table {name}", again[name], got[name],
                          1e-9 if name != "3GLDM_DE" else 1e-12)
    print(f"  {len(got) - ncross}/{len(got)} features reproduce from the pinned matrix")

    txt_h = open(TEST_H, encoding="utf-8", errors="replace").read()

    pins = parse_pins(txt_h, "gldm_3d_pyradiomics_ref_vals")
    print(f"\n# verifying {len(pins)} pinned goldens against this run")
    nok = nfail = nmiss = 0
    for name in sorted(pins):
        want = pins[name]
        if name not in got:
            print(f"  MISSING {name}: pinned {want!r} but PyRadiomics reports no counterpart")
            nmiss += 1
            continue
        if compare(name, got[name], want, PIN_ROUNDTRIP_RELTOL):
            nfail += 1
        else:
            print(f"  OK   {name}: pyradiomics={got[name]!r} pinned={want!r}")
            nok += 1

    # the reverse check: something the oracle produces that the header pins nothing for
    missing_pin = sorted(set(got) - set(pins))
    for name in missing_pin:
        print(f"  UNPINNED {name}: PyRadiomics reports {got[name]!r} and nothing pins it")

    nmat = 0
    for table, want_cells, want_dims in (
            ("gldm_3d_pyradiomics_matrix_ref_vals", cells,
             (("gldm_3d_pyradiomics_matrix_ng", Ng), ("gldm_3d_pyradiomics_matrix_nd", Nd),
              ("gldm_3d_pyradiomics_matrix_nz", Nz), ("gldm_3d_pyradiomics_matrix_np", Np))),
            ("gldm_3d_pyradiomics_smallmatrix_ref_vals", doc_cells,
             (("gldm_3d_pyradiomics_smallmatrix_ng", doc_Ng),
              ("gldm_3d_pyradiomics_smallmatrix_nd", doc_Nd),
              ("gldm_3d_pyradiomics_smallmatrix_nz", doc_Np)))):
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
          f"{nindep} definition mismatch(es), {nmat} matrix mismatch(es)")
    if nfail or nmiss or nbad or ncross or nindep or nmat or missing_pin:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
