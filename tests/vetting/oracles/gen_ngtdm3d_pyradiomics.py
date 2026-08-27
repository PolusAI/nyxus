"""OFFLINE PyRadiomics oracle for the 3D NGTDM family, on the NGTDM compatibility phantom.

    python tests/vetting/oracles/gen_ngtdm3d_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_3d_ngtdm_pyradiomics.h,
exiting non-zero on any mismatch, on any pin it cannot produce, and on any value it produces that
the header pins nothing for.

Recipe `ngtdm3d.pyradiomics_binwidth1`: the 4x4x3 NGTDM phantom
(tests/data/nifti/compat_int/compat_int_ngtdm_3d.nii + compat_seg/compat_seg_ngtdm_3d.nii,
label 57) at binWidth=1, no resampling, distances=[1], imageType=Original. On the Nyxus side that
is GREYDEPTH=100, IBSI=false, NGTDM_GREYDEPTH=0 (no binning) and NGTDM_RADIUS=1.

Recipe `ngtdm3d.pyradiomics_binwidth1_r2` is the same phantom and the same binning at
NGTDM_RADIUS=2, against PyRadiomics distances=[1, 2].

DISTANCES ARE SHELLS, NOT A RADIUS. PyRadiomics' `distances` lists the Chebyshev shells the
neighbourhood is drawn from: distances=[2] is the 98 offsets at Chebyshev distance exactly 2 and
excludes the 26 at distance 1. Nyxus' NGTDM_RADIUS scans the solid cube -r..r. So the config-match
for radius 2 is distances=[1, 2], not distances=[2], and distances_semantics_check() measures both
readings against the same numpy neighbourhood rather than leaving the choice asserted.

The phantom's intensities are the discrete levels 0..5. PyRadiomics' binWidth=1 discretisation maps
a value x to floor(x/1) - floor(min/1) + 1, i.e. 1..6; Nyxus keeps the raw levels and then shifts by
one because the minimum is zero. Both sides therefore work on the levels 1..6 and no binning
convention separates them.

THE PUBLIC EXTRACTOR CANNOT LOAD THIS PHANTOM. Its mask is label 57 in every one of the 48 voxels,
with no background, and imageoperations.getMask() raises "No labels found in this mask (i.e. nothing
is segmented)!" whenever numpy.unique(mask) has a single entry. So the invocation the test header
used to record (pyradiomics <image> <mask> --param settings.yaml) has never been runnable against
it. This generator constructs RadiomicsNGTDM directly instead, which is the same feature code the
extractor would reach, and cross-checks it against reference_ngtdm() below.

TWO REFERENCES, NOT ONE. reference_ngtdm() is a plain-numpy NGTDM built from the IBSI definition
with no PyRadiomics import in its path, and every value in this file is produced by both. A pin only
PyRadiomics reproduces would say the two implementations agree with each other; a pin both reproduce
says the definition is what Nyxus is being held to.

THE MATRIX, NOT ONLY THE FIVE SCALARS. All five features are contractions of one (n_i, p_i, s_i)
table over six grey levels, so two errors in that table can cancel in any one of them. The per-level
table is pinned as well, from PyRadiomics' own P_ngtdm array -- the quantity it computes before the
feature formulas, so intercepting it reimplements nothing.

Provenance: tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs
Python <= 3.9); generator=tests/vetting/oracles/gen_ngtdm3d_pyradiomics.py. Run offline; CI never
invokes it.
"""
import os
import re
import sys
from fractions import Fraction

import numpy
import SimpleITK as sitk
import radiomics
from radiomics import ngtdm

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(TESTS, "data", "nifti")
INTEN = os.path.join(DATA, "compat_int", "compat_int_ngtdm_3d.nii")
MASK = os.path.join(DATA, "compat_seg", "compat_seg_ngtdm_3d.nii")
TEST_H = os.path.join(TESTS, "test_3d_ngtdm_pyradiomics.h")

LABEL = 57
BINWIDTH = 1
RELTOL = 1e-12         # the measured residual between the two references is 0 everywhere

# Chebyshev neighbourhood radius -> the header tables pinning that recipe's goldens.
RADII = {
    1: ("ngtdm_3d_pyradiomics_ref_vals", "ngtdm_3d_pyradiomics_matrix_ref_vals"),
    2: ("ngtdm_3d_pyradiomics_r2_ref_vals", "ngtdm_3d_pyradiomics_r2_matrix_ref_vals"),
}

# Nyxus feature -> PyRadiomics NGTDM feature. These line up by name, unlike the GLCM family's.
PYRAD = {
    "3NGTDM_BUSYNESS": "Busyness",
    "3NGTDM_COARSENESS": "Coarseness",
    "3NGTDM_COMPLEXITY": "Complexity",
    "3NGTDM_CONTRAST": "Contrast",
    "3NGTDM_STRENGTH": "Strength",
}

# The 4x4 image PyRadiomics' NGTDM docstring works through by hand, which
# test_3d_ngtdm_matrix_correctness_pyradiomics() drives through D3_NGTDM_feature's own matrix
# builder as a single-slice volume. The published s_i are rounded to three figures; the pins are
# this run's full-precision values.
DOC_IMAGE = [[1, 2, 5, 2],
             [3, 5, 1, 3],
             [1, 3, 5, 5],
             [3, 1, 1, 1]]


def reference_ngtdm(levels, mask, delta=1):
    """-> [(level, n_i, p_i, s_i)], the IBSI NGTDM of an integer level volume. No PyRadiomics.

    `levels` is (z, y, x) integer grey levels, `mask` the boolean ROI. A voxel's neighbourhood is
    every in-volume, in-ROI voxel within Chebyshev distance `delta`; A_k is their mean level, and a
    voxel with no such neighbour contributes nothing. Fractions keep p_i and s_i exact, so a
    residual against PyRadiomics is a real difference rather than a summation order.
    """
    levels = numpy.asarray(levels)
    mask = numpy.asarray(mask, dtype=bool)
    nz, ny, nx = levels.shape
    n = {}
    s = {}
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not mask[z, y, x]:
                    continue
                nb = [int(levels[zz, yy, xx])
                      for zz in range(max(0, z - delta), min(nz, z + delta + 1))
                      for yy in range(max(0, y - delta), min(ny, y + delta + 1))
                      for xx in range(max(0, x - delta), min(nx, x + delta + 1))
                      if (zz, yy, xx) != (z, y, x) and mask[zz, yy, xx]]
                if not nb:
                    continue
                i = int(levels[z, y, x])
                n[i] = n.get(i, 0) + 1
                s[i] = s.get(i, Fraction(0)) + abs(Fraction(i) - Fraction(sum(nb), len(nb)))
    nvp = sum(n.values())
    return [(i, n[i], float(Fraction(n[i], nvp)), float(s[i])) for i in sorted(n)]


def reference_features(rows):
    """-> {nyxus feature: value} from an NGTDM table, by the PyRadiomics/IBSI formulas."""
    i = numpy.array([r[0] for r in rows], dtype=float)
    p = numpy.array([r[2] for r in rows], dtype=float)
    s = numpy.array([r[3] for r in rows], dtype=float)
    nvp = float(sum(r[1] for r in rows))
    ngp = int(numpy.sum(p > 0))
    d2 = (i[:, None] - i[None, :]) ** 2
    ps = p * s
    return {
        "3NGTDM_COARSENESS": 1.0 / float(numpy.sum(ps)),
        "3NGTDM_CONTRAST": (float(numpy.sum(p[:, None] * p[None, :] * d2)) / (ngp * (ngp - 1))
                            * float(numpy.sum(s)) / nvp),
        "3NGTDM_BUSYNESS": (float(numpy.sum(ps))
                            / float(numpy.sum(numpy.abs(i[:, None] * p[:, None]
                                                        - i[None, :] * p[None, :])))),
        "3NGTDM_COMPLEXITY": float(numpy.sum(numpy.abs(i[:, None] - i[None, :])
                                             * (ps[:, None] + ps[None, :])
                                             / (p[:, None] + p[None, :]))) / nvp,
        "3NGTDM_STRENGTH": float(numpy.sum((p[:, None] + p[None, :]) * d2)) / float(numpy.sum(s)),
    }


def radius_distances(radius):
    """-> the PyRadiomics `distances` list that matches a Nyxus NGTDM_RADIUS of `radius`.

    Every shell up to `radius`, because Nyxus scans the solid cube -radius..radius while `distances`
    names shells. distances_semantics_check() is the measurement this rests on.
    """
    return list(range(1, radius + 1))


def pyradiomics_ngtdm(img, msk, label, distances):
    """-> (RadiomicsNGTDM, [(level, n_i, p_i, s_i)]) for an already-loaded image/mask pair."""
    f = ngtdm.RadiomicsNGTDM(img, msk, label=label, binWidth=BINWIDTH,
                             resampledPixelSpacing=None, force2D=False, distances=distances)
    f._initCalculation()
    n_i = f.P_ngtdm[0, :, 0]
    s_i = f.P_ngtdm[0, :, 1]
    ivec = f.P_ngtdm[0, :, 2]
    p_i = f.coefficients["p_i"][0]
    rows = [(int(ivec[k]), int(n_i[k]), float(p_i[k]), float(s_i[k])) for k in range(len(ivec))]
    return f, rows


def load_phantom():
    img = sitk.ReadImage(INTEN)
    msk = sitk.Cast(sitk.ReadImage(MASK), sitk.sitkUInt32)
    arr = sitk.GetArrayFromImage(img)
    lab = sitk.GetArrayFromImage(msk)
    # binWidth=1 over a phantom whose minimum is 0: level = value + 1, which is also what Nyxus'
    # zero-min correction produces
    return img, msk, arr.astype(int) + 1, (lab == LABEL)


def parse_scalar_pins(txt, table):
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError("table %s not found in %s" % (table, TEST_H))
    body = txt[m.end():].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"(3NGTDM_[A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def parse_matrix_pins(txt, table):
    """-> [(level, n_i, p_i, s_i)] out of a ref_vals_list<Ngtdm3dMatrixRow> literal."""
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError("table %s not found in %s" % (table, TEST_H))
    body = txt[m.end():].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    out = []
    for row in re.finditer(r"\{([^{}]*)\}", body):
        parts = [p.strip() for p in row.group(1).split(",") if p.strip()]
        if len(parts) != 4:
            raise RuntimeError("%s: row %r has %d fields, expected 4"
                               % (table, row.group(1), len(parts)))
        out.append((int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])))
    return out


def rel(have, want):
    return abs(have - want) / max(abs(want), 1e-300)


def compare(what, have, want, bad):
    """Reports and counts one comparison; returns the running failure count."""
    r = rel(have, want)
    if r > RELTOL:
        print("  FAIL %s: oracle=%r pinned=%r rel=%.3g" % (what, have, want, r))
        return bad + 1
    return bad


def cross_table_checks(txt):
    """The five scalar pins recomputed from the matrix pins alone. -> failure count.

    The two tables come from the same PyRadiomics object but through different attributes, so
    nothing in the run itself stops one being edited without the other. Every feature is a
    contraction of (i, p_i, s_i), so the matrix table determines all five scalars exactly -- which
    makes this the check that a copy-paste into one table and not the other cannot survive.
    """
    bad = 0
    for radius, (scalar_table, matrix_table) in sorted(RADII.items()):
        rows = parse_matrix_pins(txt, matrix_table)
        pins = parse_scalar_pins(txt, scalar_table)
        derived = reference_features(rows)
        print("\n# cross-table: the %d radius-%d feature pins recomputed from %s"
              % (len(pins), radius, matrix_table))
        for name in sorted(pins):
            bad = compare("%s from %s" % (name, matrix_table), derived[name], pins[name], bad)
    return bad


def range_checks(txt):
    """Bounds every pin holds by construction. -> failure count.

    A golden outside its own range is the cheapest kind of wrong, and this is the only check here
    that needs no oracle at all. It catches a rotted pin, not a wrong definition.
    """
    bad = 0
    print("\n# range and identity checks over every pin in the header")
    for scalar_table, _ in sorted(RADII.values()):
        for name, v in sorted(parse_scalar_pins(txt, scalar_table).items()):
            if not v > 0:                  # all five are sums of non-negative terms over a
                print("  FAIL %s %s: %r is not > 0" % (scalar_table, name, v))  # non-degenerate ROI
                bad += 1
    matrix_tables = [(m, 48) for _, m in sorted(RADII.values())]
    for table, nvp in matrix_tables + [("ngtdm_3d_pyradiomics_docmatrix_ref_vals", 16)]:
        rows = parse_matrix_pins(txt, table)
        levels = [r[0] for r in rows]
        if levels != sorted(levels):
            print("  FAIL %s: levels %s are not ascending" % (table, levels))
            bad += 1
        for lev, n, p, s in rows:
            if n < 1:                      # both tools drop empty levels, so no row may be empty
                print("  FAIL %s i=%d: n_i=%d < 1" % (table, lev, n))
                bad += 1
            if s < 0:
                print("  FAIL %s i=%d: s_i=%r < 0" % (table, lev, s))
                bad += 1
            bad = compare("%s i=%d p_i == n_i/Nvp" % (table, lev), n / float(nvp), p, bad)
        total_n = sum(r[1] for r in rows)
        if total_n != nvp:
            print("  FAIL %s: sum(n_i)=%d, expected the fixture's %d voxels" % (table, total_n, nvp))
            bad += 1
        bad = compare("%s sum(p_i) == 1" % table, sum(r[2] for r in rows), 1.0, bad)
    return bad


def radius_run(img, msk, levels, roi, radius):
    """-> ([(level, n_i, p_i, s_i)], {feature: value}) at one NGTDM_RADIUS, or None on disagreement.

    Both references are built at the same radius and must agree on the levels and their counts
    before anything from either is pinned, so a radius the two implementations read differently
    stops the generator instead of producing goldens.
    """
    dists = radius_distances(radius)
    f, rows = pyradiomics_ngtdm(img, msk, LABEL, dists)
    ref_rows = reference_ngtdm(levels, roi, delta=radius)

    if [r[:2] for r in rows] != [r[:2] for r in ref_rows]:
        print("  FAIL the two NGTDM references disagree on the levels/counts at radius %d:\n"
              "       pyradiomics %s\n       reference   %s"
              % (radius, [r[:2] for r in rows], [r[:2] for r in ref_rows]))
        return None

    worst = max(max(rel(a[2], b[2]), rel(a[3], b[3])) for a, b in zip(rows, ref_rows))
    print("# radius %d (distances=%s), pyradiomics vs the independent reference NGTDM:"
          " worst rel %.3g over %d levels" % (radius, dists, worst, len(rows)))

    feats = {n: float(numpy.asarray(getattr(f, "get%sFeatureValue" % p)()).ravel()[0])
             for n, p in PYRAD.items()}
    ref_feats = reference_features(ref_rows)
    worstf = max(rel(feats[n], ref_feats[n]) for n in PYRAD)
    print("# radius %d, pyradiomics vs the independent reference features: worst rel %.3g"
          % (radius, worstf))
    return rows, feats


def distances_semantics_check(img, msk, levels, roi):
    """Why the radius-2 recipe passes distances=[1, 2] and not distances=[2]. -> failure count.

    "Distance 2" reads two ways -- the shell at exactly 2, or everything out to 2 -- and they are
    different numbers, so only one of them is config-matched to a Nyxus run at NGTDM_RADIUS=2. Both
    readings are measured here against the same numpy neighbourhood, which is what keeps the recipe's
    choice of distances a measurement rather than a claim, and what would report it if a PyRadiomics
    release changed the convention under the pins.
    """
    _, shell = pyradiomics_ngtdm(img, msk, LABEL, [2])
    _, solid = pyradiomics_ngtdm(img, msk, LABEL, radius_distances(2))
    ref = reference_ngtdm(levels, roi, delta=2)
    worst_solid = max(rel(a[3], b[3]) for a, b in zip(solid, ref))
    worst_shell = max(rel(a[3], b[3]) for a, b in zip(shell, ref))

    print("\n# PyRadiomics `distances` lists Chebyshev shells, it is not a radius")
    print("  distances=[1, 2] vs the solid radius-2 neighbourhood: worst s_i rel %.3g" % worst_solid)
    print("  distances=[2]    vs the solid radius-2 neighbourhood: worst s_i rel %.3g" % worst_shell)
    bad = 0
    if worst_solid > RELTOL:
        print("  FAIL distances=[1, 2] no longer reproduces the solid radius-2 neighbourhood")
        bad += 1
    if worst_shell <= RELTOL:
        print("  FAIL distances=[2] reproduces it too, so the recipe's shell note is stale")
        bad += 1
    return bad


def main():
    for p in (INTEN, MASK):
        if not os.path.exists(p):
            print("missing phantom: %s" % p)
            return 1

    radiomics.logger.setLevel(40)
    img, msk, levels, roi = load_phantom()

    print("# pyradiomics %s, SimpleITK %s, binWidth=%d, label=%d"
          % (radiomics.__version__, sitk.__version__, BINWIDTH, LABEL))

    runs = {}
    for radius in sorted(RADII):
        run = radius_run(img, msk, levels, roi, radius)
        if run is None:
            return 1
        runs[radius] = run

    doc_levels = numpy.array(DOC_IMAGE, dtype=int)[None, :, :]
    doc_rows = reference_ngtdm(doc_levels, numpy.ones_like(doc_levels, dtype=bool))
    doc_img = sitk.GetImageFromArray(numpy.array(DOC_IMAGE, dtype=float)[None, :, :])
    doc_msk = sitk.GetImageFromArray(numpy.ones((1, 4, 4), dtype=numpy.uint32))
    _, doc_pyrad = pyradiomics_ngtdm(doc_img, doc_msk, 1, radius_distances(1))
    if [r[:2] for r in doc_pyrad] != [r[:2] for r in doc_rows]:
        print("  FAIL the two references disagree on the doc-example NGTDM")
        return 1
    worstd = max(max(rel(a[2], b[2]), rel(a[3], b[3])) for a, b in zip(doc_pyrad, doc_rows))
    print("# doc-example NGTDM, pyradiomics vs the independent reference: worst rel %.3g" % worstd)

    for radius in sorted(RADII):
        scalar_table, matrix_table = RADII[radius]
        rows, feats = runs[radius]
        print("\n# paste-ready goldens: %s" % scalar_table)
        for name in sorted(feats):
            print(('\t{"%s", %r},' % (name, feats[name])).ljust(56)
                  + "// original_ngtdm_%s" % PYRAD[name])

        print("\n# paste-ready goldens: %s   { i, n_i, p_i, s_i }" % matrix_table)
        for lev, n, p, s in rows:
            print("\t{ %d, %d, %r, %r }," % (lev, n, p, s))

    print("\n# paste-ready goldens: ngtdm_3d_pyradiomics_docmatrix_ref_vals   { i, n_i, p_i, s_i }")
    for lev, n, p, s in doc_pyrad:
        print("\t{ %d, %d, %r, %r }," % (lev, n, p, s))

    txt_h = open(TEST_H, encoding="utf-8", errors="replace").read()
    bad = 0

    tables = []
    for radius in sorted(RADII):
        scalar_table, matrix_table = RADII[radius]
        rows, feats = runs[radius]
        tables.append((matrix_table, rows))

        pins = parse_scalar_pins(txt_h, scalar_table)
        print("\n# verifying %d pinned goldens of %s against this run" % (len(pins), scalar_table))
        for name, want in sorted(pins.items()):
            if name not in feats:
                print("  FAIL %s %s: pinned but PyRadiomics produces no such feature"
                      % (scalar_table, name))
                bad += 1
                continue
            bad = compare("%s %s" % (scalar_table, name), feats[name], want, bad)
        for name in sorted(feats):
            if name not in pins:
                print("  FAIL %s: PyRadiomics produces it but %s pins nothing for it"
                      % (name, scalar_table))
                bad += 1

    for table, produced in tables + [("ngtdm_3d_pyradiomics_docmatrix_ref_vals", doc_pyrad)]:
        pinned = parse_matrix_pins(txt_h, table)
        print("\n# verifying %d pinned rows of %s against this run" % (len(pinned), table))
        if len(pinned) != len(produced):
            print("  FAIL %s: %d rows pinned, oracle produces %d"
                  % (table, len(pinned), len(produced)))
            bad += 1
            continue
        for (plev, pn, pp, ps), (lev, n, p, s) in zip(pinned, produced):
            if plev != lev or pn != n:
                print("  FAIL %s: pinned level/count (%d, %d) != oracle (%d, %d)"
                      % (table, plev, pn, lev, n))
                bad += 1
                continue
            bad = compare("%s i=%d p_i" % (table, lev), p, pp, bad)
            bad = compare("%s i=%d s_i" % (table, lev), s, ps, bad)

    bad += cross_table_checks(txt_h)
    bad += range_checks(txt_h)
    bad += distances_semantics_check(img, msk, levels, roi)

    print("\nALL CHECKS PASSED" if not bad else "\n%d MISMATCH(ES)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
