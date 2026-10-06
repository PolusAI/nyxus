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

Recipe `ngtdm3d.pyradiomics_binwidth1_ball` is the same binning and NGTDM_RADIUS=1 on a ROI that
does NOT fill its bounding box: the NGTDM ball phantom
(tests/data/nifti/compat_int/compat_int_ngtdm_3d_ball.nii + compat_seg/compat_seg_ngtdm_3d_ball.nii,
label 57), a digital ball of radius 3 whose 7x7x7 bounding box is 64% background. The 4x4x3 phantom
above has no background voxel at all, so it cannot tell a neighbourhood confined to the ROI from one
that reaches into the rest of the bounding box; this one can. Its intensities are 0..5 inside the
ROI too, so a ROI voxel of level 0 and a background cell -- which the binned cube cannot tell apart --
both occur in every neighbourhood near the surface. make_ball_phantom() defines it, `--write-ball`
writes the two files, and every run checks the committed files still hold exactly that.

The same seg file carries label 58: two voxels at opposite corners of the volume, neither within
reach of the other. No voxel of it has a ROI neighbour, so its NGTDM is empty and Nyxus reports all
five features as the soft-NaN value (test_3d_ngtdm_isolated_voxels_mechanics), as MIRP reports NaN.

A VOXEL WITH NO ROI NEIGHBOUR is where PyRadiomics parts from IBSI. Nyxus follows IBSI, as MIRP does:
the voxel is in no row and counts towards none of n_i, Nvp or Ngp, and Ngp is the number of non-empty
rows. PyRadiomics keeps the voxel as a row with s_i = 0. So ANY ROI holding one such voxel differs
from PyRadiomics on all five features, not only an all-isolated ROI like label 58 -- which is why
every PyRadiomics fixture here is a ROI in which every voxel has a neighbour, and the generator fails
on the ball or the out-of-core ellipsoid otherwise. Label 59 of the same file mixes the two kinds of
voxel and is held to MIRP instead, by gen_ngtdm3d_mirp.py. check_ball_phantom() measures the
properties of all three labels.

The out-of-core path is held to PyRadiomics on a larger ellipsoid built in
tests/python/test_3d_ngtdm_pyradiomics.py, whose footprint forces the oversized path; make_ooc_ellipsoid()
here is the same volume, and its goldens are pinned in that file's NGTDM_3D_NONBOX_PYRADIOMICS.

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
BALL_INTEN = os.path.join(DATA, "compat_int", "compat_int_ngtdm_3d_ball.nii")
BALL_MASK = os.path.join(DATA, "compat_seg", "compat_seg_ngtdm_3d_ball.nii")
TEST_OOC_PY = os.path.join(TESTS, "python", "test_3d_ngtdm_pyradiomics.py")

LABEL = 57
ISOLATED_LABEL = 58          # the ball file's two-voxel ROI with no neighbours
MIXED_LABEL = 59             # the ball file's ROI with both neighboured voxels and an isolated one

# Label 59's voxels, (z, y, x): a three-voxel chain along x on the volume's z=0, y=0 edge, and one
# voxel at the far z face that nothing of label 59 reaches at radius 1. gen_ngtdm3d_mirp.py reads the
# committed files, so the coordinates live in one place: here.
MIXED_CHAIN = ((0, 0, 5), (0, 0, 6), (0, 0, 7))
MIXED_LONE = (8, 0, 0)
BINWIDTH = 1
RELTOL = 1e-12         # the measured residual between the two references is 0 everywhere

# Chebyshev neighbourhood radius -> the header tables pinning that recipe's goldens.
RADII = {
    1: ("ngtdm_3d_pyradiomics_ref_vals", "ngtdm_3d_pyradiomics_matrix_ref_vals"),
    2: ("ngtdm_3d_pyradiomics_r2_ref_vals", "ngtdm_3d_pyradiomics_r2_matrix_ref_vals"),
}

# The ball recipe's header tables, and its valid-voxel count (every one of its 123 voxels).
BALL_TABLES = ("ngtdm_3d_pyradiomics_ball_ref_vals", "ngtdm_3d_pyradiomics_ball_matrix_ref_vals")
BALL_NVP = 123

# The out-of-core test's pin table in TEST_OOC_PY.
OOC_PINS = "NGTDM_3D_NONBOX_PYRADIOMICS"

# Nyxus feature -> PyRadiomics NGTDM feature. These line up by name, unlike the GLCM family's.
PYRAD = {
    "3NGTDM_BUSYNESS": "Busyness",
    "3NGTDM_COARSENESS": "Coarseness",
    "3NGTDM_COMPLEXITY": "Complexity",
    "3NGTDM_CONTRAST": "Contrast",
    "3NGTDM_STRENGTH": "Strength",
}

# The 4x4 image PyRadiomics' NGTDM docstring works through by hand, which
# test_3d_ngtdm_docmatrix_pyradiomics() drives through D3_NGTDM_feature's own matrix builder as a
# single-slice volume. The published s_i are rounded to three figures; the pins are this run's
# full-precision values.
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


def phantom_texture(shape, modulus):
    """-> integer levels 0..modulus-1 over a (z, y, x) volume, deterministic and textured.

    A closed form rather than a seeded generator, so the volume is the same on any numpy and the
    Python test that builds the out-of-core volume can spell it out in one line.
    """
    z, y, x = numpy.indices(shape)
    return (3 * x + 5 * y + 7 * z + x * y * z) % modulus


def make_ball_phantom():
    """-> (intensity float32, labels uint32), both (z, y, x) = 9x9x9: the ball recipe's phantom.

    Label 57 is the digital ball of radius 3 about the centre voxel (123 voxels in a 7x7x7 bounding
    box); label 58 is the two corner voxels (0,0,0) and (8,8,8), at levels 0 and 2; label 59 is
    MIXED_CHAIN, at levels 3, 0, 3, plus MIXED_LONE at level 2 -- a level no chain voxel carries. The
    texture covers the whole volume, background included, as a real image's would.
    """
    n, c = 9, 4
    z, y, x = numpy.indices((n, n, n))
    labels = numpy.zeros((n, n, n), dtype=numpy.uint32)
    labels[(z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= 9] = LABEL
    labels[0, 0, 0] = labels[n - 1, n - 1, n - 1] = ISOLATED_LABEL
    for v in MIXED_CHAIN + (MIXED_LONE,):
        labels[v] = MIXED_LABEL
    return phantom_texture((n, n, n), 6).astype(numpy.float32), labels


def make_ooc_ellipsoid():
    """-> (intensity uint16, mask bool), (z, y, x) = 24x60x60: the out-of-core test's volume.

    Must stay the volume tests/python/test_3d_ngtdm_pyradiomics.py::_make_nonbox_volume_pair builds.
    """
    shape = (24, 60, 60)
    z, y, x = numpy.indices(shape)
    mask = ((z - 11.5) / 11.5) ** 2 + ((y - 29.5) / 28.0) ** 2 + ((x - 29.5) / 28.0) ** 2 <= 1.0
    return phantom_texture(shape, 12).astype(numpy.uint16), mask


def write_ball_phantom():
    inten, labels = make_ball_phantom()
    for arr, path in ((inten, BALL_INTEN), (labels, BALL_MASK)):
        sitk.WriteImage(sitk.GetImageFromArray(arr), path)
        print("wrote %s" % path)


def isolated_voxels(roi, delta=1):
    """-> the number of ROI voxels with no ROI voxel within Chebyshev distance `delta`."""
    nz, ny, nx = roi.shape
    count = 0
    for z, y, x in zip(*numpy.nonzero(roi)):
        box = roi[max(0, z - delta):z + delta + 1, max(0, y - delta):y + delta + 1,
                  max(0, x - delta):x + delta + 1]
        if box.sum() == 1:
            count += 1
    return count


def check_ball_phantom():
    """The committed ball files are make_ball_phantom(), and hold the properties the tests rely on.
    -> failure count."""
    bad = 0
    want_i, want_l = make_ball_phantom()
    have_i = sitk.GetArrayFromImage(sitk.ReadImage(BALL_INTEN))
    have_l = sitk.GetArrayFromImage(sitk.ReadImage(BALL_MASK))
    print("\n# the committed ball phantom against make_ball_phantom()")
    if have_i.shape != want_i.shape or not numpy.array_equal(have_i, want_i):
        print("  FAIL %s is not make_ball_phantom()'s intensity volume" % BALL_INTEN)
        bad += 1
    if have_l.shape != want_l.shape or not numpy.array_equal(have_l, want_l):
        print("  FAIL %s is not make_ball_phantom()'s label volume" % BALL_MASK)
        bad += 1

    ball = want_l == LABEL
    lo = [int(numpy.nonzero(ball)[k].min()) for k in range(3)]
    hi = [int(numpy.nonzero(ball)[k].max()) for k in range(3)]
    bbox = int(numpy.prod([h - l + 1 for l, h in zip(lo, hi)]))
    print("  label %d: %d voxels in a %d-voxel bounding box, %d of them at level 0"
          % (LABEL, ball.sum(), bbox, int((want_i[ball] == 0).sum())))
    if ball.sum() == bbox:
        print("  FAIL label %d fills its bounding box, so it cannot discriminate" % LABEL)
        bad += 1
    if not (want_i[ball] == 0).any():
        print("  FAIL label %d has no voxel at level 0" % LABEL)
        bad += 1
    if isolated_voxels(ball, 2):            # radius 1 and 2 both need every voxel to have a neighbour
        print("  FAIL label %d has a voxel without a neighbour" % LABEL)
        bad += 1
    iso = want_l == ISOLATED_LABEL
    if isolated_voxels(iso) != iso.sum() or len(set(want_i[iso].tolist())) < 2:
        print("  FAIL label %d is not %d voxels at distinct levels, none with a neighbour"
              % (ISOLATED_LABEL, iso.sum()))
        bad += 1

    # label 59: some voxels with a neighbour and some without, at least two levels among the former,
    # and every level of the latter carried by no voxel of the former -- so the isolated voxels' levels
    # have empty NGTDM rows, which is the one configuration where the count of non-empty rows (Ngp)
    # differs from the count of the ROI's levels
    mixed = want_l == MIXED_LABEL
    if (mixed & ((want_l == LABEL) | iso)).any():
        print("  FAIL label %d overlaps label %d or %d" % (MIXED_LABEL, LABEL, ISOLATED_LABEL))
        bad += 1
    lone = numpy.zeros_like(mixed)
    for zz, yy, xx in zip(*numpy.nonzero(mixed)):
        box = mixed[max(0, zz - 1):zz + 2, max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2]
        lone[zz, yy, xx] = box.sum() == 1
    linked = mixed & ~lone
    linked_levels = set(want_i[linked].tolist())
    lone_levels = set(want_i[lone].tolist())
    print("  label %d: %d voxels with a neighbour at levels %s, %d without at levels %s"
          % (MIXED_LABEL, linked.sum(), sorted(linked_levels), lone.sum(), sorted(lone_levels)))
    if not lone.any() or len(linked_levels) < 2 or lone_levels & linked_levels:
        print("  FAIL label %d is not a mix of neighboured voxels at >= 2 levels and isolated voxels"
              " at levels of their own" % MIXED_LABEL)
        bad += 1
    return bad


def ball_run():
    """-> ([(level, n_i, p_i, s_i)], {feature: value}) on the ball at radius 1, or None.

    Three readings must agree before anything is pinned: RadiomicsNGTDM on the loaded pair, the
    numpy reference, and PyRadiomics' public extractor -- which this phantom, having background,
    can load where the 4x4x3 one cannot.
    """
    img = sitk.ReadImage(BALL_INTEN)
    msk = sitk.Cast(sitk.ReadImage(BALL_MASK), sitk.sitkUInt32)
    inten, labels = make_ball_phantom()
    roi = labels == LABEL
    levels = inten.astype(int) - int(inten[roi].min()) + 1   # binWidth=1: floor(x) - floor(min) + 1

    f, rows = pyradiomics_ngtdm(img, msk, LABEL, radius_distances(1))
    ref_rows = reference_ngtdm(levels, roi, delta=1)
    if [r[:2] for r in rows] != [r[:2] for r in ref_rows]:
        print("  FAIL the two NGTDM references disagree on the ball's levels/counts")
        return None
    worst = max(max(rel(a[2], b[2]), rel(a[3], b[3])) for a, b in zip(rows, ref_rows))
    print("# ball, pyradiomics vs the independent reference NGTDM: worst rel %.3g over %d levels"
          % (worst, len(rows)))

    feats = {n: float(numpy.asarray(getattr(f, "get%sFeatureValue" % p)()).ravel()[0])
             for n, p in PYRAD.items()}
    ref_feats = reference_features(ref_rows)
    print("# ball, pyradiomics vs the independent reference features: worst rel %.3g"
          % max(rel(feats[n], ref_feats[n]) for n in PYRAD))

    from radiomics import featureextractor
    ex = featureextractor.RadiomicsFeatureExtractor(binWidth=BINWIDTH, label=LABEL,
                                                    distances=radius_distances(1))
    ex.disableAllFeatures()
    ex.enableFeatureClassByName("ngtdm")
    out = ex.execute(BALL_INTEN, BALL_MASK)
    worstx = max(rel(float(out["original_ngtdm_%s" % p]), feats[n]) for n, p in PYRAD.items())
    print("# ball, the public extractor vs RadiomicsNGTDM: worst rel %.3g" % worstx)
    if worstx > RELTOL:
        print("  FAIL the public extractor and RadiomicsNGTDM disagree on the ball")
        return None
    return rows, feats


def ooc_run():
    """-> ({feature: value}, failure count), PyRadiomics on the out-of-core test's ellipsoid at
    radius 1. A voxel without a neighbour is a failure: PyRadiomics keeps it and Nyxus does not, so
    one such voxel would part the two on every feature."""
    inten, mask = make_ooc_ellipsoid()
    img = sitk.GetImageFromArray(inten.astype(float))
    msk = sitk.GetImageFromArray(mask.astype(numpy.uint32))
    f, _ = pyradiomics_ngtdm(img, msk, 1, radius_distances(1))
    lone = isolated_voxels(mask)
    print("# out-of-core ellipsoid: %d voxels, %d of them at level 0, %d without a neighbour"
          % (mask.sum(), int((inten[mask] == 0).sum()), lone))
    bad = 0
    if lone:
        print("  FAIL the out-of-core ellipsoid has %d voxel(s) without a neighbour" % lone)
        bad += 1
    return {n: float(numpy.asarray(getattr(f, "get%sFeatureValue" % p)()).ravel()[0])
            for n, p in PYRAD.items()}, bad


def parse_py_pins(txt, name):
    """-> {feature: value} out of a `NAME = { "3NGTDM_X": value, ... }` literal in a Python test."""
    m = re.search(re.escape(name) + r"\s*=\s*\{", txt)
    if not m:
        raise RuntimeError("%s not found in %s" % (name, TEST_OOC_PY))
    body = re.sub(r"#[^\n]*", "", txt[m.end():].split("}", 1)[0])
    return {n: float(v) for n, v in re.findall(r'"(3NGTDM_[A-Z0-9_]+)"\s*:\s*([-0-9.eE+]+)', body)}


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
    for scalar_table, matrix_table in [RADII[k] for k in sorted(RADII)] + [BALL_TABLES]:
        rows = parse_matrix_pins(txt, matrix_table)
        pins = parse_scalar_pins(txt, scalar_table)
        derived = reference_features(rows)
        print("\n# cross-table: the %d feature pins of %s recomputed from %s"
              % (len(pins), scalar_table, matrix_table))
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
    for scalar_table, _ in sorted(RADII.values()) + [BALL_TABLES]:
        for name, v in sorted(parse_scalar_pins(txt, scalar_table).items()):
            if not v > 0:                  # all five are sums of non-negative terms over a
                print("  FAIL %s %s: %r is not > 0" % (scalar_table, name, v))  # non-degenerate ROI
                bad += 1
    matrix_tables = [(m, 48) for _, m in sorted(RADII.values())] + [(BALL_TABLES[1], BALL_NVP)]
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
    if "--write-ball" in sys.argv[1:]:
        write_ball_phantom()

    for p in (INTEN, MASK, BALL_INTEN, BALL_MASK):
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

    ball = ball_run()
    if ball is None:
        return 1
    ooc_feats, ooc_bad = ooc_run()

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

    print("\n# paste-ready goldens: %s" % BALL_TABLES[0])
    for name in sorted(ball[1]):
        print(('\t{"%s", %r},' % (name, ball[1][name])).ljust(56)
              + "// original_ngtdm_%s" % PYRAD[name])
    print("\n# paste-ready goldens: %s   { i, n_i, p_i, s_i }" % BALL_TABLES[1])
    for lev, n, p, s in ball[0]:
        print("\t{ %d, %d, %r, %r }," % (lev, n, p, s))

    print("\n# paste-ready goldens: %s (%s)" % (OOC_PINS, os.path.relpath(TEST_OOC_PY, TESTS)))
    for name in sorted(ooc_feats):
        print('    "%s": %r,' % (name, ooc_feats[name]))

    txt_h = open(TEST_H, encoding="utf-8", errors="replace").read()
    bad = check_ball_phantom() + ooc_bad

    tables = []
    scalar_runs = [(RADII[r][0], RADII[r][1], runs[r]) for r in sorted(RADII)]
    for scalar_table, matrix_table, (rows, feats) in scalar_runs + [BALL_TABLES + (ball,)]:
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

    txt_py = open(TEST_OOC_PY, encoding="utf-8", errors="replace").read()
    py_pins = parse_py_pins(txt_py, OOC_PINS)
    print("\n# verifying %d pinned goldens of %s against this run" % (len(py_pins), OOC_PINS))
    if sorted(py_pins) != sorted(ooc_feats):
        print("  FAIL %s pins %s, PyRadiomics produces %s" % (OOC_PINS, sorted(py_pins), sorted(ooc_feats)))
        bad += 1
    for name in sorted(set(py_pins) & set(ooc_feats)):
        bad = compare("%s %s" % (OOC_PINS, name), ooc_feats[name], py_pins[name], bad)

    bad += cross_table_checks(txt_h)
    bad += range_checks(txt_h)
    bad += distances_semantics_check(img, msk, levels, roi)

    print("\nALL CHECKS PASSED" if not bad else "\n%d MISMATCH(ES)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
