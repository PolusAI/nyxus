"""OFFLINE golden generator for the Hu-invariant goldens in test_moments_skimage.h /
test_moments_regression.h (SPEC 6.4), refreshed for the calcHu_imp h5/h6 formula fix.

Context. Nyxus's calcHu_imp had two formula defects (both also present in the _nt and CUDA
twins, fixed together):
  - h5: the second bracket computed (3*(eta30+eta12))^2 == 9*(eta30+eta12)^2 instead of
    Hu's 3*(eta30+eta12)^2.
  - h6: a precedence error left "+eta03" outside the product -- 4*eta11*(eta30+eta12)*eta21 + eta03
    instead of 4*eta11*(eta30+eta12)*(eta21+eta03) -- so the stray raw eta03 dominated h6.
    (Tell-tale in the old pinned goldens: IMOM_HU6 == IMOM_NCM_03, WEIGHTED_HU_M6 ==
    WT_NORM_CTR_MOM_03, IMOM_WHU6 == IMOM_WNCM_03.)

This generator:
  A. Rebuilds the pinned 48x40 rectangle fixture of test_moments_common.h, recomputes every
     moment family with scikit-image in Nyxus's coordinate convention, verifies all goldens
     UNAFFECTED by the fix still match (validating that the oracle reproduces Nyxus's
     conventions), verifies the OLD h5/h6 goldens equal the buggy formulas (closing the loop
     that they encoded the defect), and prints the corrected goldens.
  B. Recomputes the Nyxus-specific weighted Hu snapshot values (no external oracle for the
     W-weighting itself) by feeding the PINNED weighted normalized central moments through
     skimage.measure.moments_hu -- i.e., skimage executes the Hu formula on Nyxus's eta.
  C. Emits goldens for a NEW asymmetric right-triangle fixture whose odd-order eta are large
     enough that the buggy h5/h6 formulas fail the gtest tolerance -- the regression-proof
     test the symmetric rectangle cannot provide -- and proves discriminance by printing
     |buggy - correct| against the gtest tolerance.

Provenance: tool=scikit-image, version=0.26.0; numpy 2.4.6; env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_moments_skimage.py. Run offline; CI never invokes it.

Coordinate convention: Nyxus m_pq has p on x, q on y. Arrays here are indexed A[x, y] so
skimage's moments(A)[i, j] lands on i==p, j==q with no transposition (and no h7 sign flip).
"""

import numpy as np
from skimage.measure import moments, moments_central, moments_normalized, moments_hu

W, H = 48, 40
REL_TOL = 1e-9  # validation gate against pinned goldens (gtest itself uses 1e-6)


# ---------------------------------------------------------------- fixtures

def rect_shape():
    """Binary 48x40 rectangle, A[x, y] convention."""
    return np.ones((W, H), dtype=np.float64)


def rect_intensity():
    """test_moments_common.h: I(x,y) = 10 + 3x + 5y + (x*y)%7, A[x, y] convention."""
    a = np.zeros((W, H), dtype=np.float64)
    for x in range(W):
        for y in range(H):
            a[x, y] = 10.0 + 3.0 * x + 5.0 * y + float((x * y) % 7)
    return a


WEDGE_W, WEDGE_H = 40, 8

def wedge_shape():
    """Thin right wedge: pixels (x, y) with 0<=x<40, 0<=y<8, 5*y <= x (A[x, y] convention).
    Elongated AND skewed, so the odd-order etas are large (eta30 ~ -0.23) and the buggy
    h5/h6 formulas miss the correct values by ~440x / ~1900x the gtest tolerance -- the
    regression-proof discriminance a symmetric fixture cannot provide. (A compact
    symmetric-ish shape fails: e.g. a 20x20 diagonal half-square discriminates h5 by only
    0.35x the tolerance.)"""
    a = np.zeros((WEDGE_W, WEDGE_H), dtype=np.float64)
    for x in range(WEDGE_W):
        for y in range(WEDGE_H):
            if 5 * y <= x:
                a[x, y] = 1.0
    return a


# ---------------------------------------------------------------- Hu formulas

def hu_correct(nu):
    """Standard Hu invariants (== skimage.measure.moments_hu), from eta indexed nu[p, q]."""
    return moments_hu(nu)


def hu_buggy_h5_h6(nu):
    """The two DEFECTIVE pre-fix Nyxus formulas, for closing the loop on old goldens."""
    _20, _02, _11 = nu[2, 0], nu[0, 2], nu[1, 1]
    _30, _03, _12, _21 = nu[3, 0], nu[0, 3], nu[1, 2], nu[2, 1]
    h5 = (_30 - 3 * _12) * (_30 + _12) * ((_30 + _12) ** 2 - 3 * (_21 + _03) ** 2) + \
         (3 * _21 - _03) * (_21 + _03) * ((3 * (_30 + _12)) ** 2 - (_21 + _03) ** 2)
    h6 = (_20 - _02) * ((_30 + _12) ** 2 - (_21 + _03) ** 2) + \
         (4 * _11 * (_30 + _12) * _21 + _03)
    return h5, h6


def nu_matrix_from_pinned(vals):
    """Build a nu matrix (nu[p, q]) from pinned Nyxus eta goldens {(p,q): value}."""
    nu = np.zeros((4, 4), dtype=np.float64)
    nu[0, 0] = 1.0
    for (p, q), v in vals.items():
        nu[p, q] = v
    return nu


def full_stack(a):
    """raw m, central mu, normalized eta (nu[p, q]), Hu -- all in Nyxus convention."""
    m = moments(a, order=3)
    mu = moments_central(a, order=3)
    nu = moments_normalized(mu, order=3)
    hu = hu_correct(np.nan_to_num(nu))
    return m, mu, nu, hu


def check(name, got, pinned, tol=REL_TOL):
    scale = max(1.0, abs(pinned), abs(got))
    ok = abs(got - pinned) <= tol * scale
    print(f"  {'OK ' if ok else 'FAIL'} {name}: oracle={got!r} pinned={pinned!r}")
    return ok


def main():
    all_ok = True

    # ---------------------------------------------------------------- A. rectangle fixture
    print("=== A. 48x40 rectangle fixture (test_moments_common.h) ===")
    shp = rect_shape()
    inten = rect_intensity()

    m_s, mu_s, nu_s, hu_s = full_stack(shp)
    m_i, mu_i, nu_i, hu_i = full_stack(inten)

    print("-- validation: goldens UNAFFECTED by the h5/h6 fix (oracle must reproduce them) --")
    # shape: raw + central + eta + Hu 1-4,7 (test_moments_skimage.h)
    for name, got, pinned in [
        ("SPAT_MOMENT_00", m_s[0, 0], 1920.0),
        ("SPAT_MOMENT_10", m_s[1, 0], 45120.0),
        ("SPAT_MOMENT_01", m_s[0, 1], 37440.0),
        ("CENTRAL_MOMENT_20", mu_s[2, 0], 368480.0),
        ("CENTRAL_MOMENT_02", mu_s[0, 2], 255840.0),
        ("NORM_CENTRAL_MOMENT_20", nu_s[2, 0], 0.09995659722222222),
        ("NORM_CENTRAL_MOMENT_02", nu_s[0, 2], 0.06940104166666666),
        ("HU_M1", hu_s[0], 0.16935763888888888),
        ("HU_M2", hu_s[1], 0.0009336419753086421),
        ("HU_M3", hu_s[2], 0.0),
        ("HU_M4", hu_s[3], 0.0),
        ("HU_M6", hu_s[5], 0.0),
        ("HU_M7", hu_s[6], 0.0),  # pinned -2.36e-10 is summation noise; oracle exact 0
        # intensity: raw + central + eta + Hu 1-4 (test_moments_skimage.h)
        ("IMOM_RM_00", m_i[0, 0], 346635.0),
        ("IMOM_RM_10", m_i[1, 0], 9253494.0),
        ("IMOM_CM_20", mu_i[2, 0], 62976163.595638104),
        ("IMOM_CM_03", mu_i[0, 3], -169617579.29906213),
        ("IMOM_CM_30", mu_i[3, 0], -232054083.1110376),
        ("IMOM_NCM_20", nu_i[2, 0], 0.0005241207783805112),
        ("IMOM_NCM_03", nu_i[0, 3], -2.3976723321608506e-06),
        ("IMOM_NCM_30", nu_i[3, 0], -3.280259374880525e-06),
        ("IMOM_HU1", hu_i[0], 0.0008690300706446306),
        ("IMOM_HU2", hu_i[1], 3.7112807351259333e-08),
        ("IMOM_HU3", hu_i[2], 2.6788774435431954e-11),
        ("IMOM_HU4", hu_i[3], 2.8991603200922661e-12),
    ]:
        all_ok &= check(name, got, pinned, tol=1e-6)  # gtest tolerance; noise-level pins

    print("-- validation: OLD h5/h6 goldens equal the BUGGY formulas (they encoded the bug) --")
    b5_i, b6_i = hu_buggy_h5_h6(nu_i)
    all_ok &= check("IMOM_HU5(buggy)", b5_i, -2.1393783155778043e-23, tol=1e-6)
    all_ok &= check("IMOM_HU6(buggy)", b6_i, -2.3976723312959013e-06, tol=1e-6)

    print("-- corrected goldens for test_moments_skimage.h --")
    print(f"  HU_M5    = {hu_s[4]!r}   (was 4.598098572281346e-10 summation noise; oracle exact 0)")
    print(f"  IMOM_HU5 = {hu_i[4]!r}")
    print(f"  IMOM_HU6 = {hu_i[5]!r}")

    # ---------------------------------------------------------------- B. weighted (regression)
    print("\n=== B. weighted Hu snapshots from PINNED Nyxus eta (no external W-weighting oracle) ===")
    # WT_NORM_CTR_MOM_* pinned in test_moments_regression.h
    nu_w = nu_matrix_from_pinned({
        (0, 2): 0.017193451534902194,
        (0, 3): 0.00465210928951103,
        (1, 1): 0.0005158752090264311,
        (1, 2): 0.0008429820076480387,
        (2, 0): 0.032749715293974795,
        (2, 1): 0.0011795045377311554,
        (3, 0): 0.005694719149793572,
    })
    hu_w = hu_correct(nu_w)
    b5_w, b6_w = hu_buggy_h5_h6(nu_w)
    print("-- validation: unchanged weighted Hu 1-4,7 + old buggy 5,6 --")
    all_ok &= check("WEIGHTED_HU_M1", hu_w[0], 0.04994316682887699, tol=1e-6)
    all_ok &= check("WEIGHTED_HU_M2", hu_w[1], 0.00024306185106698786, tol=1e-6)
    all_ok &= check("WEIGHTED_HU_M3", hu_w[2], 1.1262214820995351e-05, tol=1e-6)
    all_ok &= check("WEIGHTED_HU_M4", hu_w[3], 7.674925625409561e-05, tol=1e-6)
    all_ok &= check("WEIGHTED_HU_M7", hu_w[6], -1.3078000499389243e-09, tol=1e-6)
    all_ok &= check("WEIGHTED_HU_M5(buggy)", b5_w, -4.0924793325555725e-09, tol=1e-6)
    all_ok &= check("WEIGHTED_HU_M6(buggy)", b6_w, 0.004652261067232659, tol=1e-6)
    print("-- corrected snapshots for test_moments_regression.h --")
    print(f"  WEIGHTED_HU_M5 = {hu_w[4]!r}")
    print(f"  WEIGHTED_HU_M6 = {hu_w[5]!r}")

    # IMOM_WNCM_* pinned in test_moments_regression.h
    nu_iw = nu_matrix_from_pinned({
        (0, 2): 0.00011186764375543395,
        (0, 3): 4.6265447913743816e-07,
        (1, 1): 9.098620671498624e-06,
        (1, 2): -4.1575256057220215e-08,
        (2, 0): 0.0001835830638933708,
        (2, 1): -1.4928319462526036e-08,
        (3, 0): 7.484780551139263e-07,
    })
    hu_iw = hu_correct(nu_iw)
    b5_iw, b6_iw = hu_buggy_h5_h6(nu_iw)
    all_ok &= check("IMOM_WHU1", hu_iw[0], 0.00029545070764880473, tol=1e-6)
    all_ok &= check("IMOM_WHU2", hu_iw[1], 5.6161730111679807e-09, tol=1e-6)
    all_ok &= check("IMOM_WHU3", hu_iw[2], 1.1923046864432925e-12, tol=1e-6)
    all_ok &= check("IMOM_WHU4", hu_iw[3], 3.1287168309169019e-12, tol=1e-6)
    all_ok &= check("IMOM_WHU7", hu_iw[6], -5.6202519491182231e-24, tol=1e-6)
    all_ok &= check("IMOM_WHU5(buggy)", b5_iw, -5.2494915279842729e-24, tol=1e-6)
    all_ok &= check("IMOM_WHU6(buggy)", b6_iw, 4.6265447915851515e-07, tol=1e-6)
    print("-- corrected snapshots for test_moments_regression.h --")
    print(f"  IMOM_WHU5 = {hu_iw[4]!r}")
    print(f"  IMOM_WHU6 = {hu_iw[5]!r}")

    # ---------------------------------------------------------------- C. asymmetric wedge
    print(f"\n=== C. thin right-wedge fixture (5*y <= x, {WEDGE_W}x{WEDGE_H}) for the new oracle test ===")
    tri = wedge_shape()
    m_t, mu_t, nu_t, hu_t = full_stack(tri)
    b5_t, b6_t = hu_buggy_h5_h6(nu_t)

    print("-- goldens for test_moments_skimage.h (wedge_shape_hu_skimage_golden_values) --")
    print(f"  area (SPAT_MOMENT_00)    = {m_t[0, 0]!r}")
    print(f"  SPAT_MOMENT_10           = {m_t[1, 0]!r}")
    print(f"  SPAT_MOMENT_01           = {m_t[0, 1]!r}")
    print(f"  CENTRAL_MOMENT_20        = {mu_t[2, 0]!r}")
    print(f"  CENTRAL_MOMENT_02        = {mu_t[0, 2]!r}")
    print(f"  CENTRAL_MOMENT_11        = {mu_t[1, 1]!r}")
    print(f"  CENTRAL_MOMENT_30        = {mu_t[3, 0]!r}")
    print(f"  CENTRAL_MOMENT_03        = {mu_t[0, 3]!r}")
    print(f"  CENTRAL_MOMENT_21        = {mu_t[2, 1]!r}")
    print(f"  CENTRAL_MOMENT_12        = {mu_t[1, 2]!r}")
    print(f"  NORM_CENTRAL_MOMENT_20   = {nu_t[2, 0]!r}")
    print(f"  NORM_CENTRAL_MOMENT_02   = {nu_t[0, 2]!r}")
    print(f"  NORM_CENTRAL_MOMENT_11   = {nu_t[1, 1]!r}")
    print(f"  NORM_CENTRAL_MOMENT_30   = {nu_t[3, 0]!r}")
    print(f"  NORM_CENTRAL_MOMENT_03   = {nu_t[0, 3]!r}")
    print(f"  NORM_CENTRAL_MOMENT_21   = {nu_t[2, 1]!r}")
    print(f"  NORM_CENTRAL_MOMENT_12   = {nu_t[1, 2]!r}")
    for k in range(7):
        print(f"  HU_M{k + 1}                    = {hu_t[k]!r}")

    print("-- discriminance: |buggy - correct| must exceed the gtest tolerance 1e-6*scale --")
    for name, buggy, correct in [("HU_M5", b5_t, hu_t[4]), ("HU_M6", b6_t, hu_t[5])]:
        scale = max(1.0, abs(buggy), abs(correct))
        tol = 1e-6 * scale
        disc = abs(buggy - correct)
        verdict = "DISCRIMINATES" if disc > 10 * tol else "TOO WEAK"
        print(f"  {name}: correct={correct!r} buggy={buggy!r} |diff|={disc:.3e} tol={tol:.3e} -> {verdict}")
        all_ok &= disc > 10 * tol

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not paste goldens'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
