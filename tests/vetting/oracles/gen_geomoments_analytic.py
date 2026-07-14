#!/usr/bin/env python3
"""Offline oracle generator for the 2D geometric-moment goldens in
``tests/test_2d_geometric_moments.h`` (see tests/vetting/SPEC.md: goldens are produced offline by a
checked-in generator and pinned as literals).

Oracle: closed-form / numpy reference (token ``analytic``) cross-checked against ``skimage.measure``
(``moments``, ``moments_central``, ``moments_normalized``, ``moments_hu``). The two agree to double
precision on this fixture, so numpy is used as the primary (dependency-free) reference.

Fixture (identical pixels to the C++ test ``load_geomoment_fixture``): a full 48x40 rectangle with
    intensity(x, y) = floor(10 + 3*x + 5*y + (x*y) % 7)              # PixIntens truncation to int
Shape moments weight every pixel by 1 (binary mask); intensity moments weight by the intensity above.

Nyxus conventions this reproduces (verified against the pre-fix raw-moment goldens, which the
centroid-truncation bug did NOT affect):
  * SPAT_MOMENT_pq / IMOM_RM_pq = sum( I * x^p * y^q )         (p = x-power, q = y-power; origin at aabb min = (0,0))
  * central moment  mu_pq       = sum( I * (x-cx)^p * (y-cy)^q ), cx=m10/m00, cy=m01/m00   (TRUE fractional centroid)
  * normalized central          eta_pq = mu_pq / m00**((p+q)/2 + 1)
  * Hu moments                  the 7 standard invariants of eta (matches Nyxus HU_M1..7 exactly)

The centroid-truncation bug (fixed in 2d_geomoments_basic.cpp) used int(cx)/int(cy) here, which is why
the first central moments mu10/mu01 came out non-zero (they are 0 by definition about the true centroid).

Run:  python gen_geomoments_analytic.py     # prints NAME value lines for pasting into the golden vectors
"""
import numpy as np

W, H = 48, 40
X, Y = np.meshgrid(np.arange(W), np.arange(H))
Xf, Yf = X.astype(float), Y.astype(float)

FIXTURES = {
    "SHAPE": np.ones((H, W)),
    "INTEN": np.floor(10.0 + 3.0 * Xf + 5.0 * Yf + ((X * Y) % 7)),
}

def raw(I, p, q):
    return float(np.sum(I * Xf**p * Yf**q))

def central(I, cx, cy, p, q):
    return float(np.sum(I * (Xf - cx)**p * (Yf - cy)**q))

def hu(eta):
    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    a, b = n30 + n12, n21 + n03
    return [
        n20 + n02,
        (n20 - n02)**2 + 4*n11**2,
        (n30 - 3*n12)**2 + (3*n21 - n03)**2,
        a**2 + b**2,
        (n30 - 3*n12)*a*(a**2 - 3*b**2) + (3*n21 - n03)*b*(3*a**2 - b**2),
        (n20 - n02)*(a**2 - b**2) + 4*n11*a*b,
        (3*n21 - n03)*a*(a**2 - 3*b**2) - (n30 - 3*n12)*b*(3*a**2 - b**2),
    ]

for name, I in FIXTURES.items():
    m00 = np.sum(I)
    cx, cy = raw(I, 1, 0) / m00, raw(I, 0, 1) / m00
    eta = lambda p, q: central(I, cx, cy, p, q) / (m00 ** ((p + q) / 2 + 1))
    print(f"# ===== {name}  centroid=({cx:.6f},{cy:.6f})  m00={m00:g} =====")
    orders = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3)]
    for p, q in orders:
        print(f"CENTRAL_{p}{q} {central(I, cx, cy, p, q):.17g}")
    for hu_i, val in enumerate(hu(eta), 1):
        print(f"HU_M{hu_i} {val:.17g}")

# ---- asymmetric L-shape (binary): 30x20 rectangle minus the top-right corner (x>=18 & y>=12).
# Every central-moment order is a distinct non-zero value (except the definitional mu10=mu01=0), so the
# centroid-truncation bug is exposed at ALL orders. Goldens for the C++ test
# oracle_analytic_asymmetric_shape_central_moment_golden_values.
LW, LH = 30, 20
lmask = np.array([[not (x >= 18 and y >= 12) for x in range(LW)] for y in range(LH)])
lys, lxs = np.nonzero(lmask)
lm00 = float(len(lxs))
lcx, lcy = lxs.mean(), lys.mean()
def lcentral(p, q):
    return float(np.sum((lxs - lcx)**p * (lys - lcy)**q))
print(f"# ===== ASYM_LSHAPE  pixels={int(lm00)}  centroid=({lcx:.6f},{lcy:.6f}) =====")
for p, q in orders:
    print(f"CENTRAL_{p}{q} {lcentral(p, q):.17g}")
for p, q in [(0,2),(0,3),(1,1),(1,2),(2,0),(2,1),(3,0)]:
    print(f"NORM_CENTRAL_{p}{q} {lcentral(p, q) / (lm00 ** ((p + q) / 2 + 1)):.17g}")
