"""OFFLINE analytic oracle for 2D ZERNIKE2D (SPEC 4, oracle=analytic).

Computes the 30 Zernike moment magnitudes of order <= 9 from the closed form --
the standard factorial series for the radial polynomial

    R_nm(r) = sum_{k=0}^{(n-m)/2}  (-1)^k (n-k)! r^(n-2k)
                                   -----------------------------------------
                                   k! ((n+m)/2 - k)! ((n-m)/2 - k)!

    A_nm    = (n+1)/pi * sum_pixels  (I / sum I) * R_nm(r) * e^(-i m theta)

-- evaluated on the gtest fixture `shape2d_morphology_{intensity,mask}` (test_data.h)
at the geometry ZernikeFeature uses: the unit disk centred on the ROI's intensity
centroid with radius min(bbox width, bbox height), pixels outside the disk dropped.

This is an INDEPENDENT implementation of the same mathematics, not a second copy of
Nyxus' procedure: `zernike.cpp` evaluates R_nm through the Singh & Walia three-term
recurrence, this file sums the factorial series directly. Agreement therefore says the
recurrence is right; it is not circular (revet.txt SPEC 5.2).

Two optional cross-checks run when a CellProfiler environment is available (see
tests/vetting/TOOLS.md); neither is needed for the goldens:

  --centrosome   evaluate centrosome.zernike's own polynomials at the same geometry, a
                 third independent implementation (lookup-table factorial series).
  --cellprofiler run CellProfiler's MeasureObjectIntensityDistribution Zernikes and print
                 the divergence. Its numbers are NOT comparable: CellProfiler centres the
                 unit disk on the minimum enclosing circle and normalises by pixel count,
                 where Nyxus centres on the intensity centroid and normalises by total
                 intensity. That is a convention difference, not a disagreement about the
                 moments; the report records it so the candidate oracle is closed honestly.

Nothing here carries a literal copy of anything it checks: the fixture is parsed out of
tests/test_data.h, the pinned tables out of test_2d_zernike_{analytic,regression}.h, and
the geometry out of test_2d_zernike_mechanics.h.

Provenance: closed form, stdlib only, no pinned tool version. Cross-checked against
centrosome 1.2.3 / cellprofiler 4.2.8 (python 3.9). generator=tests/vetting/oracles/
gen_zernike_analytic.py. Run offline.
"""
import argparse
import cmath
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", ".."))

ORDER = 9          # ZernikeFeature::ZERNIKE2D_ORDER
NVALS = 30         # ZernikeFeature::NUM_FEATURE_VALS

# the band test_2d_zernike_analytic.h asserts, and the floor that entry (1,1) -- a mathematical
# zero -- needs so it is not compared relatively at the noise floor
REL_TOL = 1e-12
ABS_FLOOR = 1e-15


# ---------------------------------------------------------------------------- parsing

def _read(rel):
    with open(os.path.join(TESTS, rel), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _braced_body(src, start):
    """The text between the brace at/after `start` and its match, by counting braces.

    A non-greedy regex eats the last entry's closing brace and silently drops it.
    """
    i = src.index("{", start)
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
        j += 1
    raise SystemExit("unbalanced braces while parsing a reference table")


def parse_fixture():
    src = _read("test_data.h")
    out = {}
    for name in ("shape2d_morphology_intensity", "shape2d_morphology_mask"):
        m = re.search(r"\b%s\s*\[\s*\]\s*=\s*\{" % name, src)
        if not m:
            raise SystemExit("cannot find %s in test_data.h" % name)
        body = _braced_body(src, m.end() - 1)
        out[name] = [tuple(int(v) for v in t)
                     for t in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
    return out


def parse_table(relpath, table_name):
    src = _read(relpath)
    m = re.search(r"%s\s*(?:=\s*)?\{" % re.escape(table_name), src)
    if not m:
        raise SystemExit("cannot find %s in %s" % (table_name, relpath))
    body = _braced_body(src, m.end() - 1)
    for name, values in re.findall(r'\{\s*"(\w+)"\s*,\s*\{([^}]*)\}', body):
        if name == "ZERNIKE2D":
            return [float(v) for v in re.findall(r"[-+0-9.eE]+", values)]
    raise SystemExit("no ZERNIKE2D entry in %s" % table_name)


def parse_mechanics():
    """The bounding box and disk radius pinned in test_2d_zernike_mechanics.h."""
    src = _read("test_2d_zernike_mechanics.h")
    # the assertions carry a streamed failure message, so stop at the closing paren rather than
    # requiring a semicolon straight after it
    w = re.search(r"ASSERT_EQ\(cols,\s*(\d+)\)", src)
    h = re.search(r"ASSERT_EQ\(rows,\s*(\d+)\)", src)
    r = re.search(r"ASSERT_DOUBLE_EQ\(rad,\s*([0-9.]+)\)", src)
    if not (w and h and r):
        raise SystemExit("cannot find the geometry pins in test_2d_zernike_mechanics.h")
    return int(w.group(1)), int(h.group(1)), float(r.group(1))


# ------------------------------------------------------------------------- the closed form

def zernike_indexes(order=ORDER):
    """(n, m) in the order zernike.cpp emits them: n ascending, m ascending, n-m even."""
    return [(n, m) for n in range(order + 1) for m in range(n + 1) if (n - m) % 2 == 0]


def radial_poly(n, m, r):
    """R_nm(r) summed straight from the factorial series."""
    s = 0.0
    for k in range((n - m) // 2 + 1):
        s += ((-1) ** k * math.factorial(n - k) * r ** (n - 2 * k)
              / (math.factorial(k)
                 * math.factorial((n + m) // 2 - k)
                 * math.factorial((n - m) // 2 - k)))
    return s


def analytic_moments(I, cols, rows, rad):
    """|A_nm| for the 30 indexes, at ZernikeFeature's geometry."""
    total = sum(I[j][i] for j in range(rows) for i in range(cols))
    m10 = sum((i + 1) * I[j][i] for j in range(rows) for i in range(cols))
    m01 = sum((j + 1) * I[j][i] for j in range(rows) for i in range(cols))
    cx, cy = m10 / total, m01 / total

    idx = zernike_indexes()
    acc = [0j] * len(idx)
    inside = 0
    for i in range(cols):
        x = (i + 1 - cx) / rad
        for j in range(rows):
            y = (j + 1 - cy) / rad
            r = math.hypot(x, y)
            if r < 2.220446049250313e-16 or r > 1.0:
                continue
            inside += 1
            theta = math.atan2(y, x)
            w = I[j][i] / total
            for k, (n, m) in enumerate(idx):
                acc[k] += w * radial_poly(n, m, r) * cmath.exp(-1j * m * theta)
    return [abs(acc[k]) * (n + 1) / math.pi for k, (n, m) in enumerate(idx)], (cx, cy), inside


# -------------------------------------------------------------------------------- main

def pin_checks(pinned, label):
    """Range and index checks over the PINNED LITERALS, not over a fresh run.

    The C++ invariant tests assert these of the computed values; this asserts them of what the
    header actually says, so an edit to a table is caught without a rebuild.
    """
    bad = []
    if len(pinned) != NVALS:
        return ["%s pins %d values, expected %d" % (label, len(pinned), NVALS)]
    for k, (n, m) in enumerate(zernike_indexes()):
        if pinned[k] < 0.0:
            bad.append("%s[%d] = A(%d,%d) is negative (%r); it is a modulus"
                       % (label, k, n, m, pinned[k]))
        # |R_nm| <= 1 on the unit disk and the weights sum to at most 1, so |A_nm| <= (n+1)/pi
        bound = (n + 1) / math.pi
        if pinned[k] > bound * (1 + 1e-12):
            bad.append("%s[%d] = A(%d,%d) is %r, above its bound (n+1)/pi = %r"
                       % (label, k, n, m, pinned[k], bound))
    if abs(pinned[0] - 1.0 / math.pi) > 1e-14:
        bad.append("%s[0] = A(0,0) is %.17g, must be 1/pi = %.17g"
                   % (label, pinned[0], 1.0 / math.pi))
    if abs(pinned[1]) > 1e-14:
        bad.append("%s[1] = A(1,1) is %.3g, must vanish (first moment about the centroid)"
                   % (label, pinned[1]))
    return bad


def rel(a, b):
    if a == b:
        return 0.0
    if b == 0.0:
        return float("inf")
    return abs(a - b) / abs(b)


def build_image(fixture):
    inten = {(x, y): v for x, y, v in fixture["shape2d_morphology_intensity"]}
    pts = [(x, y) for x, y, v in fixture["shape2d_morphology_mask"] if v]
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    cols = max(p[0] for p in pts) - x0 + 1
    rows = max(p[1] for p in pts) - y0 + 1
    I = [[0.0] * cols for _ in range(rows)]
    for x, y in pts:
        I[y - y0][x - x0] = float(inten[(x, y)])
    return I, cols, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centrosome", action="store_true",
                    help="cross-check against centrosome's own Zernike polynomials")
    ap.add_argument("--cellprofiler", action="store_true",
                    help="run CellProfiler's Zernikes and print the convention divergence")
    a = ap.parse_args()

    fixture = parse_fixture()
    I, cols, rows = build_image(fixture)
    pin_cols, pin_rows, rad = parse_mechanics()
    if (cols, rows) != (pin_cols, pin_rows):
        raise SystemExit("fixture bbox %dx%d disagrees with the mechanics pins %dx%d"
                         % (cols, rows, pin_cols, pin_rows))

    ref, (cx, cy), inside = analytic_moments(I, cols, rows, rad)
    print("fixture bbox %dx%d, disk radius %g, intensity centroid (%.15g, %.15g)"
          % (cols, rows, rad, cx, cy))
    print("%d of %d bounding-box pixels fall inside the unit disk\n" % (inside, cols * rows))

    idx = zernike_indexes()
    ok = True

    # 1. the two magnitudes the closed form forces, independent of any implementation
    print("=== the two values the closed form forces ===")
    z00_expected = 1.0 / math.pi
    good = rel(ref[0], z00_expected) < 1e-14
    ok &= good
    print("  %s Z(0,0) = %.17g, must be 1/pi = %.17g (weights sum to 1 inside the disk)"
          % ("OK  " if good else "FAIL", ref[0], z00_expected))
    good = abs(ref[1]) < 1e-14
    ok &= good
    print("  %s Z(1,1) = %.3g, must vanish (it is the first moment about the centroid)"
          % ("OK  " if good else "FAIL", ref[1]))

    # 2. the analytic goldens against the header that pins them
    print("\n=== the analytic table against test_2d_zernike_analytic.h ===")
    pinned = parse_table("test_2d_zernike_analytic.h", "zernike_2d_analytic_ref_vals")
    if len(pinned) != NVALS:
        print("  FAIL header pins %d values, expected %d" % (len(pinned), NVALS))
        ok = False
    else:
        # The pinned table is ONE evaluation of the closed form. A different interpreter or libm
        # sums the same terms in the same order but not to the same last bit, so this is banded at
        # the tolerance the header itself asserts rather than demanded bit-exact -- requiring
        # bit-equality here made the generator fail on a second Python build for no defect at all.
        worst = 0.0
        for k, (n, m) in enumerate(idx):
            tol = abs(pinned[k]) * REL_TOL + ABS_FLOOR
            d = abs(ref[k] - pinned[k])
            if abs(pinned[k]) > ABS_FLOOR:
                worst = max(worst, rel(ref[k], pinned[k]))
            if d > tol:
                print("  FAIL (n=%d,m=%d) header=%.17g computed=%.17g diff=%.3g tol=%.3g"
                      % (n, m, pinned[k], ref[k], d, tol))
                ok = False
        exact = sum(1 for k in range(NVALS) if ref[k] == pinned[k])
        print("  all %d within rel=%g + abs=%g of a fresh evaluation (%d bit-exact, worst rel %.3g)"
              % (NVALS, REL_TOL, ABS_FLOOR, exact, worst))

    # 3. Nyxus' own snapshot against the closed form -- the vetting claim
    print("\n=== Nyxus' pinned output against the closed form ===")
    nyx = parse_table("test_2d_zernike_regression.h", "zernike_2d_regression_ref_vals")
    if len(nyx) != NVALS:
        print("  FAIL regression header pins %d values, expected %d" % (len(nyx), NVALS))
        ok = False
    else:
        worst, worst_at = 0.0, None
        for k, (n, m) in enumerate(idx):
            # (1,1) is a mathematical zero; compare it absolutely
            tol = abs(ref[k]) * REL_TOL + ABS_FLOOR
            if abs(ref[k]) <= ABS_FLOOR:
                mark, r_ = "abs", abs(nyx[k] - ref[k])
            else:
                mark, r_ = "rel", rel(nyx[k], ref[k])
                if r_ > worst:
                    worst, worst_at = r_, (n, m)
            bad = abs(nyx[k] - ref[k]) > tol
            if bad:
                print("  FAIL (n=%d,m=%d) nyxus=%.17g analytic=%.17g %s=%.3g"
                      % (n, m, nyx[k], ref[k], mark, r_))
                ok = False
        print("  worst relative residual %.3g at (n=%d,m=%d); Z(1,1) matches absolutely to %.3g"
              % (worst, worst_at[0], worst_at[1], abs(nyx[1] - ref[1])))

    # 4. the reverse direction -- an index the closed form produces that nothing pins
    if len(idx) != NVALS:
        print("  FAIL the closed form produces %d indexes, the headers pin %d" % (len(idx), NVALS))
        ok = False

    # 5. range and identity checks over the pinned literals in BOTH headers
    print("\n=== range and identity checks over the pinned values ===")
    bad = (pin_checks(pinned, "analytic") + pin_checks(nyx, "regression"))
    if bad:
        for b in bad:
            print("  FAIL", b)
        ok = False
    else:
        print("  both tables: every magnitude non-negative and within its (n+1)/pi bound,")
        print("  A(0,0) = 1/pi and A(1,1) = 0 in each")

    if a.centrosome:
        ok &= cross_centrosome(I, cols, rows, rad, cx, cy, ref, idx)
    if a.cellprofiler:
        report_cellprofiler(fixture, nyx, idx)

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- do not ship"))
    return 0 if ok else 1


def cross_centrosome(I, cols, rows, rad, cx, cy, ref, idx):
    """A third implementation of the same polynomials, from CellProfiler's library."""
    import numpy as np
    from centrosome.zernike import construct_zernike_polynomials, get_zernike_indexes

    arr = np.array(I)
    tot = arr.sum()
    jj, ii = np.mgrid[0:rows, 0:cols]
    X = (ii + 1 - cx) / rad
    Y = (jj + 1 - cy) / rad
    cidx = get_zernike_indexes(ORDER + 1)
    if [tuple(t) for t in cidx] != idx:
        print("\n  FAIL centrosome's index order differs from zernike.cpp's")
        return False
    Z = construct_zernike_polynomials(X, Y, cidx)
    mag = [abs(((arr / tot) * Z[:, :, k]).sum()) * (n + 1) / math.pi
           for k, (n, m) in enumerate(cidx)]
    worst = max(rel(mag[k], ref[k]) for k in range(NVALS) if abs(ref[k]) > 1e-14)
    print("\n=== the closed form against centrosome's own polynomials ===")
    print("  worst relative difference over the 29 non-zero magnitudes: %.3g" % worst)
    print("  (both evaluated at Nyxus' geometry, so only the polynomial is under test)")
    if worst > 1e-12:
        print("  FAIL the two implementations disagree")
        return False
    return True


def report_cellprofiler(fixture, nyx, idx):
    """CellProfiler's own Zernikes -- a different convention, recorded not asserted."""
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import cellprofiler_core.preferences as cpprefs
    cpprefs.set_headless()
    import cellprofiler_core.image as cpi
    import cellprofiler_core.measurement as cpmeas
    import cellprofiler_core.object as cpo
    import cellprofiler_core.pipeline as cpp
    import cellprofiler_core.workspace as cpw
    from cellprofiler.modules import measureobjectintensitydistribution as moid

    inten = {(x, y): v for x, y, v in fixture["shape2d_morphology_intensity"]}
    pts = [(x, y) for x, y, v in fixture["shape2d_morphology_mask"] if v]
    grid = 1 + max(max(x for x, _, _ in fixture["shape2d_morphology_intensity"]),
                   max(y for _, y, _ in fixture["shape2d_morphology_intensity"]))
    PAD = 3
    lab = np.zeros((grid + 2 * PAD, grid + 2 * PAD), dtype=np.int32)
    img = np.zeros((grid + 2 * PAD, grid + 2 * PAD), dtype=float)
    for x, y in pts:
        lab[y + PAD, x + PAD] = 1
        img[y + PAD, x + PAD] = inten[(x, y)] / 255.0

    module = moid.MeasureObjectIntensityDistribution()
    module.images_list.value = "img"
    module.objects[0].object_name.value = "objs"
    module.bin_counts[0].bin_count.value = 1
    module.wants_zernikes.value = moid.Z_MAGNITUDES
    module.zernike_degree.value = ORDER

    objects = cpo.Objects()
    objects.segmented = lab
    oset = cpo.ObjectSet()
    oset.add_objects(objects, "objs")
    imgset = cpi.ImageSetList().get_image_set(0)
    imgset.add("img", cpi.Image(img))
    meas = cpmeas.Measurements()
    module.run(cpw.Workspace(cpp.Pipeline(), module, imgset, oset, meas, None))

    print("\n=== CellProfiler's own Zernikes (different convention, recorded not asserted) ===")
    print("  CP centres the disk on the minimum enclosing circle and divides by pixel count;")
    print("  Nyxus centres on the intensity centroid and divides by total intensity.")
    print("  %-8s %22s %22s %10s" % ("(n,m)", "nyxus", "cellprofiler", "rel"))
    diverging = 0
    for k, (n, m) in enumerate(idx):
        v = meas.get_current_measurement(
            "objs", "RadialDistribution_ZernikeMagnitude_img_%d_%d" % (n, m))[0]
        r_ = rel(v, nyx[k])
        if r_ > 1e-2:
            diverging += 1
        print("  (%d,%d)%3s %22.17g %22.17g %10.3g" % (n, m, "", nyx[k], v, r_))
    print("  %d of %d disagree by more than 1%% -- the convention gap, not a moment disagreement"
          % (diverging, NVALS))


if __name__ == "__main__":
    raise SystemExit(main())
