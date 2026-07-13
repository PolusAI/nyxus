"""Oracle validation of the 2D fractal-dimension features on the production featurize() path.

FRACT_DIM_BOXCOUNT and FRACT_DIM_PERIMETER are checked two ways:
  - analytic shapes with a known dimension: box-count square 2.0, line 1.0, Sierpinski log2(3);
    perimeter disk 1.0, Koch snowflake log4/log3.
  - a large arbitrary ROI (tests/data/fractal_blob512_seg.ome.tif) vs offline ImageJ/FracLac
    oracles: box count of the filled ROI (same method), and box count of its edge vs the
    Richardson divider perimeter (cross-method, same boundary dimension).
"""
import math
from pathlib import Path
import numpy as np
import pytest
import nyxus


# ----------------------------- helpers --------------------------------------
def _run(features, inten, label, **kw):
    nyx = nyxus.Nyxus(features=features, n_feature_calc_threads=1, **kw)
    df = nyx.featurize(inten.astype(np.float64), label.astype(np.uint32),
                       intensity_names=["i"], label_names=["l"])
    return df


def _fd(label):
    """Run nyxus on a binary mask and return (box-count D, perimeter D)."""
    inten = label.astype(np.float64) * 100.0
    row = _run(["*ALL_MORPHOLOGY*"], inten, label).iloc[0]
    bc = [c for c in row.index if c.endswith("FRACT_DIM_BOXCOUNT")][0]
    pf = [c for c in row.index if c.endswith("FRACT_DIM_PERIMETER")][0]
    return float(row[bc]), float(row[pf])


# ----------------------------- analytic shapes ------------------------------
def _square(n=256):
    return np.ones((n, n), np.uint32)


def _line(n=256):
    m = np.zeros((n, n), np.uint32)
    m[n // 2, :] = 1
    return m


def _sierpinski_triangle(order=8):
    """(x & y) == 0 bit pattern -> Sierpinski triangle, box-count D = log2(3) = 1.585."""
    n = 1 << order
    rows = np.arange(n)
    m = np.zeros((n, n), np.uint32)
    for y in range(n):
        m[y] = ((rows & y) == 0).astype(np.uint32)
    return m


def _disk(r=120, pad=12):
    """Filled disk: smooth boundary -> perimeter (divider) dimension -> 1.0."""
    n = 2 * (r + pad)
    yy, xx = np.ogrid[:n, :n]
    c = r + pad
    return (((xx - c) ** 2 + (yy - c) ** 2) <= r * r).astype(np.uint32)


def _koch_snowflake(depth=5, size=1400):
    """Filled Koch snowflake (pure numpy, no external deps): boundary divider
    dimension = log4/log3 = 1.2619."""
    def koch(p1, p2, d):
        if d == 0:
            return [p1]
        (x1, y1), (x2, y2) = p1, p2
        ax, ay = x1 + (x2 - x1) / 3.0, y1 + (y2 - y1) / 3.0
        cx, cy = x1 + 2 * (x2 - x1) / 3.0, y1 + 2 * (y2 - y1) / 3.0
        ang = math.atan2(y2 - y1, x2 - x1) - math.pi / 3.0
        L = math.hypot(x2 - x1, y2 - y1) / 3.0
        px, py = ax + L * math.cos(ang), ay + L * math.sin(ang)
        return (koch(p1, (ax, ay), d - 1) + koch((ax, ay), (px, py), d - 1)
                + koch((px, py), (cx, cy), d - 1) + koch((cx, cy), p2, d - 1))

    m = size * 0.1
    s = size - 2 * m
    h = s * math.sqrt(3) / 2.0
    A = (m, m + h * 2 / 3)
    B = (m + s, m + h * 2 / 3)
    C = (m + s / 2, m + h * 2 / 3 - h)
    poly = (koch(A, B, depth) + koch(B, C, depth) + koch(C, A, depth))
    # even-odd scanline polygon fill (numpy only)
    P = np.array(poly, float)
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    out = np.zeros((size, size), np.uint32)
    for row in range(size):
        cond = ((y0 <= row) & (y1 > row)) | ((y1 <= row) & (y0 > row))
        idx = np.where(cond)[0]
        if idx.size == 0:
            continue
        tt = (row - y0[idx]) / (y1[idx] - y0[idx])
        xs = np.sort(x0[idx] + tt * (x1[idx] - x0[idx]))
        for i in range(0, len(xs) - 1, 2):
            a = int(np.ceil(xs[i]))
            b = int(np.floor(xs[i + 1]))
            if b >= a:
                out[row, max(0, a):min(size, b + 1)] = 1
    return out


# ============================ box-count oracle ==============================
@pytest.mark.parametrize("name,mask,truth", [
    ("filled_square",       _square(),               2.0),
    ("straight_line",       _line(),                 1.0),
    ("sierpinski_triangle", _sierpinski_triangle(),  math.log(3) / math.log(2)),
])
def test_boxcount_recovers_known_dimension(name, mask, truth):
    """nyxus FRACT_DIM_BOXCOUNT must recover the analytic box-counting dimension
    (matches the ImageJ/FracLac oracle). Before the grid-alignment fix the square read
    ~1.75 and the Sierpinski triangle ~1.39; both now land on truth."""
    bc, _ = _fd(mask)
    assert abs(bc - truth) < 0.1, f"{name}: box-count {bc:.3f} vs truth {truth:.3f}"


# ============================ perimeter oracle ==============================
def test_perimeter_disk_is_smooth():
    """A disk has a smooth (non-fractal) boundary -> divider dimension -> 1.0."""
    _, pf = _fd(_disk())
    assert abs(pf - 1.0) < 0.05, f"disk perimeter D {pf:.3f} vs 1.0"


def test_perimeter_koch_snowflake():
    """Koch snowflake boundary: divider dimension = log4/log3 = 1.2619."""
    truth = math.log(4) / math.log(3)
    _, pf = _fd(_koch_snowflake())
    assert abs(pf - truth) < 0.05, f"Koch perimeter D {pf:.3f} vs truth {truth:.3f}"


# ============================ arbitrary ROI ================================
# tests/data/fractal_blob512_seg.ome.tif: a large (512x512) irregular ROI. Offline ImageJ/FracLac:
#   box-count = box count of the filled ROI (same method as nyxus' large-ROI single grid) -> 1.8706
#   perimeter = box count of the ROI edge (cross-method vs nyxus' Richardson divider)     -> 1.0493
ORACLE_BOXCOUNT = 1.8706        # vetted by ImageJ/FracLac: an independent implementation of the same box-count method
ORACLE_PERIMETER_EDGE = 1.0493  # vetted by ImageJ/FracLac edge box-count: cross-method vs nyxus' divider
_FIXTURE = str(Path(__file__).resolve().parent.parent / "data" / "fractal_blob512_seg.ome.tif")


def test_arbitrary_roi_matches_oracle():
    """Large arbitrary ROI read from the committed OME-TIFF (production file path):
      - box-count matches the same-method box-count oracle tightly (1%);
      - the divider perimeter matches the cross-method edge box-count oracle (10%)."""
    nyx = nyxus.Nyxus(features=["*ALL_MORPHOLOGY*"], n_feature_calc_threads=1)
    row = nyx.featurize_files(intensity_files=[_FIXTURE], mask_files=[_FIXTURE],
                              single_roi=False).iloc[0]
    bc = float(row[[c for c in row.index if c.endswith("FRACT_DIM_BOXCOUNT")][0]])
    pf = float(row[[c for c in row.index if c.endswith("FRACT_DIM_PERIMETER")][0]])
    assert abs(bc - ORACLE_BOXCOUNT) < ORACLE_BOXCOUNT * 0.01, \
        f"box-count {bc:.4f} vs same-method oracle {ORACLE_BOXCOUNT}"
    assert abs(pf - ORACLE_PERIMETER_EDGE) < ORACLE_PERIMETER_EDGE * 0.10, \
        f"perimeter {pf:.4f} vs cross-method edge-box-count oracle {ORACLE_PERIMETER_EDGE}"
