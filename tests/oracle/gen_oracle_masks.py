"""Generate the analytic oracle masks (known fractal dimension) as 8-bit TIFFs, for
cross-checking nyxus' FRACT_DIM_BOXCOUNT / FRACT_DIM_PERIMETER against ImageJ/FracLac.

  square256        -> box-count D = 2.000
  line256          -> box-count D = 1.000
  sierpinski_tri   -> box-count D = log2(3) = 1.585
  sierpinski_carpet-> box-count D = log8/log3 = 1.893
  koch_curve       -> perimeter/divider D = log4/log3 = 1.262

Then run the shifting-grid oracle headlessly:
  <fiji> --headless -macro tests/oracle/shiftgrid_boxcount.ijm "<dir with these tifs>"

Requires numpy + Pillow.
"""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "masks"
OUT.mkdir(exist_ok=True)


def save(arr, name):
    Image.fromarray((np.asarray(arr) > 0).astype(np.uint8) * 255, "L").save(OUT / name)
    print(f"{name:22s} {arr.shape[1]}x{arr.shape[0]}")


save(np.ones((256, 256), bool), "square256.tif")

ln = np.zeros((256, 256), bool); ln[128, :] = True
save(ln, "line256.tif")

N = 256
tri = np.zeros((N, N), bool)
for y in range(N):
    tri[y] = ((np.arange(N) & y) == 0)
save(tri, "sierpinski_tri.tif")


def carpet(order):
    size = 3 ** order
    a = np.ones((size, size), bool)
    def punch(x, y, s):
        if s < 3:
            return
        t = s // 3
        a[y + t:y + 2 * t, x + t:x + 2 * t] = False
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                punch(x + dx * t, y + dy * t, t)
    punch(0, 0, size)
    return a
save(carpet(5), "sierpinski_carpet.tif")


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

S = 729
img = Image.new("L", (S, S), 0)
poly = koch((20.0, S / 2), (S - 20.0, S / 2), 6) + [(S - 20.0, S / 2)]
ImageDraw.Draw(img).line(poly, fill=255, width=1)
img.save(OUT / "koch_curve.tif")
print(f"{'koch_curve.tif':22s} {S}x{S}")
