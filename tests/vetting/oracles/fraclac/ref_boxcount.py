"""Transparent reference box-counting dimension, second independent oracle.

Two estimators reported per mask:
  D_ls  = -slope of a LEAST-SQUARES fit of log(count) vs log(box)   (standard, robust)
  D_end = -endpoint slope (first vs last box)                        (what nyxus' mean-of-
          local-slopes collapses to under uniform log-spacing)

Box sizes are powers of two (like nyxus' Power2Padded box counter): the image is padded
up to a square power-of-two canvas, then boxes of size 1,2,4,...,W/2 count how many tiles
contain any foreground pixel.
"""
import sys, math
from pathlib import Path
import numpy as np
from PIL import Image

def boxcount_pow2(mask):
    H, W = mask.shape
    n = 1 << (max(H, W) - 1).bit_length()          # next power of two
    pad = np.zeros((n, n), bool)
    pad[:H, :W] = mask
    sizes, counts = [], []
    s = n
    while s > 1:                                    # nyxus loops s=width; s>1; s/=2
        nt = n // s
        cnt = 0
        for r in range(nt):
            for c in range(nt):
                if pad[r*s:(r+1)*s, c*s:(c+1)*s].any():
                    cnt += 1
        sizes.append(s); counts.append(cnt)
        s //= 2
    return np.array(sizes, float), np.array(counts, float)

def dims(sizes, counts):
    x = np.log(sizes); y = np.log(counts)
    # least-squares slope of log(count) vs log(size); D = -slope
    A = np.vstack([x, np.ones_like(x)]).T
    slope_ls = np.linalg.lstsq(A, y, rcond=None)[0][0]
    # endpoint slope (uses only extreme scales)
    slope_end = (y[-1] - y[0]) / (x[-1] - x[0])
    return -slope_ls, -slope_end

EXPECT = {
    "square256.tif": 2.0, "line256.tif": 1.0,
    "sierpinski_tri.tif": math.log(3)/math.log(2),
    "sierpinski_carpet.tif": math.log(8)/math.log(3),
    "koch_curve.tif": math.log(4)/math.log(3),
    "nyxus_roi.tif": None,
}

masks = Path(sys.argv[1] if len(sys.argv) > 1 else "masks")
print(f"{'mask':22s} {'D_leastsq':>10s} {'D_endpt':>9s} {'expected':>9s}  boxes(size:count)")
for f in sorted(masks.glob("*.tif")):
    m = np.asarray(Image.open(f).convert("L")) > 127
    sizes, counts = boxcount_pow2(m)
    d_ls, d_end = dims(sizes, counts)
    exp = EXPECT.get(f.name)
    exps = f"{exp:9.4f}" if exp else "     n/a "
    trail = " ".join(f"{int(s)}:{int(c)}" for s, c in zip(sizes, counts))
    print(f"{f.name:22s} {d_ls:10.4f} {d_end:9.4f} {exps}  {trail}")
