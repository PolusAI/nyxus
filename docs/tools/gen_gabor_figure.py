"""Draw docs/source/Math/gabor_gamma.png, the Gabor kernel figure of the Gabor feature page.

    python docs/tools/gen_gabor_figure.py [output.png]

The kernel is a line-for-line port of GaborFeature::Gabor, so the figure shows what Nyxus computes.
"""
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def gabor(f0, sig2lam, gamma, theta, fi, n):
    """Line-for-line port of GaborFeature::Gabor (src/nyx/features/gabor.cpp:393-449)."""
    lam = 2 * np.pi / f0
    sig = sig2lam * lam
    t = np.arange(n) - ((n - 1) // 2 if n % 2 else n // 2)
    ty, tx = np.meshgrid(t, t, indexing="ij")          # row index y, column index x
    xte = tx * np.cos(theta) + ty * np.sin(theta)
    yte = ty * np.cos(theta) - tx * np.sin(theta)
    ge = np.exp(-(xte**2 + gamma**2 * yte**2) / (2 * sig**2))
    re, im = ge * np.cos(xte * f0 + fi), ge * np.sin(xte * f0 + fi)
    s = np.sqrt(re**2 + im**2).sum()
    return re / s, im / s

theta, f0, n = 3 * np.pi / 4, np.pi / 4, 64
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=110)
for ax, g in zip(axes, (0.25, 0.5, 1.0)):
    re, _ = gabor(f0, 0.8, g, theta, 0.0, n)
    v = np.abs(re).max()
    im = ax.imshow(re, cmap="gray", vmin=-v, vmax=v, origin="upper")
    ax.set_title(f"$\\gamma$ = {g:g}")
    ax.set_xlabel("column $x$"); ax.set_ylabel("row $y$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Real part of the Nyxus Gabor kernel, $\\theta = 3\\pi/4$, $f = \\pi/4$, sig2lam = 0.8, 64 x 64 taps", fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "docs/source/Math/gabor_gamma.png", facecolor="white")
