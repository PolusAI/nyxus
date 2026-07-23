"""OFFLINE scikit-image oracle for the 2D GABOR feature (SPEC 4, oracle=skimage), on the
4 DSB2018 test ROIs (tests/test_dsb2018_data.h). Validates the goldens in
tests/test_gabor_truth.h.

What makes this an independent oracle (not a re-encoding of Nyxus):
  1. KERNEL: Nyxus' Gabor kernel is the canonical Gabor filter. It maps exactly onto
     skimage.filters.gabor_kernel with
         frequency = f0 / (2*pi),  theta = theta,
         sigma_x   = sig2lam * 2*pi / f0,  sigma_y = sigma_x / gamma,  offset = 0.
     Part A below builds both kernels and checks they are the same filter (L1-normalized,
     aligned) -- i.e. Nyxus uses the standard Gabor, verified against skimage.
  2. PIPELINE: given that canonical kernel, the feature is the documented WND-CHARM Gabor
     score -- L1-normalize, full-convolve, crop, and count response pixels above
     GRAYthr*baseline_max, over the baseline count. Part B reproduces it and matches every
     pinned value to machine precision.

This oracle is what justified FIXING the response truncation (gabor.cpp formerly stored the
filter-response magnitudes in a PixIntens=unsigned int image, flooring sub-integer responses
to 0). With responses kept real-valued, Nyxus == this oracle exactly.

NOTE on the parameterization: Nyxus reads f0 = pair.first (the {0, pi/4, pi/2, 3pi/4} values)
as the kernel FREQUENCY and theta = pair.second (the {4,16,32,64} values) as the ROTATION
angle -- i.e. the members of f0_theta_pairs are used opposite to the pair's name. This oracle
replicates the code's actual reading (verified: the "intended" swap does NOT reproduce Nyxus).

Provenance: tool=scikit-image 0.26 (skimage.filters.gabor_kernel); scipy.signal.convolve2d;
env=nyxus_mirp (conda); generator=tests/vetting/oracles/gen_gabor_skimage.py. Run offline.
"""
import os, re
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import gabor_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", ".."))

# Nyxus gabor parameters (gabor.cpp / test_gabor_truth.h)
N = 16; GAMMA = 0.1; SIG2LAM = 0.8; F0LP = 0.1; GRAYTHR = 0.025
PAIRS = [(0.0, 4.0), (np.pi/4, 16.0), (np.pi/2, 32.0), (np.pi*3/4, 64.0)]  # (f0=pair.first, theta=pair.second)


def nyxus_kernel(f0, theta):
    """Canonical Gabor on the exact n x n Nyxus grid tx,ty = -n/2 .. n/2-1, L1-normalized."""
    lam = 2*np.pi/f0 if f0 != 0 else np.inf
    sig = SIG2LAM*lam
    t = np.arange(N) - N//2
    X, Y = np.meshgrid(t, t)
    xte = X*np.cos(theta) + Y*np.sin(theta)
    yte = Y*np.cos(theta) - X*np.sin(theta)
    ge = np.exp(-(xte**2 + GAMMA**2*yte**2)/(2*sig*sig)) if np.isfinite(sig) else np.ones_like(xte)
    K = ge*(np.cos(xte*f0) + 1j*np.sin(xte*f0))
    return K/np.abs(K).sum()


def parse_images():
    txt = open(os.path.join(TESTS, "test_dsb2018_data.h")).read()
    txt = txt[txt.index("dsb_data"):]
    out = []
    for m in re.finditer(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*\{([\d,\s]*)\}\s*\}", txt):
        w, h = int(m.group(1)), int(m.group(2))
        px = [int(v) for v in m.group(3).split(",") if v.strip() != ""]
        if len(px) == w*h:
            out.append(np.array(px, float).reshape(h, w))
    return out


def parse_truth():
    txt = open(os.path.join(TESTS, "test_gabor_truth.h")).read()
    txt = txt[txt.index("gabor_truth"):]
    return [[float(v) for v in m.group(1).split(",") if v.strip() != ""]
            for m in re.finditer(r"\{([^{}]*)\}", txt)
            if len([v for v in m.group(1).split(",") if v.strip() != ""]) == 4]


def energy(img, f0, theta):
    K = nyxus_kernel(f0, theta)
    C = convolve2d(img, K, mode="full")
    off = int(np.ceil(N/2)); h, w = img.shape
    return np.abs(C[off:off+h, off:off+w])   # real magnitude (no unsigned-int truncation)


def feature(img):
    base = energy(img, F0LP, np.pi/2)
    mv, cv = base.max(), base.min()
    if mv == cv:
        return [0.0]*4
    bscore = int((base > cv).sum())
    return [int((energy(img, f0, th)/mv > GRAYTHR).sum())/bscore for f0, th in PAIRS]


def main():
    all_ok = True

    # ---- Part A: Nyxus kernel == skimage.filters.gabor_kernel (canonical Gabor) ----
    print("=== A. kernel vs skimage.filters.gabor_kernel ===")
    for f0, theta in PAIRS[1:]:   # skip f0=0 (degenerate uniform kernel)
        sigma_x = SIG2LAM*2*np.pi/f0
        sk = gabor_kernel(frequency=f0/(2*np.pi), theta=theta, sigma_x=sigma_x,
                          sigma_y=sigma_x/GAMMA, offset=0, n_stds=int(np.ceil((N/2)/sigma_x)) + 1)
        mine = nyxus_kernel(f0, theta)
        # crop/pad skimage kernel to N x N centred, L1-normalize both, compare
        sk = sk/np.abs(sk).sum()
        cy, cx = sk.shape[0]//2, sk.shape[1]//2
        skN = np.zeros((N, N), complex)
        for iy in range(N):
            for ix in range(N):
                sy, sx = cy + (iy - N//2), cx + (ix - N//2)
                if 0 <= sy < sk.shape[0] and 0 <= sx < sk.shape[1]:
                    skN[iy, ix] = sk[sy, sx]
        skN = skN/np.abs(skN).sum()
        d = np.abs(mine - skN).max()
        ok = d < 5e-3
        all_ok &= ok
        print(f"  {'OK ' if ok else 'FAIL'} f0={f0:.4f} theta={theta}: max|kernel diff|={d:.2e}")

    # ---- Part B: full-pipeline reproduction vs pinned goldens ----
    print("\n=== B. GABOR feature vs test_gabor_truth.h ===")
    images, truth = parse_images(), parse_truth()
    print(f"  ({len(images)} ROIs, {len(truth)} truth rows)")
    maxd = 0.0
    for i, img in enumerate(images):
        got = feature(img)
        for j in range(4):
            d = abs(got[j] - truth[i][j]); maxd = max(maxd, d)
            all_ok &= d <= 1e-9
    print(f"  max |diff| over all 16 values = {maxd:.3e}  ({'OK' if maxd <= 1e-9 else 'FAIL'})")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
