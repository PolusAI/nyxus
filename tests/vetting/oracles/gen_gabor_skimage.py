"""OFFLINE scikit-image oracle for the 2D GABOR feature (SPEC 4, oracle=skimage), on the
4 DSB2018 test ROIs (tests/test_dsb2018_data.h). Validates every golden pinned in
tests/test_2d_gabor_skimage.cc, at both config points the file pins.

What makes this an independent oracle (not a re-encoding of Nyxus):
  1. KERNEL: Nyxus' Gabor kernel is the canonical Gabor filter. It maps exactly onto
     skimage.filters.gabor_kernel with
         frequency = f0 / (2*pi),  theta = theta,
         sigma_x   = sig2lam * 2*pi / f0,  sigma_y = sigma_x / gamma,  offset = 0.
     skimage_kernel() below is the ONLY kernel used to produce the pinned values: it calls
     skimage.filters.gabor_kernel and crops it to Nyxus' n x n grid. Part A additionally
     prints the residual against a hand-derived closed form, purely as a cross-check that
     the parameter mapping is the one documented above.
  2. PIPELINE: given that canonical kernel, the feature is the documented WND-CHARM Gabor
     score -- L1-normalize, full-convolve, crop, and count response pixels above
     GRAYthr*baseline_max, over the baseline count. Part B reproduces it and matches every
     pinned value to machine precision.

  SCOPE of the vetting: the kernel is skimage's, the scoring pipeline is Nyxus' own definition
  reproduced here. skimage has no native equivalent of the WND-CHARM score to compare against,
  so what this oracle establishes is "Nyxus' filter is the canonical Gabor filter and its score
  is the documented count ratio over that filter", not agreement with a second implementation
  of the whole feature. Parts C and D bound how much of that claim the numbers can carry: the
  score is a ratio of pixel counts, so it is insensitive to kernel differences below the counting
  threshold (C), and the 16x16 crop is a defining part of the recipe rather than a detail (D) --
  because gamma=0.1 makes sigma_y ten times sigma_x, the analytic kernel is hundreds of pixels
  wide at the low frequencies and only 8-48% of its mass lies inside Nyxus' window. Running the
  same score off skimage's own filtering (skimage.filters.gabor, untruncated support, reflect
  border) moves values by up to ~1.0, i.e. the entire range of the feature.

TWO CONFIG POINTS (tests/vetting/config_recipes.md). Nyxus carries two different default
(frequency, angle) sets, and both are pinned:
  * gabor.cpp_static_defaults  -- GaborFeature::f0_theta_pairs as compiled in gabor.cpp:
        f0 = {0, pi/4, pi/2, 3pi/4}, theta = {4, 16, 32, 64} radians.
    Consumers read pair.first as the FREQUENCY and pair.second as the ANGLE, so this is what a
    run that sets no Gabor options computes.
  * gabor.documented_defaults  -- what GaborOptions::parse_input builds from the documented
    defaults gabor_freqs=[4,16,32,64], gabor_thetas=[0,45,90,135] deg, i.e.
        f0 = {4, 16, 32, 64}, theta = {0, pi/4, pi/2, 3pi/4} radians.
    This is the config the Python API always runs, and a CLI run that passes the flags.
The two produce entirely different values (up to 0.84 absolute apart); see
tests/vetting/audit/gabor_2d_skimage_vetting_report.md.

NOTE on the f0 = 0 filter: gabor_kernel cannot take frequency=0 (it divides by frequency to
size its own support). That pair is degenerate rather than approximated: at f0 = 0 the
wavelength is infinite, so the Gaussian envelope is identically 1 and the carrier
cos(0) + i*sin(0) is identically 1 -- the kernel is a flat n x n window of 1/n^2. It occurs
only in the gabor.cpp_static_defaults config, and it is derived here in closed form, not from
skimage.

Tolerance: the pinned goldens are ratios of pixel counts, so agreement is exact -- measured
max |diff| = 0.000e+00 over all 16 values at either config. The gtest asserts at the SPEC 7
same-definition tier (rel 1e-3); one miscounted pixel would move a value by 1/baseline_count,
which part C measures at 4.5e-3 .. 1.1e-2 on these ROIs, so that tolerance cannot mask a real
disagreement.

Provenance: tool=scikit-image 0.26.0 (skimage.filters.gabor_kernel); scipy 1.17.1
(scipy.signal.convolve2d); numpy 2.4.6; env=nyxus_mirp (conda, see TOOLS.md);
generator=tests/vetting/oracles/gen_gabor_skimage.py. Run offline.
"""
import os, re
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import gabor_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", ".."))

# Nyxus gabor parameters shared by both configs (gabor.cpp / test_2d_gabor_skimage.cc)
N = 16; GAMMA = 0.1; SIG2LAM = 0.8; F0LP = 0.1; GRAYTHR = 0.025

# (table name in test_2d_gabor_skimage.cc) -> (recipe id, [(f0, theta), ...])
CONFIGS = {
    "gabor_2d_skimage_cpp_defaults_ref_vals": (
        "gabor.cpp_static_defaults",
        [(0.0, 4.0), (np.pi/4, 16.0), (np.pi/2, 32.0), (np.pi*3/4, 64.0)]),
    "gabor_2d_skimage_documented_defaults_ref_vals": (
        "gabor.documented_defaults",
        [(4.0, 0.0), (16.0, np.pi/4), (32.0, np.pi/2), (64.0, np.pi*3/4)]),
}


def crop_to_nyxus_grid(K):
    """Centre-crop an arbitrary-sized kernel onto Nyxus' grid tx,ty = -N/2 .. N/2-1."""
    cy, cx = K.shape[0]//2, K.shape[1]//2
    out = np.zeros((N, N), complex)
    for iy in range(N):
        for ix in range(N):
            sy, sx = cy + (iy - N//2), cx + (ix - N//2)
            if 0 <= sy < K.shape[0] and 0 <= sx < K.shape[1]:
                out[iy, ix] = K[sy, sx]
    return out


def skimage_kernel(f0, theta):
    """The oracle kernel: skimage.filters.gabor_kernel, cropped to Nyxus' grid, L1-normalized."""
    if f0 == 0:
        # Degenerate baseline filter -- see the f0 = 0 note in the module docstring.
        # gabor_kernel(frequency=0) is undefined; the closed form is a flat window.
        K = np.ones((N, N), complex)
        return K/np.abs(K).sum()
    sigma_x = SIG2LAM*2*np.pi/f0
    sk = gabor_kernel(frequency=f0/(2*np.pi), theta=theta, sigma_x=sigma_x,
                      sigma_y=sigma_x/GAMMA, offset=0,
                      n_stds=int(np.ceil((N/2)/sigma_x)) + 1)
    sk = sk/np.abs(sk).sum()             # L1-normalize before cropping, as Nyxus normalizes its full kernel
    skN = crop_to_nyxus_grid(sk)
    return skN/np.abs(skN).sum()


def closed_form_kernel(f0, theta):
    """Hand-derived canonical Gabor on the Nyxus grid -- cross-check only, not the value path."""
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


def parse_table(name):
    """The rows of one ref_vals table in test_2d_gabor_skimage.cc, comments stripped.

    Every pinned value is read back and re-verified -- a hand-picked validation list is the
    failure mode this guards against, since it silently stops covering whatever is added later.
    """
    txt = open(os.path.join(TESTS, "test_2d_gabor_skimage.cc")).read()
    txt = re.sub(r"//[^\n]*", "", txt)                       # drop line comments (they hold digits too)
    start = txt.index(name)
    body = txt[txt.index("{", start): txt.index("};", start)]
    rows = [[float(v) for v in m.group(1).split(",") if v.strip() != ""]
            for m in re.finditer(r"\{([^{}]*)\}", body)]
    return [r for r in rows if r]


def energy(img, f0, theta):
    K = skimage_kernel(f0, theta)
    C = convolve2d(img, K, mode="full")
    off = int(np.ceil(N/2)); h, w = img.shape
    return np.abs(C[off:off+h, off:off+w])   # real magnitude (no unsigned-int truncation)


def feature(img, pairs):
    base = energy(img, F0LP, np.pi/2)
    mv, cv = base.max(), base.min()
    if mv == cv:
        return [0.0]*len(pairs)
    bscore = int((base > cv).sum())
    return [int((energy(img, f0, th)/mv > GRAYTHR).sum())/bscore for f0, th in pairs]


def main():
    all_ok = True
    images = parse_images()

    # ---- Part A: cross-check -- the skimage kernel used below == the documented closed form ----
    print("=== A. skimage.filters.gabor_kernel (value path) vs documented closed form ===")
    seen = set()
    for _, pairs in CONFIGS.values():
        for f0, theta in pairs:
            if f0 == 0 or (f0, theta) in seen:   # f0=0 is the degenerate uniform kernel, closed form by definition
                continue
            seen.add((f0, theta))
            d = np.abs(skimage_kernel(f0, theta) - closed_form_kernel(f0, theta)).max()
            ok = d < 5e-3
            all_ok &= ok
            print(f"  {'OK ' if ok else 'FAIL'} f0={f0:.4f} theta={theta:.4f}: max|kernel diff|={d:.2e}")

    # ---- Part B: full-pipeline reproduction vs the pinned goldens, table by table ----
    for table, (recipe, pairs) in CONFIGS.items():
        print(f"\n=== B. {table} ({recipe}) ===")
        truth = parse_table(table)
        print(f"  ({len(images)} ROIs, {len(truth)} pinned rows, {sum(len(r) for r in truth)} pinned values)")
        if len(truth) != len(images):
            print("  FAIL: pinned row count does not match the fixture ROI count")
            all_ok = False
            continue
        maxd = 0.0
        for i, img in enumerate(images):
            got = feature(img, pairs)
            if len(got) != len(truth[i]):
                print(f"  FAIL: ROI{i} pins {len(truth[i])} values, the config produces {len(got)}")
                all_ok = False
                continue
            for j in range(len(got)):
                maxd = max(maxd, abs(got[j] - truth[i][j]))
        ok = maxd <= 1e-9
        all_ok &= ok
        print(f"  max |diff| over all pinned values = {maxd:.3e}  ({'OK' if ok else 'FAIL'})")

    # ---- Part C: what the count-ratio scoring can resolve ----
    # The score counts pixels over a threshold, so it only moves in steps of 1/baseline_count.
    # Printing that step is what keeps the rel=1e-3 assertion tolerance honest: it is two orders
    # of magnitude below the smallest difference a real disagreement could produce.
    print("\n=== C. resolution of the score (smallest difference one miscounted pixel makes) ===")
    for i, img in enumerate(images):
        base = energy(img, F0LP, np.pi/2)
        bscore = int((base > base.min()).sum())
        print(f"  ROI{i}: baseline count = {bscore}, one pixel = {1.0/bscore:.3e}")

    # ---- Part D: how much of the recipe is the 16x16 crop? (informational, never fails) ----
    # gamma=0.1 makes sigma_y ten times sigma_x, so the analytic Gabor is far wider than Nyxus'
    # window at the low frequencies. This measures what the window keeps, and what the score
    # becomes if the oracle uses skimage's own filtering instead of Nyxus' truncate-and-zero-pad
    # convention. Printed, not asserted: it states the scope of the vetting claim, and re-measures
    # it whenever skimage changes.
    print("\n=== D. the 16x16 crop is part of the recipe, not a detail ===")
    print("  f0       sigma_x  sigma_y  skimage support  L1 mass inside 16x16")
    seen_f0 = set()
    for _, pairs in CONFIGS.values():
        for f0, theta in pairs:
            if f0 == 0 or f0 in seen_f0:
                continue
            seen_f0.add(f0)
            sigma_x = SIG2LAM*2*np.pi/f0
            K = gabor_kernel(frequency=f0/(2*np.pi), theta=theta, sigma_x=sigma_x,
                             sigma_y=sigma_x/GAMMA, offset=0)          # skimage's own n_stds=3
            cy, cx = K.shape[0]//2, K.shape[1]//2
            sub = K[max(0, cy - N//2):cy + N//2, max(0, cx - N//2):cx + N//2]
            print(f"  {f0:7.4f}  {sigma_x:7.2f}  {sigma_x/GAMMA:7.2f}  {K.shape[0]:5d}x{K.shape[1]:<5d}"
                  f"      {100*np.abs(sub).sum()/np.abs(K).sum():6.2f}%")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
