"""OFFLINE OpenCV oracle for the image-quality focus-score features
(SPEC 4, oracle=opencv). Runs the real cv2.Laplacian on the gtest fixture
`im_quality_intensity` / `im_quality_mask` (test_data.h) and validates the goldens
pinned in test_imq_opencv.h.

Vets (2):
  FOCUS_SCORE        -- cv2.Laplacian(roi, CV_64F, ksize=1, BORDER_CONSTANT).var()
  LOCAL_FOCUS_SCORE  -- the same call on the extracted tile, / scale^2

FOCUS_SCORE is the Pech-Pacheco et al. (2000) "variance of the Laplacian" focus measure,
which is what `cv2.Laplacian(img, cv2.CV_64F).var()` computes. Two independent parts:

  1. The convolution. Nyxus' hand-rolled laplacian() uses the ksize=1 kernel
     [[0,1,0],[1,-4,1],[0,1,0]] and drops out-of-range taps, i.e. zero padding. That is
     exactly cv2.Laplacian(..., ksize=1, borderType=cv2.BORDER_CONSTANT); this generator
     asserts the two filtered images are bit-identical (max abs diff 0.0), so the
     convolution is proved, not merely close.
  2. The variance. Plain population variance of the signed filtered image,
     mean((x - mean(x))^2) == numpy/cv2 .var() (ddof=0).

WHY THE GOLDENS MOVED. focus_score.cpp' variance() used to take |x| *before* computing
the variance, i.e. Var(|X|) = E[X^2] - E[|X|]^2, which understates the true
Var(X) = E[X^2] - E[X]^2 whenever the raw Laplacian's mean is not 0 -- and it is not,
because zero padding pulls the mean away from 0 (mean = -0.958 on this fixture). That
made FOCUS_SCORE 12.109375 against OpenCV's 34.95659722, a 65% relative error. The abs()
is now gone; it was never in the documented definition (docs/source/Math/
f_image_quality.rst) either.

SCOPE OF THE CLAIM -- what these two assertions do and do not cover:
  * ksize=1 only. The ksize>1 kernel [[2,0,2],[0,-8,0],[2,0,2]] is a Nyxus/CellProfiler
    convention with no cv2.Laplacian counterpart (cv2's ksize=3 Sobel-derived Laplacian
    is [[0,1,0],[1,-4,1],[0,1,0]] scaled differently), and Nyxus never calls it from
    calculate() -- not covered here.
  * LOCAL_FOCUS_SCORE covers only the top-left tile. get_local_focus_score() loops
    `for (y = 0; y < height - M; y += M)` with M = height/scale, so for scale=2 the
    condition is 0 < h/2 -> true, 6 < 6 -> false: exactly ONE tile is visited, not
    scale^2 = 4. The pinned value is therefore var(Laplacian(top-left h/2 x w/2 tile))
    / scale^2, and the oracle reproduces that. The loop bound (`<` where `<=` is
    presumably meant) and the docs' claim that "the mean and median values of the tiles
    are returned" are a separate, untouched discrepancy -- this assertion pins current
    tiling, it does not endorse it.
  * The out-of-core path (get_focus_score_NT) is not covered: it computes the variance by
    Welford over a padded window buffer and passes (width, height) to laplacian() in the
    (m_image, n_image) order, i.e. transposed. Out of scope here.

NOT vetted here: MIN_SATURATION / MAX_SATURATION (-> gen_imq_cellprofiler.py),
POWER_SPECTRUM_SLOPE (radial-binning bug, needs a rewrite and a >=24 px fixture),
SHARPNESS (needs a reference-DOM oracle).

Provenance: tool=opencv, version=cv2 4.13.0 (opencv-python), numpy 2.4.6, python 3.11.15;
cv2.Laplacian(src=float64 ROI, ddepth=cv2.CV_64F, ksize=1,
borderType=cv2.BORDER_CONSTANT) then ndarray.var() (ddof=0).
generator=tests/vetting/oracles/gen_imq_opencv.py. Run offline; CI never invokes it --
OpenCV is not a Nyxus runtime dependency.
"""
import numpy as np
import cv2

# The ROI image matrix Nyxus builds from im_quality_intensity / im_quality_mask
# (tests/test_data.h): 8 wide x 12 tall. Rows y=7..9 of the fixture literal repeat the
# coordinates of rows 1..3, so x=3..8 there is never assigned and stays background 0.
IMG = np.array([
    [1, 4, 4, 1, 1, 4, 1, 1], [1, 4, 6, 1, 1, 6, 1, 1], [4, 1, 6, 4, 1, 6, 4, 1],
    [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1],
    [1, 4, 0, 0, 0, 0, 0, 0], [1, 4, 0, 0, 0, 0, 0, 0], [4, 1, 0, 0, 0, 0, 0, 0],
    [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1],
], float)

KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)  # focus_score.cpp ksize=1
SCALE = 2  # FocusScoreFeature::get_local_focus_score default

# goldens pinned in tests/test_imq_opencv.h (== Nyxus output after the variance fix)
NYXUS = {"FOCUS_SCORE": 34.95659722222222, "LOCAL_FOCUS_SCORE": 7.57638888888889}
TOL = 1e-9


def cv_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT)


def nyxus_laplacian(img):
    """Reimplementation of FocusScoreFeature::laplacian() (zero padding, ksize=1)."""
    h, w = img.shape
    out = np.zeros((h, w), float)
    for i in range(h):
        for j in range(w):
            for ik in range(3):
                for jk in range(3):
                    ii, jj = i + 1 - ik, j + 1 - jk
                    if 0 <= ii < h and 0 <= jj < w:
                        out[i, j] += img[ii, jj] * KERNEL[ik, jk]
    return out


def nyxus_tiles(img, scale):
    """The tiles get_local_focus_score() actually visits (see SCOPE above)."""
    h, w = img.shape
    m, n = h // scale, w // scale
    return [img[y:y + m, x:x + n]
            for y in range(0, h - m, m) for x in range(0, w - n, n)]


def main():
    all_ok = True
    print("=== OpenCV cv2.Laplacian focus scores vs Nyxus goldens ===")
    print(f"    cv2 {cv2.__version__}, fixture {IMG.shape[1]}x{IMG.shape[0]}")

    # (1) prove the convolution, not just the scalar
    conv_diff = np.abs(cv_laplacian(IMG) - nyxus_laplacian(IMG)).max()
    ok = conv_diff == 0.0
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'} convolution: max|cv2 - nyxus laplacian| = {conv_diff!r}")
    print(f"       raw Laplacian mean = {cv_laplacian(IMG).mean()!r} "
          f"(non-zero -> abs()-before-variance would understate the variance)")

    # (2) FOCUS_SCORE
    focus = float(cv_laplacian(IMG).var())
    ok = abs(focus - NYXUS["FOCUS_SCORE"]) <= TOL * max(1.0, abs(focus))
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'} FOCUS_SCORE: opencv={focus!r} "
          f"nyxus={NYXUS['FOCUS_SCORE']!r}")

    # (3) LOCAL_FOCUS_SCORE
    tiles = nyxus_tiles(IMG, SCALE)
    local = sum(float(cv_laplacian(t).var()) for t in tiles) / (SCALE * SCALE)
    ok = abs(local - NYXUS["LOCAL_FOCUS_SCORE"]) <= TOL * max(1.0, abs(local))
    all_ok &= ok
    print(f"  {'OK ' if ok else 'FAIL'} LOCAL_FOCUS_SCORE: opencv={local!r} "
          f"nyxus={NYXUS['LOCAL_FOCUS_SCORE']!r}  "
          f"({len(tiles)} tile(s) of {tiles[0].shape[1]}x{tiles[0].shape[0]}, /scale^2)")

    print(f"\n{'ALL OPENCV-VET CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
