"""OFFLINE analytic oracle for the image-quality (imq) features on the im_quality fixture
(test_data.h). Independently reimplements the documented algorithms in numpy/scipy and
validates the goldens pinned in test_imq_regression.h.

Vets (4):
  FOCUS_SCORE        -- variance of the Laplacian magnitude (scipy convolution with the
                        [[0,1,0],[1,-4,1],[0,1,0]] kernel). REQUIRED a fix first: focus_score.cpp
                        multiplied PixIntens (unsigned) by the negative kernel weights, wrapping
                        the Laplacian (FOCUS_SCORE reached ~2.81e18); now cast to double -> 12.109.
  LOCAL_FOCUS_SCORE  -- same, on the top-left (h/2 x w/2) tile (scale=2), /4.
  MIN_SATURATION     -- fraction of pixels at the image minimum.
  MAX_SATURATION     -- fraction of pixels at the image maximum.

NOT vetted here (documented in oracle_coverage.csv):
  SHARPNESS          -- correct (all-double DOM sharpness, Kumar et al. 2012), no bug; needs a
                        reference-DOM oracle.
  POWER_SPECTRUM_SLOPE -- 0.0 on this fixture (rps() returns 0 when min(h,w)/8 < 3), and its radial
                        binning uses the FFT value as the radius index (power_spectrum.cpp:169) --
                        a real bug; needs a rewrite + a >=24 px fixture.

The im_quality fixture is 8x12; rows y=7..9 leave x=3..8 unset (background 0), so min=0. Verified
by reconstructing the ImageMatrix from test_data.h's im_quality_intensity.

Provenance: tool=analytic (numpy + scipy.ndimage.convolve); env=nyxus_mirp (conda);
generator=tests/vetting/oracles/gen_imq_analytic.py. Run offline; CI never invokes it.
"""
import numpy as np
from scipy.ndimage import convolve

# reconstructed im_quality_intensity image (8 wide x 12 tall); background (unset) = 0
IMG = np.array([
    [1,4,4,1,1,4,1,1], [1,4,6,1,1,6,1,1], [4,1,6,4,1,6,4,1], [4,4,6,4,1,6,4,1],
    [4,4,6,4,1,6,4,1], [4,4,6,4,1,6,4,1], [1,4,0,0,0,0,0,0], [1,4,0,0,0,0,0,0],
    [4,1,0,0,0,0,0,0], [4,4,6,4,1,6,4,1], [4,4,6,4,1,6,4,1], [4,4,6,4,1,6,4,1],
], float)

# goldens from test_imq_regression.h (FOCUS/LOCAL are the post-fix values)
NYX = {"FOCUS_SCORE": 12.1094, "LOCAL_FOCUS_SCORE": 2.68056,
       "MIN_SATURATION": 0.1875, "MAX_SATURATION": 0.166667}
TOL = 1e-3
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)


def nyx_variance(v):        # Nyxus variance(): mean((|x| - mean|x|)^2)
    a = np.abs(v.ravel())
    return float(np.mean((a - a.mean())**2))


def focus_score(img):
    return nyx_variance(convolve(img, LAP, mode="constant", cval=0.0))


def local_focus_score(img, scale=2):
    h, w = img.shape
    M, N = h // scale, w // scale
    total, y = 0.0, 0
    while y < h - M:
        x = 0
        while x < w - N:
            total += nyx_variance(convolve(img[y:y+M, x:x+N], LAP, mode="constant", cval=0.0))
            x += N
        y += M
    return total / (scale * scale)


def main():
    got = {"FOCUS_SCORE": focus_score(IMG),
           "LOCAL_FOCUS_SCORE": local_focus_score(IMG),
           "MIN_SATURATION": float(np.mean(IMG == IMG.min())),
           "MAX_SATURATION": float(np.mean(IMG == IMG.max()))}
    all_ok = True
    print("=== imq analytic oracle vs test_imq_regression.h ===")
    for k in NYX:
        ok = abs(got[k] - NYX[k]) <= TOL * max(1.0, abs(NYX[k]))
        all_ok &= ok
        print(f"  {'OK ' if ok else 'FAIL'} {k}: oracle={got[k]:.6f} nyxus={NYX[k]}")
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
