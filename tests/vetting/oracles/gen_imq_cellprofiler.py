"""OFFLINE CellProfiler oracle for the image-quality saturation features
(SPEC 4, oracle=cellprofiler). Runs the real cellprofiler.modules.MeasureImageQuality
module on the gtest fixture `im_quality_intensity` / `im_quality_mask` (test_data.h) and
validates the goldens pinned in test_imq_cellprofiler.h.

Vets (2):
  MIN_SATURATION -- CP Image_ImageQuality_PercentMinimal / 100
  MAX_SATURATION -- CP Image_ImageQuality_PercentMaximal / 100

Both tools use the same convention: the fraction of pixels equal to the image's OWN
observed extremum (CP: numpy.min/numpy.max of the pixel data, masked when a mask is
present), not a fixed bit-depth threshold such as 255. CP's calculate_saturation():

    number_pixels_maximal = numpy.sum(pixel_data == numpy.max(pixel_data))
    number_pixels_minimal = numpy.sum(pixel_data == numpy.min(pixel_data))
    percent_maximal = 100.0 * number_pixels_maximal / pixel_count
    percent_minimal = 100.0 * number_pixels_minimal / pixel_count

The metric is scale-invariant (it only compares pixels to the extremum), so CP's usual
[0,1] float images and Nyxus' integer PixIntens give the same fraction; this generator
checks that explicitly by running raw and max-normalized inputs.

SCOPE OF THE CLAIM -- two places where the two implementations are NOT interchangeable,
neither of them exercised by this fixture:
  * Constant ROI (min == max): CP counts minimal and maximal independently and reports
    100% for both; Nyxus' get_percent_max_pixels() uses `else if`, so a pixel equal to
    both extrema is counted only as maximal and MIN_SATURATION comes out 0. This
    assertion does not cover that case.
  * Nyxus computes over the ROI's bounding-box image matrix, in which pixels inside the
    box but outside the mask are 0 and DO take part in the min/extremum count; CP
    restricts to image.mask when a mask is present. They coincide here only because
    im_quality_mask covers the whole 8x12 box.

NOT vetted here: FOCUS_SCORE / LOCAL_FOCUS_SCORE. CellProfiler has features by those
names, but they are a different statistic -- CP's FocusScore is the *normalized variance
of the raw image*, sum((x-mean)^2)/(N*mean), and CP's LocalFocusScore is
var(local_norm_var)/median(local_norm_var) over a grid. Nyxus implements the
Pech-Pacheco variance-of-Laplacian instead, which is vetted against OpenCV in
gen_imq_opencv.py. Do not point a CP oracle at those two names.

Environment: a dedicated CellProfiler env is required to RUN this generator
(cellprofiler-core + centrosome + the cellprofiler module package, headless).
CI never invokes it -- CellProfiler is not a runtime dependency.

Provenance: tool=cellprofiler, version=4.2.8 (module package) / cellprofiler-core
4.2.8.1, centrosome 1.2.3, numpy 1.26.4, scipy 1.10.1; python 3.9;
MeasureImageQuality, calculate_saturation() on a single grayscale image, no mask.
generator=tests/vetting/oracles/gen_imq_cellprofiler.py. Run offline.

Note: importing cellprofiler.modules.measureimagequality pulls in
cellprofiler.gui.help.content, which calls os.path.relpath() against the current working
directory. Run this generator from a directory on the same drive as the CellProfiler
install, or the import dies with "path is on mount 'C:', start on mount 'D:'".
"""
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
from cellprofiler.modules import measureimagequality as miq

# The ROI image matrix Nyxus builds from im_quality_intensity / im_quality_mask
# (tests/test_data.h): 8 wide x 12 tall. Rows y=7..9 of the fixture literal repeat the
# coordinates of rows 1..3, so x=3..8 there is never assigned and stays background 0 --
# which is why the observed minimum of this ROI is 0, not 1.
IMG = np.array([
    [1, 4, 4, 1, 1, 4, 1, 1], [1, 4, 6, 1, 1, 6, 1, 1], [4, 1, 6, 4, 1, 6, 4, 1],
    [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1],
    [1, 4, 0, 0, 0, 0, 0, 0], [1, 4, 0, 0, 0, 0, 0, 0], [4, 1, 0, 0, 0, 0, 0, 0],
    [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1], [4, 4, 6, 4, 1, 6, 4, 1],
], float)

# goldens pinned in tests/test_imq_cellprofiler.h (== Nyxus output)
NYXUS = {"MIN_SATURATION": 0.1875, "MAX_SATURATION": 0.16666666666666666}
TOL = 1e-6

IMAGE_NAME = "imq"


def run_cp(pixel_data):
    """Run MeasureImageQuality's saturation metrics on one grayscale image."""
    module = miq.MeasureImageQuality()
    group = module.image_groups[0]
    module.images_choice.value = miq.O_SELECT
    group.image_names.value = IMAGE_NAME  # ImageListSubscriber parses a ", "-joined string
    group.check_blur.value = False
    group.check_saturation.value = True
    group.check_intensity.value = False
    group.calculate_threshold.value = False

    isl = cpi.ImageSetList()
    image_set = isl.get_image_set(0)
    image_set.add(IMAGE_NAME, cpi.Image(pixel_data))

    m = cpmeas.Measurements()
    ws = cpw.Workspace(cpp.Pipeline(), module, image_set, cpo.ObjectSet(), m, isl)
    module.calculate_saturation(group, ws)

    get = lambda f: float(m.get_current_measurement(
        "Image", "{}_{}_{}".format(miq.C_IMAGE_QUALITY, f, IMAGE_NAME)))
    return {"MIN_SATURATION": get(miq.F_PERCENT_MINIMAL) / 100.0,
            "MAX_SATURATION": get(miq.F_PERCENT_MAXIMAL) / 100.0}


def main():
    raw = run_cp(IMG)
    normalized = run_cp(IMG / IMG.max())  # CP's usual [0,1] float convention

    all_ok = True
    print("=== CellProfiler MeasureImageQuality (saturation) vs Nyxus goldens ===")
    print(f"    fixture: {IMG.shape[1]}x{IMG.shape[0]} ROI, min={IMG.min():.0f} "
          f"max={IMG.max():.0f}, {int((IMG == IMG.min()).sum())} at min, "
          f"{int((IMG == IMG.max()).sum())} at max, {IMG.size} pixels")
    for feat, nyx in NYXUS.items():
        cp = raw[feat]
        ok = abs(cp - nyx) <= TOL * max(1.0, abs(nyx))
        scale_ok = abs(normalized[feat] - cp) <= TOL
        all_ok &= ok and scale_ok
        print(f"  {'OK ' if ok else 'FAIL'} {feat}: cellprofiler={cp!r} nyxus={nyx!r}"
              f"  (scale-invariant: {'yes' if scale_ok else 'NO'})")

    print(f"\n{'ALL CP-VET CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
