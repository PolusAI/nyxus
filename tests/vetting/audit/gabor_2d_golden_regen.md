# Regenerating the 2D GABOR goldens

Both golden tables in `tests/test_2d_gabor_skimage.cc` come from one generator,
`tests/vetting/oracles/gen_gabor_skimage.py`. It runs offline; scikit-image is never a CI
dependency.

## 1. Environment

Any env with scikit-image, scipy and numpy works. The pinned values were produced in the conda env
`nyxus_mirp` (TOOLS.md), which already carries them:

```
conda create -n nyxus_skimage -c conda-forge python=3.11 scikit-image scipy numpy   # if starting fresh
python -c "import skimage, scipy, numpy; print(skimage.__version__, scipy.__version__, numpy.__version__)"
# pinned with: 0.26.0 1.17.1 2.4.6
```

## 2. Fixture

None to prepare — the generator reads the four DSB2018 ROIs straight out of
`tests/test_dsb2018_data.h` (benchmark `bench_dsb2018_2d`), so the C++ tests and the oracle share one
copy of the pixels. Each ROI is the whole image (the gtest feeds every pixel under one label), so no
mask is applied.

## 3. Run

```
python tests/vetting/oracles/gen_gabor_skimage.py
```

Exit 0 means every value pinned in `test_2d_gabor_skimage.cc` was re-derived from a fresh skimage
run. The script parses **both** tables out of the `.cc` — it holds no copy of the numbers, so a pin
added later is checked automatically. Output sections:

- **A** — the skimage kernel used for the values vs the hand-derived canonical Gabor closed form, at
  every (f0, theta) point in both recipes. Expect `<= 1e-16`.
- **B** — one block per table: `max |diff|` over its pins. Expect `0.000e+00`.
- **C** — `1/baseline_count` per ROI, i.e. the smallest change one miscounted pixel makes. This is
  what justifies the `rel=1e-3` assertion tolerance.
- **D1** — kernel widths and the share of L1 mass that survives Nyxus' 16×16 crop, per frequency.
- **D2** — the negative control: the same score computed off `skimage.filters.gabor`, i.e. skimage's
  own untruncated kernel support and border handling, against the pinned values. Informational, never
  fails; it is what makes the "Nyxus' crop convention is part of the recipe" claim reproducible
  instead of remembered. Expect ~1.005 and ~0.417 at `constant`; the `reflect` rows move between runs
  and are labelled as such (report §4.1).

Parts A–C are instant. D2 convolves against kernels ~3000 px a side, so the whole script takes ~40 s.

## 4. Re-pinning after an intentional change

The generator verifies; it does not print a paste-ready table. To emit one (e.g. after the
`gabor.cpp` default is corrected — see `gabor_2d_skimage_vetting_report.md` §3.1):

```python
import importlib.util
spec = importlib.util.spec_from_file_location("gen", "tests/vetting/oracles/gen_gabor_skimage.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
recipe, pairs = g.CONFIGS["gabor_2d_skimage_python_raw_defaults_ref_vals"]
for img in g.parse_images():
    print("    {   " + ",\n        ".join(repr(v) for v in g.feature(img, pairs)) + " },\n")
```

Paste the rows into the matching table, keep `repr()` precision (17 significant digits), then re-run
the generator and `runAllTests --gtest_filter=*GABOR*`.

## 5. Mapping the oracle onto Nyxus

| Nyxus setting | oracle equivalent |
|---|---|
| `gabor_kersize` n = 16 | kernel centre-cropped to 16×16 (`crop_to_nyxus_grid`) |
| `gabor_gamma` = 0.1 | `sigma_y = sigma_x / gamma` |
| `gabor_sig2lam` = 0.8 | `sigma_x = sig2lam * 2*pi / f0` |
| frequency f0 | `gabor_kernel(frequency = f0 / (2*pi))` |
| angle theta (radians) | `gabor_kernel(theta = theta)` |
| `gabor_f0` = 0.1 (baseline) | baseline response, computed at `theta = pi/2` |
| `gabor_thold` = 0.025 | `GRAYthr`, the response/baseline-max cut |
| feature value | `count(response/baseline_max > GRAYthr) / count(baseline > baseline_min)` |

Convention differences to keep in mind:

- **Kernel normalization order.** Nyxus L1-normalizes its full kernel; the oracle L1-normalizes the
  skimage kernel, crops it to 16×16, then renormalizes. Anything outside the 16×16 grid is dropped
  by both — and at `gamma=0.1` that is most of the kernel: 7.9% of the L1 mass survives the crop at
  f0 = pi/4, 21% at pi/2, 31% at 3pi/4, 48% at f0 = 4, and 100% from f0 >= 16 (generator part D1).
- **Convolution and border.** `mode="full"` then cropped at `offset = ceil(n/2)`, which is what
  `GaborEnergy` does. skimage's own `skimage.filters.gabor()` convolves with `mode="reflect"` over
  an untruncated kernel and is *not* interchangeable here: scoring off it moves values by up to 1.005
  (report §4.1, generator part D2). If you regenerate through anything other than `gabor_kernel` +
  crop + zero-padded `full` convolution, you are pinning a different feature. `mode='reflect'` is
  additionally not reproducible at these ROI sizes — see §4.1.
- **Real-valued response.** The magnitude is kept in `double`. Nyxus stored it in a `PixIntens`
  (`unsigned int`) image until the truncation fix, which floored sub-integer responses to 0.
- **Which pair element is which.** Nyxus' `f0_theta_pairs` is consumed as
  `(frequency, angle-in-radians)`, but the compiled-in default in `gabor.cpp` is written in the
  opposite order. The two recipes in `config_recipes.md` are exactly that difference; regenerate
  against the recipe the table names, not against the documented parameter list.
- **Frequency units.** `gabor_freqs` is documented as denominators of pi and implemented as raw
  angular frequency — `4` reaches the filter as f0 = 4. The generator uses the implemented reading,
  which is why the recipe is `gabor.python_raw_defaults`; report §8 item 3 carries the contract.
