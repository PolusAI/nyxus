# 2D GABOR config matrix

Axes are the settings `GaborFeature` actually reads, per SPEC §5.2 step 1 — extracted from
`calculate()` and `GaborEnergy()`/`Gabor()` in `src/nyx/features/gabor.cpp`, not from the settings
struct at large. `calculate (LR&, const Fsettings& s)` touches `s` only to fetch the NaN placeholder
for the degenerate branch; every parameter that changes a value is a static on the feature class:

| static | CLI / Python knob | what it does |
|---|---|---|
| `f0_theta_pairs` | `--gaborfreqs` + `--gabortheta` / `gabor_freqs` + `gabor_thetas` | the (frequency, angle) set — one feature value per pair |
| `n` | `--gaborkersize` / `gabor_kersize` | kernel side, 16 |
| `gamma` | `--gaborgamma` / `gabor_gamma` | envelope aspect ratio, 0.1 |
| `sig2lam` | `--gaborsig2lam` / `gabor_sig2lam` | sigma over lambda, 0.8 |
| `f0LP` | `--gaborf0` / `gabor_f0` | baseline lowpass frequency, 0.1, always at theta = pi/2 |
| `GRAYthr` | `--gaborthold` / `gabor_thold` | response/baseline-max cut, 0.025 |

`f0_theta_pairs` is the only axis the two shipped configurations differ on; the other five keep their
defaults everywhere in the tree, so the matrix stays one-dimensional.

## Config points

| `f0_theta_pairs` | reached by | verdict | recipe / oracle |
|---|---|---|---|
| f0 = {0, pi/4, pi/2, 3pi/4}, theta = {4, 16, 32, 64} rad | a CLI run with neither `--gaborfreqs` nor `--gabortheta`, and any direct `GaborFeature::calculate` caller including the gtest fixture | **VALID** → oracle | `gabor.cpp_static_defaults` — skimage at kernel level, `rel=1e-3`, `TEST_NYXUS.TEST_2D_GABOR_CPP_STATIC_DEFAULTS_SKIMAGE` on `bench_dsb2018_2d` |
| f0 = {4, 16, 32, 64}, theta = {0, pi/4, pi/2, 3pi/4} rad | every Python API call (the wrapper always passes its defaults through the parser), and a CLI run that passes both flags | **VALID** → oracle | `gabor.python_raw_defaults` — same oracle and band, `TEST_NYXUS.TEST_2D_GABOR_PYTHON_RAW_DEFAULTS_SKIMAGE` |
| f0 = {pi/4, pi/16, pi/32, pi/64}, i.e. the frequency list read as denominators of pi | **nothing** — no code path divides by pi | **not reachable** | the units the Python docstring and the CLI help both describe; see below |
| CLI help's `1,2,4,8,16,32,64` | nothing — `parse_input` rebuilds the pairs only when *both* flags are non-empty, so this list is never a default | **not reachable** | a third documented default, `src/nyx/environment.cpp` |
| any pair set, on the GPU | `--useGpu=true` | **INVALID for an oracle** → `_mechanics` | `calculate_gpu` convolves through an FFT and diverges from the CPU path on this benchmark; guarded size-only by `TEST_NYXUS.TEST_2D_GABOR_GPU_RUNS_MECHANICS` |
| `f0 = 0` as a member of a pair set | only inside `gabor.cpp_static_defaults` | **degenerate, kept** | the envelope and carrier are both identically 1, so the filter is a flat window; `gabor_kernel` cannot express it, so that one kernel is `analytic` |

## Why two default cells exist at all

They are the same defect seen from two entry points. `gabor.cpp` writes its static pairs as
`{angle, frequency}` while `cli_gabor_options.cpp:37` writes `{frequency, angle}`, and every
consumer — `calculate`, `calculate_gpu`, `calculate_gpu_multi_filter` — reads `.first` as the
frequency. So the compiled-in default runs the angles as frequencies and vice versa, and its first
filter has f0 = 0. The two cells produce values up to **0.84 absolute** apart on the same ROI.

Both are asserted rather than one, because both are reachable: dropping either would leave a real
entry point uncovered. Correcting `gabor.cpp` changes public feature values, so it is measured in
`audit/gabor_2d_skimage_vetting_report.md` §3.1 and deferred to its own branch; when it lands, the
`gabor.cpp_static_defaults` table is the one to regenerate and the two tables converge.

## The two rows that cannot be reached

The third and fourth cells are documentation, not configurations. `gabor_freqs` is described as
"comma-separated denominators of `\pi`" in `src/nyx/python/nyxus/nyxus.py` and as the same thing in
the CLI help, but `GaborOptions::parse_input` stores the list verbatim and `Gabor()` consumes it as
`lambda = 2*pi/f0`. Nothing divides by pi anywhere in the tree, and the CLI help additionally states
a default list no code path can produce.

That is why the second recipe is named `gabor.python_raw_defaults` and not for the documented
contract: the assertion covers the numbers that reach the filter. Deciding which units are intended
is a source change with public value impact, tracked in the vetting report §8.

## What would move a cell

- **Resolving the units contract** turns the third cell into either a real config point (if the
  documentation is right and the parser should divide) or a documentation fix (if the parser is
  right). Either way the `gabor.python_raw_defaults` goldens are re-measured, not re-labelled.
- **A full-feature oracle.** Both VALID cells are vetted at kernel level only, because scikit-image
  has no equivalent of the WND-CHARM count-ratio score. SPEC §4 names `wndcharm` and
  `feature2djava` as the highest-value oracles for exactly this class of Nyxus-original feature;
  building either would upgrade both cells from "the filter is canonical and the score is the
  documented ratio over it" to a second implementation of the whole feature.
