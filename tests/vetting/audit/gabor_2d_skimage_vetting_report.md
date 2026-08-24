# 2D GABOR — scikit-image vetting report

Re-vetting of the 2D `gabor` family (one feature, `GABOR`, four filter values per ROI) against a
fresh scikit-image run. Every pinned golden was compared against the oracle re-run from scratch,
not merely against "the C++ test passes".

**Result:** every pinned value reproduces exactly (`max |diff| = 0.000e+00`, 32 of 32 values), and
**the family's only vetted claim turned out to describe a configuration Nyxus does not run through
its Python API** — the pinned goldens come from the compiled-in `f0_theta_pairs` default, while the
test header's provenance comment named the documented defaults, whose values differ from the pinned
ones by up to **0.84 absolute**. Both config points are now pinned and asserted separately.

## 1. Reproduction

| item | value |
|---|---|
| oracle | scikit-image 0.26.0 (`skimage.filters.gabor_kernel`) |
| support | scipy 1.17.1 (`scipy.signal.convolve2d`), numpy 2.4.6, Python 3.11.15 |
| env | conda `nyxus_mirp` (see TOOLS.md) |
| generator | `tests/vetting/oracles/gen_gabor_skimage.py` |
| fixture | the 4 DSB2018 ROIs in `tests/test_dsb2018_data.h`, whole image as one ROI |
| Nyxus build | `runAllTests`, MSVC 19.44, Release, `USEGPU=OFF` |

```
python tests/vetting/oracles/gen_gabor_skimage.py      # exit 0 = every pin re-derived
```

The generator parses **both** golden tables out of `tests/test_2d_gabor_skimage.cc` and re-derives
every value in them; it does not carry its own copy of the numbers and has no hand-picked
validation list.

## 2. Result table — pinned vs fresh run

Shared settings for both recipes: `kersize n=16`, `gamma=0.1`, `sig2lam=0.8`, baseline `f0LP=0.1`
at `theta=pi/2`, `GRAYthr=0.025`.

#### gabor.cpp_static_defaults

| ROI | filter (f0, theta rad) | pinned | fresh skimage run | rel diff |
|---|---|---|---|---|
| 0 | (0.0000, 4.0000) | 1.0112359550561798 | 1.0112359550561798 | 0.0e+00 |
| 0 | (0.7854, 16.0000) | 0.9213483146067416 | 0.9213483146067416 | 0.0e+00 |
| 0 | (1.5708, 32.0000) | 0.9662921348314607 | 0.9662921348314607 | 0.0e+00 |
| 0 | (2.3562, 64.0000) | 0.6179775280898876 | 0.6179775280898876 | 0.0e+00 |
| 1 | (0.0000, 4.0000) | 1.0044843049327354 | 1.0044843049327354 | 0.0e+00 |
| 1 | (0.7854, 16.0000) | 0.9327354260089686 | 0.9327354260089686 | 0.0e+00 |
| 1 | (1.5708, 32.0000) | 0.11210762331838565 | 0.11210762331838565 | 0.0e+00 |
| 1 | (2.3562, 64.0000) | 0.17488789237668162 | 0.17488789237668162 | 0.0e+00 |
| 2 | (0.0000, 4.0000) | 1.0053763440860215 | 1.0053763440860215 | 0.0e+00 |
| 2 | (0.7854, 16.0000) | 0.978494623655914 | 0.978494623655914 | 0.0e+00 |
| 2 | (1.5708, 32.0000) | 0.3709677419354839 | 0.3709677419354839 | 0.0e+00 |
| 2 | (2.3562, 64.0000) | 0.0 | 0.0 | 0.0e+00 |
| 3 | (0.0000, 4.0000) | 1.0051546391752577 | 1.0051546391752577 | 0.0e+00 |
| 3 | (0.7854, 16.0000) | 0.9587628865979382 | 0.9587628865979382 | 0.0e+00 |
| 3 | (1.5708, 32.0000) | 0.4845360824742268 | 0.4845360824742268 | 0.0e+00 |
| 3 | (2.3562, 64.0000) | 0.04639175257731959 | 0.04639175257731959 | 0.0e+00 |

#### gabor.python_raw_defaults (new)

| ROI | filter (f0, theta rad) | pinned | fresh skimage run | rel diff |
|---|---|---|---|---|
| 0 | (4.0000, 0.0000) | 0.9775280898876404 | 0.9775280898876404 | 0.0e+00 |
| 0 | (16.0000, 0.7854) | 1.0 | 1.0 | 0.0e+00 |
| 0 | (32.0000, 1.5708) | 1.0112359550561798 | 1.0112359550561798 | 0.0e+00 |
| 0 | (64.0000, 2.3562) | 0.9101123595505618 | 0.9101123595505618 | 0.0e+00 |
| 1 | (4.0000, 0.0000) | 0.5874439461883408 | 0.5874439461883408 | 0.0e+00 |
| 1 | (16.0000, 0.7854) | 0.9955156950672646 | 0.9955156950672646 | 0.0e+00 |
| 1 | (32.0000, 1.5708) | 0.9282511210762332 | 0.9282511210762332 | 0.0e+00 |
| 1 | (64.0000, 2.3562) | 0.8340807174887892 | 0.8340807174887892 | 0.0e+00 |
| 2 | (4.0000, 0.0000) | 0.8494623655913979 | 0.8494623655913979 | 0.0e+00 |
| 2 | (16.0000, 0.7854) | 0.9731182795698925 | 0.9731182795698925 | 0.0e+00 |
| 2 | (32.0000, 1.5708) | 0.9516129032258065 | 0.9516129032258065 | 0.0e+00 |
| 2 | (64.0000, 2.3562) | 0.8387096774193549 | 0.8387096774193549 | 0.0e+00 |
| 3 | (4.0000, 0.0000) | 0.6649484536082474 | 0.6649484536082474 | 0.0e+00 |
| 3 | (16.0000, 0.7854) | 0.9432989690721649 | 0.9432989690721649 | 0.0e+00 |
| 3 | (32.0000, 1.5708) | 0.9639175257731959 | 0.9639175257731959 | 0.0e+00 |
| 3 | (64.0000, 2.3562) | 0.8814432989690721 | 0.8814432989690721 | 0.0e+00 |

Verdict for both tables: **vetted**, `rel=1e-3` (SPEC 7 same-definition tier).

## 3. The finding — the goldens were pinned at a config the documentation does not describe

`tests/test_gabor_truth.h` (now deleted; its values live in `test_2d_gabor_skimage.cc`) stated its
recipe as

```
'gabor_freqs' : [4.0, 16.0, 32.0, 64.0]
'gabor_thetas' : [0.0, 45.0, 90.0, 135.0]
```

Running the oracle at exactly that recipe does **not** reproduce the pinned values:

| ROI | pinned | oracle at the documented recipe | max abs diff |
|---|---|---|---|
| 0 | 1.011236, 0.921348, 0.966292, 0.617978 | 0.977528, 1.000000, 1.011236, 0.910112 | 0.292 |
| 1 | 1.004484, 0.932735, 0.112108, 0.174888 | 0.587444, 0.995516, 0.928251, 0.834081 | 0.816 |
| 2 | 1.005376, 0.978495, 0.370968, 0.000000 | 0.849462, 0.973118, 0.951613, 0.838710 | 0.839 |
| 3 | 1.005155, 0.958763, 0.484536, 0.046392 | 0.664948, 0.943299, 0.963918, 0.881443 | 0.835 |

What the pinned values *do* come from is the compiled-in default in `src/nyx/features/gabor.cpp`:

```cpp
std::vector<std::pair<double, double>> GaborFeature::f0_theta_pairs
{ {0, 4.0}, {M_PI_4, 16.0}, {M_PI_2, 32.0}, {M_PI_4*3.0, 64.0} };
```

read by `GaborFeature::calculate` as `f0 = pair.first`, `theta = pair.second`, i.e.
frequencies `{0, pi/4, pi/2, 3pi/4}` at angles `{4, 16, 32, 64}` **radians**. The oracle at that
configuration reproduces every pinned value to 0.

### 3.1 Root cause — two sites build the pair in opposite orders

| site | what it puts in the pair | consequence |
|---|---|---|
| `src/nyx/features/gabor.cpp` (static default) | `{angle, frequency}` — 0, pi/4, pi/2, 3pi/4 first | first filter has **f0 = 0**, a degenerate flat window; angles are 4..64 radians |
| `src/nyx/cli_gabor_options.cpp:37` | `std::pair p (f[i], deg2rad(t[i]))` — `{frequency, angle}` | the documented behaviour |
| `src/nyx/features/gabor.h:62` (the member's comment) | "Pairs of orientation angles … and frequency" | agrees with the static default, contradicts the parser and every consumer |
| `GaborFeature::calculate`, `calculate_gpu`, `calculate_gpu_multi_filter` | read `.first` as f0, `.second` as theta | the parser's order is the one the code consumes |

So the two entry points diverge, and the divergence is user-visible:

- **Python API** — the wrapper always passes its defaults (`gabor_freqs=[4,16,32,64]`,
  `gabor_thetas=[0,45,90,135]`) through `GaborOptions::parse_input`, so the static default is
  overwritten before any ROI is processed. Measured on the same four fixture ROIs with the built
  `nyxus.backend`, `Nyxus(["GABOR"])` returns exactly the `gabor.python_raw_defaults` column above
  (0.977528, 1.000000, 1.011236, 0.910112 for ROI 0, …), and `get_params()` reports
  `gabor_freqs=[4.0, 16.0, 32.0, 64.0]`, `gabor_thetas=[0.0, 45.0, 90.0, 135.0]`.
- **CLI** — `rawFreqs`/`rawTheta` default to empty strings and `parse_input` only rebuilds the pairs
  when *both* are non-empty, so `nyxus` without `--gaborfreqs`/`--gabortheta` keeps the static
  default and computes the other column.
- **gtest** — calls `GaborFeature::calculate` directly, so it also sees the static default. That is
  why the goldens are the static-default values while the comment described the API defaults.

This is a defect in `gabor.cpp`, not in the test: whichever order is intended, the two defaults
should not disagree, and a default whose first filter has frequency 0 (an unmodulated flat window,
which is also the baseline filter) is not a plausible intent. It changes public feature values, so
per the vetting-series rule it is **measured and documented here and deferred to its own branch**,
not fixed in a vetting PR.

### 3.2 What the tests now assert

Both configurations are pinned and asserted, so neither entry point is left uncovered:

| recipe | table | test |
|---|---|---|
| `gabor.cpp_static_defaults` | `gabor_2d_skimage_cpp_static_defaults_ref_vals` | `TEST_NYXUS.TEST_2D_GABOR_CPP_STATIC_DEFAULTS_SKIMAGE` |
| `gabor.python_raw_defaults` | `gabor_2d_skimage_python_raw_defaults_ref_vals` | `TEST_NYXUS.TEST_2D_GABOR_PYTHON_RAW_DEFAULTS_SKIMAGE` |

The second test installs the parser-built pairs into the process-wide
`GaborFeature::f0_theta_pairs` through an RAII guard, so the compiled-in value is put back on every
exit path including a failing assertion, and test order stays irrelevant. If the `gabor.cpp` default
is corrected on its own branch, the first table is the one that must be regenerated, and the two
tables will then hold the same values.

Neither name says "documented". `gabor_freqs` is documented as denominators of pi in
`src/nyx/python/nyxus/nyxus.py` and in the CLI help, and no code path divides by pi — the parser
stores `4` and `Gabor()` reads it as `lambda = 2*pi/4`. The recipe is therefore
`gabor.python_raw_defaults`, named for the numbers that reach the filter; §8 item 3 carries the
unresolved contract. The GPU path is asserted separately and size-only in
`tests/test_2d_gabor_mechanics.h`, so no case named `_skimage` runs a branch that has no oracle.

## 4. Is the oracle independent, or a re-encoding of Nyxus?

The circularity question this framework insists on. Answer, stated exactly:

- **The kernel is skimage's.** `skimage.filters.gabor_kernel(frequency=f0/2pi, theta=theta,
  sigma_x=sig2lam*2pi/f0, sigma_y=sigma_x/gamma, offset=0)`, cropped to the 16×16 Nyxus grid and
  L1-normalized, is what produces every pinned value. Cross-checked against the hand-derived
  canonical Gabor closed form at all seven non-degenerate (f0, theta) points used by the two
  recipes: `max |kernel diff|` 1.7e-18 … 1.1e-16.
- **The scoring pipeline is Nyxus' own definition, reproduced.** skimage has no equivalent of the
  WND-CHARM Gabor score (count of response pixels above `GRAYthr × baseline_max`, over the baseline
  count), so there is no native function to compare the *feature* against — only the filter. The
  claim this family can honestly carry is therefore "the filter is the canonical Gabor filter, and
  the score is the documented count ratio over it", which is what the registry notes now say.
- **The `f0 = 0` filter is not skimage's.** `gabor_kernel` divides by the frequency to size its own
  support and cannot express frequency 0. At f0 = 0 the envelope and carrier are both identically 1,
  so that kernel is the flat window in closed form. It occurs only in `gabor.cpp_static_defaults`.

### 4.1 How much of the recipe is Nyxus' own convention — measured, not asserted

The three legs the framework asks for (Nyxus / our reimplementation / the tool's own machinery) were
all run. A and B agree exactly, as tabled above. **C — scoring off `skimage.filters.gabor()`, i.e.
skimage's own kernel support, border mode and convolution — does not reproduce the feature at all.**
This leg is generator **part D2**, so the numbers below are re-derived on every run rather than
quoted from a one-off script:

| config | border mode | max abs diff vs pinned | values saturated at 0 or ≥1 |
|---|---|---|---|
| `gabor.cpp_static_defaults` | `constant` (zero pad, as Nyxus) | **1.005** | 16/16 |
| `gabor.python_raw_defaults` | `constant` | **0.417** | 8/16 |
| `gabor.cpp_static_defaults` | `reflect` (skimage default) | 1.005 … 1.011 | varies, 11–15/16 |
| `gabor.python_raw_defaults` | `reflect` | 0.964 … 1.011 | varies, 12–15/16 |

i.e. up to the entire range of the feature. **The cause is the 16×16 kernel crop, not the border
mode.** Zero padding does help `gabor.python_raw_defaults` — ≈1.0 at `reflect` down to 0.417 — but
0.417 is still 40% of the feature's range, and for `gabor.cpp_static_defaults` matching Nyxus' border
buys nothing at all: 1.005 either way. `gamma = 0.1` is what makes the crop dominant:
`sigma_y = sigma_x / gamma` is ten times `sigma_x`, so the analytic kernel is hundreds of pixels wide
at the low frequencies while the fixture ROIs are 9×10 … 17×11:

| f0 | sigma_x | sigma_y | skimage's own support | L1 mass inside Nyxus' 16×16 |
|---|---|---|---|---|
| 0.7854 | 6.40 | 64.00 | 369×113 | **7.90%** |
| 1.5708 | 3.20 | 32.00 | 163×107 | 21.14% |
| 2.3562 | 2.13 | 21.33 | 53×119 | 31.48% |
| 4.0000 | 1.26 | 12.57 | 77×9 | 47.64% |
| 16.0000 | 0.31 | 3.14 | 15×15 | 100.00% |
| 32.0000 | 0.16 | 1.57 | 3×11 | 100.00% |
| 64.0000 | 0.08 | 0.79 | 5×5 | 100.00% |

Untruncated, the low-frequency filters are wider than the image and act as near-uniform smoothers,
so the score saturates: at `constant` the whole `gabor.cpp_static_defaults` table comes back at the
baseline ratio, 16 values out of 16.

**The `reflect` rows are a range because they are not reproducible.** Running the same control five
times on the same input gives `gabor.python_raw_defaults` maxima of 0.964, 0.996, 0.996, 1.011 and
1.011. Isolated to one call: `skimage.filters.gabor(img, …, mode='reflect')` on the 9×10 ROI0 with
the baseline filter — `sigma_x = 50.3`, `sigma_y = 502.7`, so a kernel roughly 3000 px a side against
a 10 px image — returns responses that differ by up to **7.7e-4** between runs, on values of order
0.33. The same call at `mode='constant'` is bit-identical across runs. Reflect-padding an image to
many times its own extent is where it goes; the varying response then flips individual pixels across
the `GRAYthr` cut, and the score moves in whole 1/baseline_count steps. Only the `constant` rows are
quoted as measurements anywhere in this tree, and the generator prints the instability rather than
hiding it. This is also a second reason the `reflect` configuration could not serve as an oracle even
if the crop question were settled.

**What this means for the claim.** The vetting is exact at kernel level (§4 bullet 1) and exact at
value level *given Nyxus' truncate-to-16×16 + zero-padded-full-convolution convention*. It is not
evidence that an independent Gabor implementation would report these numbers — with skimage's own
conventions it would not, by up to 1.005. That convention is therefore recorded as part of both
recipes in `config_recipes.md` rather than left as an implementation detail. Generator part D1 prints
the mass table and part D2 re-runs the native-filtering control on every run, so both halves of the
statement stay measured rather than remembered.

**The registry says so too.** `audit/gabor_2d_coverage.csv` lists this family as `skimage;analytic`,
not `skimage`, and both `oracle_coverage.csv` rows carry the split in `notes`: kernel from skimage,
f0=0 kernel and count-ratio score analytic. The `oracle` column itself stays `skimage` because
`check_coverage.py` takes one SPEC §4 token per row — the same arrangement SPEC §4 already documents
for `matlab`, where the token names the semantics and every artifact repeats what produced the
numbers.

## 5. Tolerance

`rel=1e-3` (SPEC 7 same-definition tier), asserted as `agrees_gt(..., 1000.)`.

The measured agreement is exact, so the band is not covering a residual. It is also not loose enough
to hide a real disagreement: the feature is a ratio of pixel counts, so the smallest possible wrong
answer differs by one counted pixel, i.e. by `1/baseline_count`. Measured on the fixture
(generator part C): baseline counts 89, 223, 186, 194 → one pixel is **4.5e-3 … 1.1e-2**, four to
eleven times the tolerance.

## 6. Scope limits

- **The GPU path is not vetted.** `calculate_gpu` (FFT-based convolution) diverges from the
  direct-convolution CPU path on these ROIs, so the size-only GPU run lives in
  `tests/test_2d_gabor_mechanics.h` as `TEST_NYXUS.TEST_2D_GABOR_GPU_RUNS_MECHANICS`, rather than
  inside a case named for an oracle. The Python-side `test_gabor_gpu`
  (`tests/python/test_nyxus.py`, `skip_ci`) compares GPU against CPU and asserts no oracle value.
- **No full-feature oracle.** scikit-image supplies the kernel and nothing else; the score has no
  second implementation here. SPEC §4 names `wndcharm` (Nyxus' own lineage) and `feature2djava` as
  the right oracles for this class of feature, and neither is built in this tree. Recorded in
  `not_covered.md`.
- **One fixture.** Four DSB2018 ROIs, whole image as one ROI, no anisotropy.
- The out-of-core (`osized_`) Gabor path is a no-op stub for pixel feed and is not covered here.

## 7. Registry corrections made

| what | before | after |
|---|---|---|
| rows | 1 (`GABOR`) | 2 — one per config point |
| `config_recipe` | the generic "Not mode-specific …" boilerplate | `gabor.cpp_static_defaults` / `gabor.python_raw_defaults`, both defined in `config_recipes.md` |
| `current_test` | `test_2d_gabor_skimage.cc;test_nyxus.py` | `test_2d_gabor_skimage.cc` — `test_nyxus.py` asserts no GABOR value (it is a GPU-vs-CPU equality test plus a parameter-validation test), so crediting it was an over-credit |
| `target_test` | `test_2d_gabor_skimage.h` | empty — the file exists and holds only declarations; the assertions are in the `.cc` |
| `source` | `audit` | `in-tree` |
| `notes` | carried the history of the response-truncation fix | current-state scope statements only; the history lives in this report |

## 8. Other findings, for other branches

1. **`gabor.cpp` static default vs parser order** — §3.1. Public feature values change; own branch.
2. **`GaborFeature::get_theta_in_degrees(int)`** (`gabor.cpp:645`) has no callers anywhere in the
   tree — only its definition and its declaration. It returns `pair.second` in degrees, so with the
   compiled-in default it would report 229°, 917°, 1833°, 3667°. Dead code that would mislead the
   moment it is used; remove it or fix it together with §1.
3. **The `gabor_freqs` units contract is stated three ways and implemented as none of them.**
   `src/nyx/python/nyxus/nyxus.py:103` documents `gabor_freqs` as "comma-separated denominators of
   `\pi`" with a default of `[4, 16, 32, 64]`; `src/nyx/environment.cpp:219` documents the same flag
   as denominators of pi with a *different* default, `1,2,4,8,16,32,64`; and
   `src/nyx/cli_gabor_options.cpp:37` stores whatever number it is given, which
   `GaborFeature::Gabor` consumes as `lambda = 2*pi/f0`. Nothing in the tree divides by pi. So a
   documented `4` is 4 angular units, wavelength `pi/2` — near Nyquist on a 16 px kernel — where the
   documented reading would give wavelength 8 px. Which is intended decides public feature values, so
   it is recorded rather than fixed here, and it is why the second recipe is named
   `gabor.python_raw_defaults`: no test in this tree may be named for a contract nothing implements.
4. **`skimage.filters.gabor(mode='reflect')` is not reproducible when the kernel exceeds the image.**
   §4.1 — same input, same process, responses differing by 7.7e-4 between runs on a 9×10 ROI with a
   ~3000 px kernel; `mode='constant'` is bit-stable. It affects no pinned value (the oracle path uses
   `gabor_kernel` + `convolve2d`, not `gabor`), but it is worth an upstream report, and any future
   generator that reaches for `skimage.filters.gabor` on small ROIs should know.
5. **`oracle_coverage.csv` line 566, `3D,3ROBUST_MEAN`** carries an unquoted comma in `notes`, so
   the row parses as 15 fields. `check_coverage.py` does not notice (it reads with `DictReader`,
   which sweeps the surplus into a `None` key). Any full-file CSV rewrite would corrupt it — this
   pass therefore edited the GABOR line textually and left every other byte alone. Worth a one-line
   quoting fix plus a field-count assertion in the checker.
