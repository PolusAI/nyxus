# Audit: SHARPNESS vs the reference DOM implementation

**Verdict: the candidate oracle is refuted.** Nyxus returns **2.1904708385718963** where the
published reference returns **0.54592951157710823** on the same fixture — a factor of four, and
structural rather than numerical. `SHARPNESS` stays `regression` with no oracle claim.

This closes a registry row that read `candidate_oracle = "reference DOM sharpness (Kumar et al.
2012)"`, `flag = promote-after-deepdive`, `notes = "Correct (all-double DOM sharpness port, no
overflow bug); left regression pending an independent reference-DOM oracle to confirm the
median-blur/edge/contrast parameterization."` The deep dive has now happened. The parameterization is
not what needed confirming.

Covers `tests/test_imq_regression.h` (the `SHARPNESS` pin) and
`tests/vetting/audit/imq_sharpness_reference_dom.py` (the measurement).

## Method

- **Reference**: Kumar, Chen, Doermann, *Sharpness estimation for document and scene images*
  (ICPR 2012), as implemented in `https://github.com/umang-singhal/pydom`, file `dom/dom.py`, entry
  point `DOM.get_sharpness` with its defaults `width=2`, `sharpness_threshold=2`,
  `edge_threshold=0.0001`. **The reference is invoked, not vendored**: pydom is GPL-3.0 and Nyxus is
  MIT, so no line of it lives in this repository. `imq_sharpness_reference_dom.py` imports the
  installed package and calls its public API — `get_sharpness` for the score, and
  `load`/`edges`/`sharpness_matrix` for the intermediates tabulated below — then asserts the
  intermediates recompose the score, so the diagnostics describe the same run the entry point
  produced. The package is not on PyPI under a usable name (`pip install dom` fetches an unrelated
  domain-lookup CLI), so it is installed from git into the offline audit env only:
  `pip install git+https://github.com/umang-singhal/pydom.git`. It is not a Nyxus dependency and CI
  never invokes this script.
- **Nyxus side**: the same script carries a line-for-line Python port of
  `SharpnessFeature::sharpness()`. That port is not decoration — it is what makes the comparison a
  statement about the shipped C++ rather than about a script, and the script asserts it: the port
  reproduces the golden pinned in `test_imq_regression.h` to **2.03e-16 relative**
  (2.1904708385718958 against 2.1904708385718963). If `sharpness.cpp` changes, that check fails and
  this report has to be re-derived.
- **Fixture**: the same 8×12 ROI, parsed out of `test_data.h`.
- **Command**: `python tests/vetting/audit/imq_sharpness_reference_dom.py`, in `nyxus_mirp`.

## Result

| | Nyxus | reference | |
|---|---:|---:|---|
| SHARPNESS | 2.1904708385718963 | 0.54592951157710823 | rel 3.01 (**301%**) |
| Rx | 2.1621 | 0.5000 | |
| Ry | 0.3514 | 0.2192 | |
| numerator, x | 157.833 (a **sum**) | 28 (a **count**) | |
| numerator, y | 19.679 (a **sum**) | 16 (a **count**) | |
| edge pixels, x | 73 | 56 | |
| edge pixels, y | 56 | 73 | |

## The six differences

Each is a statement about a specific line, and together they account for the factor of four. None is
a tolerance question.

**1. The aggregation is a different statistic.** The reference counts the pixels whose sharpness
matrix reaches `sharpness_threshold` (default 2) — measured here as 28 and 16. Nyxus sums the
sharpness values themselves:

```cpp
auto n_sharpx = std::accumulate(sx.begin(), sx.end(), 0.);
```

and has no `sharpness_threshold` parameter anywhere. `Rx` is then `n_sharp/n_edge` in both, but in
the reference it is a *fraction of edge pixels that are sharp* — bounded by 1, which is what makes
the paper's `0 < S < sqrt(2)` bound hold — and in Nyxus it is a mean sharpness magnitude, which is
not bounded by anything. Nyxus' 2.19 is already above `sqrt(2)`.

**2. `Sy` is summed along the wrong axis.** The reference's `sharpness_matrix` computes `Sx` over row
windows and `Sy` over *column* windows — it walks `domy`'s columns and sums each window along axis 1.
Nyxus runs the identical row-wise loop for both, differing only in which `dom`/`contrast` field it
reads. The y half of the measure is therefore not the reference's y half at all.

**3. The edge maps are assigned to opposite axes.** The reference builds `edgex` from the
**column** convolution (`smoothenImage(image, transpose=True)`) and `edgey` from the row one. Nyxus'
`smooth_image()` returns the row-convolved image first and calls it `smooth_x`. The measurement
shows it cleanly: Nyxus `edge_x`=73, `edge_y`=56; reference `edgex`=56, `edgey`=73 — the same two
numbers, exchanged.

**4. One maximum normalizes both smoothed images.** The reference normalizes each smoothed image by
its own maximum inside `smoothenImage`. Nyxus takes `max` from `smoothed` (the row-convolved one)
and divides both by it, so the transposed image is scaled by a maximum that is not its own — which
shifts which of its pixels clear `edge_threshold`.

**5. `Sx`/`Sy` are never masked by the edge maps before aggregating.** The reference applies the mask
twice — once to the contrast terms and again to `Sx`/`Sy` immediately before counting. Measured: the
raw matrices out of `sharpness_matrix` hold 50 and 28 pixels at or above the threshold, and masking
drops those to the 28 and 16 the score is built from. Nyxus applies the mask only to `cx`/`cy`, so
sharpness computed at non-edge pixels reaches the numerator.

**6. The last `width` columns are never written.** Nyxus fills `sx`/`sy` for `k < cols - width`; the
reference fills every column. At `width=2` on an 8-wide ROI that is a quarter of each row dropped.

## Two more, smaller

Neither changes the six above, and both are worth recording because they would otherwise be
rediscovered:

- **`contrast()` uses the forward difference** `|Im[i+1] - Im[i]|` where the reference uses the
  backward `|Im[i] - Im[i-1]|`. The two fields are the same set of values shifted by one row (and one
  column), which offsets the contrast field against the DOM field the window sums pair it with.
- **`median_blur()` pads by `(rows, cols)`** rather than by `(ksize-1)/2`, building a 3×-sized image
  to take a 3×3 median over, and its `remove_padding()` ends with
  `img.erase(img.begin() + img_row*img_col, img.end())` where `img_row*img_col` is already the full
  padded size — so the erase is a no-op and the blurred vector keeps a 768-element tail that nothing
  reads. Correct output, ~9× the work and ~9× the memory.

## What this means for the row

`SHARPNESS` keeps `status = regression`. The pin moves from the five-digit `2.19047` to the full
`%.17g` value `2.1904708385718963` — the truncated pin sat 3.8e-7 relative from the value it was
guarding, which ate 0.04% of the old `rel=1e-3` band before the test started — and the band tightens
to an absolute 2.2e-9, i.e. `rel=1e-9` at this magnitude.

`candidate_oracle` becomes "reference DOM sharpness (Kumar et al. 2012) — measured and refuted" and
`flag` becomes `impl-defect`. Promotion is not blocked on finding an oracle; it is blocked on
deciding which of the six differences are bugs. Difference 1 is the one that decides the others:
until Nyxus counts above a threshold rather than summing, its `SHARPNESS` is not the DOM measure
whatever the rest of the pipeline does.

## Reproduction

```
conda activate nyxus_mirp                     # numpy 2.4.6, cv2 4.13.0, python 3.11.15
pip install git+https://github.com/umang-singhal/pydom.git    # the reference, GPL-3.0, audit env only
python tests/vetting/audit/imq_sharpness_reference_dom.py
```

The script has two checks that can fail, and both are meant to: the port must still reproduce the
pinned golden, and the reference must still disagree. If the second one ever fails, `SHARPNESS`
became promotable and this report is out of date.
