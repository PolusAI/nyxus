# Audit: 2D moments vs a fresh scikit-image run

**Verdict: 118 of 180 rows are vetted against scikit-image; 62 stay `regression` for a measured
reason.** The 32 rows that carried a `vetted` verdict with no in-tree oracle assertion
(`NORM_SPAT_MOMENT_ij`, `IMOM_NRM_ij`) now have one, and they reproduce skimage **exactly** - a
relative difference of 0.0e+00 on all 32.

Covers `tests/test_2d_moments_skimage.h`, `tests/test_2d_moments_common.h` (fixtures) and
`tests/vetting/oracles/gen_moments_skimage.py`.

## Method

- **Tool**: scikit-image **0.26.0**, numpy 2.4.6, conda env `nyxus_mirp`
  (`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy scikit-image`).
- **Fixtures**: the pinned 48x40 rectangle of `test_2d_moments_common.h` -
  `I(x,y) = 10 + 3x + 5y + (x*y)%7` - and the thin right wedge (`5y <= x`) for the Hu invariants.
- **Convention**: Nyxus `m_pq` has `p` on x and `q` on y, so the generator indexes arrays `A[x, y]`
  and skimage's `moments(A)[i, j]` lands on `i == p`, `j == q` with no transposition and no h7 sign
  flip. Getting this wrong silently swaps `m_12` with `m_21`.
- **Command**: `conda run -n nyxus_mirp python tests/vetting/oracles/gen_moments_skimage.py`.

The fixture reconstruction is self-checking: skimage's `m_00` comes out 1920 on the shape fixture
and 346635 on the intensity one, which are exactly the pinned `SPAT_MOMENT_00` / `IMOM_RM_00`.

**Every** golden in `test_2d_moments_skimage.h` is re-verified against the run, not a sample: 142 of
142 across both fixtures, none without a skimage counterpart. The generator previously validated a
hand-picked list of 25 names, which is the kind of subset that stops covering whatever is added
later.

## The 32 rows this PR closes

`NORM_SPAT_MOMENT_pq` and `IMOM_NRM_pq` are **normalized raw moments**:

```
NORM_SPAT_MOMENT_pq = m_pq / m_00^((p+q)/2 + 1)
```

skimage has **no native function for this**. Its `moments_normalized()` applies that same exponent
to the **central** moments, which is a different quantity - and one Nyxus already exposes separately
as `NORM_CENTRAL_MOMENT_pq` / `IMOM_NCM_pq`. Measured on the shape fixture:

| p,q | Nyxus `NORM_SPAT_MOMENT` | skimage `moments_normalized` | Nyxus `NORM_CENTRAL_MOMENT` |
|---|---|---|---|
| 2,0 | 0.3875868055555556 | 0.09995659722222222 | 0.09995659722222222 |
| 0,2 | 0.2674479166666667 | 0.06940104166666666 | 0.06940104166666666 |

So the two families are not interchangeable, and the right column confirms which one
`moments_normalized` actually answers.

**Why this is still a skimage verdict and not a circular one.** The non-trivial part - the moment
sums `m_pq` themselves - is skimage's, computed by an independent implementation from the same
pixels. The only step layered on top is the normalization exponent `(p+q)/2 + 1`, which is not a
Nyxus invention: it is the exponent skimage's own `moments_normalized` uses, just applied to the raw
matrix instead of the central one. Result over all 32 features, both fixtures:

```
worst relative difference: 0.0e+00
```

The generator asserts, as a guard, that normalized-raw and `moments_normalized` are numerically
**distinct**, so a later edit cannot quietly swap one for the other and still pass.

Contrast with the intensity-histogram percentiles demoted in the sibling PR: there, Nyxus' quantity
matched *none* of eighteen native percentile implementations, so the closed form was only a
restatement of Nyxus itself. Here the tool supplies the quantity and agreement is exact.

## The 62 rows that stay `regression`

All 62 are the distance-to-contour **weighted** moments (`WEIGHTED_SPAT_MOMENT_*`,
`WEIGHTED_CENTRAL_MOMENT_*`, `WT_NORM_CTR_MOM_*`, `WEIGHTED_HU_M*`, and the `IMOM_W*` intensity
twins). Nyxus weights each pixel by `I(p) * log(dist(p, contour) + eps)`, `eps = 0.001`, and takes
`dist` from `Pixel2::min_sqdist` - an approximate hill-descent whose own doc comment warns that it
overestimates and says to use `exact_min_sqdist` where correctness matters.

Re-measured against current code on the pinned 48x40 fixture. For a solid rectangle the contour is
the outermost ring, so the exact Euclidean distance from a pixel to the contour is
`min(x, y, W-1-x, H-1-y)` and needs no distance transform:

| | `WEIGHTED_SPAT_MOMENT_00` |
|---|---|
| exact distance to contour | 1821.28398639964 |
| Nyxus (pinned) | 2498.7137504625134 |
| ratio | **1.372x**, i.e. a 37.2% overestimate |

These values therefore match no external tool by construction, which is why they are pinned as drift
guards and claim no oracle. **Fixing it is deliberately out of scope here**: it changes 62 published
feature values and swaps an O(pixels x contour) hill-descent for a distance transform, which is a
behaviour and performance change, not a vetting change.

One correction to the record for whoever takes that on: the blast radius is smaller than previously
noted. There is exactly **one live** weighting path,
`BasicGeomoms2D::apply_dist2contour_weighting` (`src/nyx/features/2d_geomoments_basic.cpp`), plus a
whole-slide variant. A second implementation using a *different* formula, `inten / (dist + eps)`,
exists at `src/nyx/features/image_matrix.cpp` - and is **called by nothing in the repository**; and
no GPU kernel performs dist-to-contour weighting at all.

## What this report does and does not establish

The in-tree goldens were emitted by the generator named above, so "golden == fresh run" only shows
the pin is reproducible. The vetting claim rests on the **Nyxus vs skimage** comparison: an
independent implementation of the same published definitions, on the same pixels, in a coordinate
convention that is stated and checked rather than assumed.
