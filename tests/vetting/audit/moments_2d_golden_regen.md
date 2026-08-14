# Regenerating the 2D moments goldens

Every number pinned in `tests/test_2d_moments_skimage.h` comes out of a checked-in generator;
nothing there is transcribed by hand. `tests/test_2d_moments_regression.h` is a snapshot file and is
covered at the end. Run everything offline - CI never invokes a reference tool.

## 1. Stand the tool up

```bash
conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy scikit-image
conda run -n nyxus_mirp python -c "import skimage, numpy; print(skimage.__version__, numpy.__version__)"
# -> 0.26.0 2.4.6
```

Record what the solver resolved next to the goldens; the generator's provenance header names the
version it was last run at.

## 2. The fixtures

Both are built in the generator, not read from disk, and mirror
`tests/test_2d_moments_common.h` exactly:

- **48x40 rectangle** - `I(x,y) = 10 + 3x + 5y + (x*y)%7`, every pixel in the ROI. Used for the raw,
  central, normalized and intensity moment families.
- **Thin right wedge** - `0<=x<40`, `0<=y<8`, `5y <= x`. Elongated *and* skewed, so the odd-order
  etas are large; this is the fixture that can discriminate an h5/h6 formula slip, which the
  symmetric rectangle cannot (its odd etas vanish, so a wrong formula agrees there).

## 3. Run the generator

```bash
conda run -n nyxus_mirp python tests/vetting/oracles/gen_moments_skimage.py
```

It prints `ALL CHECKS PASSED` or `SOME CHECKS FAILED -- do not paste goldens`, and emits paste-ready
table bodies. Sections:

| section | what it emits |
|---|---|
| A | validation that the oracle reproduces every golden the tree already pins |
| B | the weighted Hu snapshot values (skimage runs the Hu formula over Nyxus' pinned eta) |
| C | the wedge-fixture goldens, with a discriminance print for h5/h6 |
| D | the normalized **raw** moments: `NORM_SPAT_MOMENT_pq`, `IMOM_NRM_pq` |
| E | re-verification of **every** golden pinned in `test_2d_moments_skimage.h` (142, both fixtures) against this run |

Paste a section's body into the matching `ref_vals_list` in `test_2d_moments_skimage.h`, keep the
comment above it, rebuild `runAllTests`, run `--gtest_filter=*MOMENTS*`.

## 4. The coordinate convention (the thing most likely to go wrong)

Nyxus `m_pq` has **p on x and q on y**. The generator therefore indexes its arrays `A[x, y]`, so
`skimage.measure.moments(A)[i, j]` lands on `i == p`, `j == q` with no transposition - and Hu's h7
keeps its sign. Build the array the other way round and `m_12`/`m_21` swap silently: the totals
still look plausible and only the asymmetric fixture will catch it.

## 5. Mapping skimage to Nyxus features

| Nyxus | skimage |
|---|---|
| `SPAT_MOMENT_pq` / `IMOM_RM_pq` | `moments(A)[p, q]` - shape fixture / intensity fixture |
| `CENTRAL_MOMENT_pq` / `IMOM_CM_pq` | `moments_central(A)[p, q]` |
| `NORM_CENTRAL_MOMENT_pq` / `IMOM_NCM_pq` | `moments_normalized(mu)[p, q]` |
| `HU_M1..7` / `IMOM_HU1..7` | `moments_hu(nu)` |
| `NORM_SPAT_MOMENT_pq` / `IMOM_NRM_pq` | **no native function** - `m[p,q] / m[0,0]**((p+q)/2+1)` |
| `WEIGHTED_*`, `WT_NORM_CTR_MOM_*`, `IMOM_W*` | **none** - see section 7 |

## 6. Convention differences to account for

- **Normalized raw vs normalized central.** `moments_normalized()` is defined on the *central*
  moments. Nyxus applies the same exponent to the *raw* moments and calls that
  `NORM_SPAT_MOMENT_pq`. They are different numbers (0.3876 vs 0.0999 at p=2,q=0 on the rectangle),
  and Nyxus exposes the central one separately. Section D asserts they are distinct so the two
  cannot be silently interchanged.
- **`moments_normalized` returns NaN for order < 2**; the generator runs it through
  `np.nan_to_num` before the Hu step, matching what Nyxus feeds its own Hu implementation.
- **h5/h6 are the discriminating invariants.** Two classic formula slips - squaring
  `3*(eta30+eta12)` instead of tripling the square, and losing the `(eta21+eta03)` grouping in h6 -
  are invisible on a symmetric fixture. Section C prints `|slip - correct|` against the gtest
  tolerance so the wedge fixture's discriminance is proven, not assumed. A tell-tale of the h6 slip
  is `HU6` coming out equal to `NCM_03`.
- **`moments_central(order=N)` leaves every entry with `p+q > N` at zero.** The tables pin moments
  up to `p+q = 6` (`CENTRAL_MOMENT_33`, `IMOM_CM_33`, ...), so section E computes its matrices at
  order 6; verifying those against an order-3 matrix compares them to 0 and "fails" for no reason.
  `moments()` does fill the full matrix, which is why only the central families show it.
- **Noise-level pins.** Several Hu goldens are ~1e-10 or smaller, i.e. summation noise around an
  exact zero. Where the oracle is exactly 0 the golden is 0; do not "restore" a tiny non-zero pin.

## 7. The goldens no generator produces

`tests/test_2d_moments_regression.h` pins the **weighted** (distance-to-contour) families. There is
no oracle for them: the weighting uses an approximate distance that overestimates the true one by a
measured 1.372x on this fixture, so no external tool reproduces the values - see
`moments_2d_skimage_vetting_report.md`. They are drift guards. Section B of the generator does
recompute the weighted *Hu* values, but only by running skimage's Hu formula over Nyxus' own pinned
eta - that checks the Hu step, not the weighting, and must not be read as vetting the weighted
family.
