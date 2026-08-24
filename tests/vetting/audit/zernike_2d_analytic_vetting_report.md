# ZERNIKE2D vs the closed form — vetting report

**Verdict: promoted to vetted, after fixing a defect that made all 30 values wrong.**

`ZernikeFeature::mb_zernike2D` seeded its angular lookup tables one repetition off. Every Zernike
moment of repetition `m` was computed with `cos((m+1)θ)` and `sin((m+1)θ)` instead of `cos(mθ)` and
`sin(mθ)`, so **every one of the 30 published `ZERNIKE2D` magnitudes was wrong** — the 29 non-zero
ones by **0.052% to 347%** against the closed form (median 48%), and the one that should be zero by
an amount no relative error describes: it read 0.0358.

With the seeding corrected, Nyxus' Singh & Walia recurrence reproduces the closed form to
**3.4e-15 worst case**, and a third implementation (centrosome's) agrees with both to 4.9e-15. The
row moves from `status=regression` with no oracle to `status=vetted, oracle=analytic, rel=1e-12`.

- Fixture: `shape2d_morphology_{intensity,mask}` (`test_data.h`) — one 26-pixel concave ROI with an
  interior hole, total intensity 1048. Recipe `zernike.shape2d_native`.
- Reference: the closed form, stdlib only — no pinned tool version. Corroborated against
  `centrosome` 1.2.3 and `cellprofiler` 4.2.8, python 3.9.
- Generator: `tests/vetting/oracles/gen_zernike_analytic.py`, with `--centrosome` and
  `--cellprofiler` for the two cross-checks.
- Regeneration: `tests/vetting/audit/zernike_2d_golden_regen.md`.

---

## 1. The defect

`src/nyx/features/zernike.cpp`, in the per-pixel loop of `mb_zernike2D`:

```cpp
/* compute COST SINT and save in tables */
a = COST[0] = x / r;          // cos(theta)
b = SINT[0] = y / r;          // sin(theta)
for (m = 1; m <= L; m++)
{
    COST[m] = a * COST[m - 1] - b * SINT[m - 1];
    SINT[m] = a * SINT[m - 1] + b * COST[m - 1];
}
```

The recurrence is the angle-addition formula and is correct as written — line 285 reads `COST[m-1]`
*before* line 284's write to `COST[m]` can matter, so there is no aliasing bug. The error is the
starting point. Seeded at `cos θ`, the tables hold

    COST[m] = cos((m+1)·theta),   SINT[m] = sin((m+1)·theta)

and they are then consumed as the repetition-`m` angular term:

```cpp
AR[n][m] += const_t * Rnm * COST[m];
AI[n][m] -= const_t * Rnm * SINT[m];
```

A moment of repetition `m` needs `cos(mθ)`. It got `cos((m+1)θ)`. The fix seeds the tables at
`m = 0` — `COST[0] = 1`, `SINT[0] = 0` — after which the same recurrence yields `COST[1] = cos θ`,
`COST[2] = cos 2θ`, i.e. `COST[m] = cos(mθ)`.

**Scope: one site.** `osized_calculate` rebuilds the ROI's dense image and delegates to
`calculate()`, so the out-of-core path inherits the fix. `mb_Znl`, the other moment generator in the
file, is defined but never called.

## 2. Two proofs that need no reference implementation

Both are forced by the definition, and both are now asserted in `test_2d_zernike_invariant.h`.

| identity | why it must hold | before the fix | after |
|---|---|---:|---:|
| `A(0,0) = 1/π` | `R₀₀ = 1`, the weights are `I/ΣI`, and every bounding-box pixel is inside the disk on this fixture, so the weights sum to exactly 1 | **0.020497** | **0.31830988618379069** |
| `A(1,1) = 0` | it *is* the first moment about the point the disk is centred on, and `mb_zernike2D` centres on the intensity centroid | **0.035831** | **1.25e-17** |

`1/π = 0.31830988618379069`. The corrected value is that number to every digit printed.

That the fixture puts every pixel inside the disk is not assumed — it is asserted in
`test_2d_zernike_every_pixel_is_inside_the_unit_disk_mechanics`, because the `A(0,0) = 1/π` identity
depends on it and a future fixture could break it without breaking anything else.

## 3. The closed form, and why the comparison is not circular

The reference sums the standard factorial series directly:

    R_nm(r) = sum_k (-1)^k (n-k)! r^(n-2k) / ( k! ((n+m)/2-k)! ((n-m)/2-k)! )
    A_nm    = (n+1)/pi * sum_pixels (I / sum I) * R_nm(r) * exp(-i*m*theta)

evaluated at **Nyxus' own geometry**: the unit disk centred on the ROI's intensity centroid with
radius `min(bbox width, bbox height)`, pixels outside the disk dropped.

`zernike.cpp` never evaluates that series. It uses the Singh & Walia three-term recurrence, with the
`H1`/`H2`/`H3` coefficient tables. So the two share the mathematics and nothing else — which is
precisely the distinction SPEC §5.2 asks for. A match says the recurrence computes the polynomial it
claims to; it is not the fixture and the assertion encoding the same procedure.

Taking the geometry from Nyxus is deliberate and is stated rather than hidden: the disk's centre and
radius are a **convention choice**, not a fact about Zernike moments, and they are pinned separately
in `test_2d_zernike_mechanics.h`. What is under test here is the moment computation given that
convention.

**What this does not establish.** The reference takes the centre and the radius as *inputs*, read out
of the mechanics header — which is to say, out of Nyxus' own output. So the vetting covers the moment
computation and not the disk it is computed on. **Nothing external endorses `radius = min(bbox width,
bbox height)` centred on the intensity centroid**: CellProfiler uses the minimum enclosing circle,
and the radius here is the *full* minimum dimension rather than half of it, so the object occupies
roughly the inner two-thirds of the disk instead of filling it. If that convention is wrong, these are
correct Zernike moments of the wrong disk, and this report would not have caught it.

That is why the claim is scoped by a recipe rather than stated flat (SPEC §1): the registry row reads
`config_recipe=zernike.shape2d_native`, and the recipe *is* the geometry. Deciding whether the
convention itself should change is a separate question from whether the moments are computed
correctly, and only the second one is answered here.

## 4. Result

Old code against the closed form, over the 29 non-zero magnitudes:

| | relative error | at |
|---|---:|---|
| smallest | 0.052% | (n=5, m=1) |
| median | 48.4% | |
| largest | **346.7%** | (n=3, m=1) |

`A(1,1)` is excluded because its true value is zero: old code read 0.0358 where the answer is 0, and
no relative error is definable. `tests/vetting/oracles/gen_zernike_analytic.py` reproduces the
comparison; the per-index table is what it prints.

New code:

| | worst relative difference |
|---|---:|
| Nyxus vs the closed form, 29 non-zero magnitudes | **3.4e-15** at (n=8, m=8) |
| Nyxus vs the closed form, `A(1,1)` (a mathematical zero) | 3.3e-18 **absolute** |
| the closed form vs centrosome's own polynomials | **4.9e-15** |
| a second Python build vs the first, closed form against itself | 1.9e-15 |

The last row is why the pinned analytic table is checked against a band rather than bit-exactly: it
is one evaluation of a float computation, and a different interpreter sums the same terms to a
different last bit. The generator's first version demanded bit-equality and failed on a second
Python build with no defect present.

**Band: `rel=1e-12` with an absolute floor of `1e-15`.** Three orders above the measured 3.4e-15,
because the residual is float accumulation across two different summation orders rather than a method
difference, and because the ASan run compiles the same source with gcc. The floor exists for exactly
one entry — `A(1,1)`, the mathematical zero, where a purely relative band would compare two values at
the noise floor. `1e-15` sits far below the smallest real magnitude in the table, `1.0e-4`.

**centrosome is a corroborating cross-check, not the pinned reference.** `MIGRATION.md` §9.1 records
that `mahotas`, `DIPlib` and `Centrosome` are **not** accepted SPEC §4 oracle tokens. The pinned table
is therefore the closed form, computed by this repo's own generator with no third-party import; the
centrosome run is reported in the generator's output and here, and pins nothing. That keeps the
`oracle=analytic` claim honest under the rule the migration plan set.

## 5. CellProfiler was run, and is not comparable

`MeasureObjectIntensityDistribution` reports Zernike magnitudes under the same `(n, m)` indexes and
the same default degree 9, so it is the obvious candidate. It measures a different quantity:

| | CellProfiler | Nyxus |
|---|---|---|
| disk centre | centre of the object's **minimum enclosing circle** | the **intensity centroid** |
| disk radius | radius of that circle | `min(bbox width, bbox height)` |
| weight | the raw pixel value | `I / sum(I)` |
| normalisation | divide by the **pixel count** | multiply by `(n+1)/π` |

All 30 values disagree by more than 1%. That is a convention gap, not a disagreement about the
moments — the same code path in centrosome that CellProfiler calls agrees with Nyxus to 4.9e-15 once
both are evaluated on the same disk. The generator prints the comparison under `--cellprofiler` so
the candidate oracle is closed with numbers rather than an opinion.

There is no other candidate. `TOOLS.md` lists `wndcharm` for Zernike, which is the code's own
ancestry ("Wind-Charm's adaptation of Ilya Goldberg's adaptation of Michael Boland's mb_Znl.c") and a
Python 2.7 source build; matching it would in any case prove nothing, since **whether wndcharm carries
the same seeding has not been checked** — if it does, it is wrong too.

## 6. Verification

Negative control, and the reason to trust the two invariants: reverting the seeding to `COST[0] = x/r`
and rebuilding fails **four** of the eight cases — the analytic oracle, the regression snapshot, and
both identity invariants — while the bound invariant, the index-count invariant and both mechanics
tests still pass, which is correct because none of them constrains the angular term. The `A(0,0)`
failure reports a difference of 0.29781250022683375 from `1/π`; the `A(1,1)` failure reports 0.035831
against a bound of 1e-14.

Those two invariants are the whole lesson of this family: they are three lines each, they need no
oracle, and either one would have caught this at any point in the feature's life.
