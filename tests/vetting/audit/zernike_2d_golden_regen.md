# Regenerating the 2D Zernike goldens

`ZERNIKE2D` — 30 magnitudes, pinned twice: the closed-form reference in
`tests/test_2d_zernike_analytic.h` and Nyxus' own output in `tests/test_2d_zernike_regression.h`.

---

## 1. The analytic table — the oracle

Stdlib only. No build, no oracle environment:

```
python tests/vetting/oracles/gen_zernike_analytic.py
```

It parses the fixture out of `test_data.h` and the geometry out of `test_2d_zernike_mechanics.h`,
sums the factorial series for `R_nm`, and then does four things:

1. checks the two magnitudes the closed form forces — `A(0,0) = 1/π` and `A(1,1) = 0`;
2. checks the analytic header's 30 pins against a fresh evaluation, at the band the header itself
   asserts (`rel=1e-12` plus a `1e-15` floor). **Not bit-exact on purpose** — a different interpreter
   sums the same terms to a different last bit, and demanding equality made the generator fail on a
   second Python build with no defect present;
3. checks Nyxus' pinned output against the closed form, which is the vetting claim;
4. checks the reverse direction — an index the closed form produces that neither header pins.

Re-pin at `%.17g`.

## 2. The regression table — Nyxus' own output

```
cmake -S . -B build-test -DRUN_GTEST=ON -DBUILD_CLI=OFF -DBUILD_LIB=OFF
cmake --build build-test --target runAllTests
build-test/tests/runAllTests --gtest_filter=*ZERNIKE*
```

On Windows the binary needs the dependency DLLs from the build environment on `PATH`, or it exits 53
with no message.

To read the values out rather than compare them, add a temporary dump to
`test_2d_zernike_moments_regression()` — `std::cout << std::setprecision(17)` over the vector. Use
`std::cout`, not `printf`: an escaped `\n` inside a shell heredoc is an easy way to write a literal
newline into a C++ string literal and get `C2001: newline in constant`.

## 3. The two cross-checks

Both need the CellProfiler environment (`tests/vetting/TOOLS.md` has the recipe and the three traps
that stop the run before any module executes):

```
conda run -n nyxus_cellprofiler python tests/vetting/oracles/gen_zernike_analytic.py --centrosome --cellprofiler
```

- `--centrosome` evaluates `centrosome.zernike`'s own polynomials at the same geometry — a third
  independent implementation, agreeing to 4.9e-15. It fails the run if that ever exceeds 1e-12.
- `--cellprofiler` runs `MeasureObjectIntensityDistribution` and prints the divergence. Its numbers
  are **not** comparable and are recorded, not asserted; see §4.

## 4. Convention differences to account for

`ZernikeFeature`'s disk is a convention, not a fact about Zernike moments, and it is pinned in
`test_2d_zernike_mechanics.h`:

| | Nyxus | CellProfiler |
|---|---|---|
| centre | the ROI's intensity centroid | centre of the minimum enclosing circle |
| radius | `min(bbox width, bbox height)` in pixels | radius of that circle |
| weight | `I / sum(I)` | the raw pixel value |
| scale | `(n+1)/π` | divide by the pixel count |

On this fixture that is a 6×7 bounding box, radius 6, centroid (3.8416, 4.4389) in 1-based pixel
coordinates. All 42 bounding-box pixels fall inside the disk, which is what makes `A(0,0)` exactly
`1/π`; if a future fixture changes that, the identity stops holding and the mechanics test says so
first.

Note the radius is the **full** minimum dimension, not half of it, so the object occupies roughly the
inner two-thirds of the disk rather than filling it. That is what the code does; it is recorded here
because it is the first thing that looks like a bug to a reader who knows the standard definition,
and it is not one — it is a scale convention that cancels out of nothing but is applied consistently.

## 5. Index order

30 magnitudes, one per `(n, m)` with `n <= 9`, `m >= 0` and `n - m` even, emitted n-ascending then
m-ascending: (0,0), (1,1), (2,0), (2,2), (3,1), (3,3), (4,0) … (9,9). `centrosome`'s
`get_zernike_indexes(10)` produces the same order, which the generator asserts rather than assumes.
