# Fractal-dimension oracle validation

nyxus' `FRACT_DIM_BOXCOUNT` (box counting of the filled region) and
`FRACT_DIM_PERIMETER` (Richardson/divider walk on the contour) are validated
two ways:

1. **Analytic ground truth** — shapes whose fractal dimension is known in closed
   form and is convention-independent. Asserted in
   `tests/python/test_fractal_dim_oracle.py`:

   | shape | feature | dimension |
   |-------|---------|-----------|
   | filled square | box-count | 2.000 |
   | straight line | box-count | 1.000 |
   | Sierpinski triangle | box-count | log2(3) = 1.585 |
   | disk | perimeter | 1.000 |
   | Koch snowflake | perimeter | log4/log3 = 1.262 |

2. **Independent software oracle (ImageJ / FracLac)** — box counting cross-check.
   FracLac's distinguishing feature is *shifting grids* (multiple grid origins,
   minimum count per scale), which removes the grid-registration bias. FracLac
   itself is GUI-only, so `shiftgrid_boxcount.ijm` reproduces that method
   headlessly.

## Running the ImageJ box-count oracle

```bash
python tests/oracle/gen_oracle_masks.py          # writes tests/oracle/masks/*.tif
<fiji> --headless -macro tests/oracle/shiftgrid_boxcount.ijm "$(pwd)/tests/oracle/masks/"
```

Expected (shifting-grid box count):

```
square256        D = 2.000
line256          D = 1.000
sierpinski_tri   D = 1.585
sierpinski_carpet D = 1.887
koch_curve       D = 1.266
```

These match the analytic column, confirming the oracle; the Python tests then
confirm nyxus recovers the same values.
