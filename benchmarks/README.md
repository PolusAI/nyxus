# Nyxus Benchmarks

This directory contains small, dependency-light microbenchmarks for performance-sensitive feature code.

## Convex Hull

`bench_convex_hull` measures the 2D convex hull path used by `ConvexHullFeature::calculate()`.
It intentionally reports both timing and output shape (`hull_size`, `hull_area`, `solidity`) so performance comparisons can also catch ordering-sensitive correctness regressions.

Build:

```bash
cmake -S . -B build-bench -DRUN_BENCHMARKS=ON -DNOEXTRAS=ON
cmake --build build-bench --target bench_convex_hull -j
```

Run the default workload:

```bash
./build-bench/bench_convex_hull
```

Run a focused workload:

```bash
./build-bench/bench_convex_hull --cycles 5 --case y_major_raster --side 1024
```

Input-order cases:

- `x_major_sorted`: already sorted by the convex hull comparator (`x`, then `y`).
- `y_major_raster`: row-major scan order, matching the usual trivial ROI pixel ingestion path.
- `reversed`: reverse of the sorted comparator order.
- `shuffled`: deterministic pseudorandom order.

For before/after comparisons, build the same target in two clean worktrees and compare median times for the same case/side/cycle settings.
