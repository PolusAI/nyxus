# Regenerating the 2D radial goldens

`FRAC_AT_D`, `MEAN_FRAC`, `RADIAL_CV` — 8 radial bins each, 24 pinned values in
`tests/test_2d_radial_regression.h`.

The family is `status=regression`: the goldens are Nyxus' own output, so "regenerate" means re-run
Nyxus and re-pin, not re-run a tool. The CellProfiler side of this file is here because the registry
names CellProfiler as the candidate oracle and someone will try it again; running it is how you
confirm the divergence is still what `radial_2d_cellprofiler_vetting_report.md` says it is.

---

## 1. Re-pinning the regression goldens

The values come from one gtest fixture, so there is no separate generator to run.

```
cmake -S . -B build-test -DRUN_GTEST=ON -DBUILD_CLI=OFF -DBUILD_LIB=OFF
cmake --build build-test --target runAllTests
build-test/tests/runAllTests --gtest_filter=*RADIAL*
```

On Windows the binary needs the dependency DLLs from the build environment, so put that
environment's `Library\bin` on `PATH` first or it exits 53 with no message.

To read the values out rather than compare them, the shortest route is the model in the generator —
it reproduces all 24 bit-exactly and needs no build:

```
python tests/vetting/oracles/gen_radial_cellprofiler.py --skip-cellprofiler
```

That path is stdlib-only — no numpy, no CellProfiler env. As well as reproducing the 24 values it
runs the range and identity checks over the pinned literals themselves: every `FRAC_AT_D` entry a
fraction of a whole pixel count summing to one, every `MEAN_FRAC` entry inside the ROI's intensity
range on a non-empty bin, `RADIAL_CV` within `sqrt(num_bins - 1)`, the empty bins consistent across
all three tables, and the two intensity tables reconstructing the ROI's total intensity. The C++
invariant tests assert the same properties of the *computed* values; this asserts them of what the
header says, so an edit to the table is caught without a rebuild.

Pin at `%.17g`. A value truncated shorter eats most of the `rel=1e-9` band before the test starts.

**Three inputs decide every one of the 24 numbers**, and all three are pinned in
`tests/test_2d_radial_mechanics.h`: the ROI (26 pixels of `shape2d_morphology_mask`), the centre
pixel `(3,4)`, and the squared normalising radius `10`. If a source change moves any of them, all 24
goldens move together and the mechanics pins are what tells you which of the three it was.

## 2. Running CellProfiler on the same fixture

```
conda create -n nyxus_cellprofiler -c conda-forge python=3.9 cellprofiler=4.2.8
conda run -n nyxus_cellprofiler python tests/vetting/oracles/gen_radial_cellprofiler.py
```

Two things will stop you before the module imports:

- **`JDK_HOME` / `JAVA_HOME` must point at `<env>\Library\lib\jvm`**, or the process dies with a bare
  exit 127 and no traceback.
- **On Windows the working directory must be on the same drive as the installed package.**
  `cellprofiler.modules.measureobjectintensitydistribution` imports `cellprofiler.gui.help.content`,
  which calls `os.path.relpath` between the package directory and the cwd; across drives that raises
  `ValueError: path is on mount 'C:', start on mount 'D:'` at import time. `cd` to any directory on
  the package's drive first — the script takes absolute paths and does not care where it runs from.

The generator exits non-zero if a pin stops reproducing, if the header pins a feature the model does
not produce (or the reverse), if the independent numpy rebuild stops reproducing the CellProfiler
module, **or if CellProfiler starts agreeing with Nyxus** — that last one means the divergence record
is stale and the three rows became promotable, which is a result, not a pass.

## 3. Mapping CellProfiler's names to Nyxus'

| Nyxus | CellProfiler measurement |
|---|---|
| `FRAC_AT_D[i]` | `RadialDistribution_FracAtD_<image>_<i+1>of8` |
| `MEAN_FRAC[i]` | `RadialDistribution_MeanFrac_<image>_<i+1>of8` |
| `RADIAL_CV[i]` | `RadialDistribution_RadialCV_<image>_<i+1>of8` |

CellProfiler bins are 1-based in the measurement name; Nyxus' vectors are 0-based.

## 4. Convention differences to account for — all six of them

These are not tolerances. They are different quantities, and the table in
`radial_2d_cellprofiler_vetting_report.md` §3 is what they produce. Do not try to close any of them
with a band.

1. **Centre.** CellProfiler takes the pixel of maximum distance-to-edge. Nyxus takes the pixel
   minimising (max − min) squared distance to the contour, through an approximate search that
   returns `(3,4)` where an exact scan of the same criterion returns `(4,4)`.
2. **Radial coordinate.** CellProfiler normalises per pixel by `d_centre / (d_centre + d_edge)`.
   Nyxus divides by one global maximum radius, obtained from an approximate search that returns
   `sqrt(10)` where the true maximum over the same contour is `sqrt(13)`.
3. **Bin width.** CellProfiler scales by `bin_count`; Nyxus scales by `bin_count - 1`, giving 7 rings
   plus a last bin only `r >= r_max` reaches.
4. **`FracAtD`.** CellProfiler's is a fraction of intensity; Nyxus' is a fraction of pixel count and
   never reads the image.
5. **`MeanFrac`.** CellProfiler's is normalised by the ROI's mean intensity and lands near 1; Nyxus'
   is the bin's mean intensity in absolute units.
6. **`RadialCV`.** CellProfiler takes the CV of the 8 wedge *means* over the non-empty wedges; Nyxus
   takes the CV of the 8 wedge *sums* over all 8.

## 5. If the family is ever to be vetted

Three things have to be true at once, and none of them is true today:

- the six divergences above are resolved in `radial_distribution.cpp`, on top of the contour
  off-by-one in `contour.cpp` that shifts the contour these features measure against
  (`radial_2d_cellprofiler_vetting_report.md` §6 defects 1-6);
- the fixture has a **unique** distance-to-edge maximum. `shape2d_morphology_mask` has an 8-way tie,
  so CellProfiler's own centre moves with the label image's padding;
- the fixture has enough pixels that 8 radial bins are not mostly empty. 26 pixels leaves 3 of the 8
  bins empty on one side or the other, which cannot distinguish a binning rule from a coincidence.

A second, larger, tie-free ROI added to `test_data.h` would satisfy the last two without touching the
existing goldens, and would be the natural first commit of the fix branch.
