# 3D GLDZM vs the MIRP oracle — vetting report

The family computes the IBSI GLDZM and is vetted against MIRP 2.6.0 on the compatibility phantom, to
a worst **absolute residual of 5.7e-15** over the sixteen features MIRP computes. Sixteen of the
eighteen registry rows are `status=vetted` at that recipe. The two that are not — `3GLDZM_GLM` and
`3GLDZM_ZDM` — have no counterpart in MIRP or in IBSI at all.

Getting there took four defects out of `src/nyx/features/3d_gldzm.cpp`. All eighteen values move.

## Reproduction

| | |
|---|---|
| generator | `tests/vetting/oracles/gen_gldzm3d_mirp.py` |
| oracle | MIRP 2.6.0 (numpy 2.4.6, Python 3.11), env `nyxus_mirp` |
| fixtures | `bench_compat_gldzm_3d` (the vetted cell) and `bench_ut57_3d` (the drift guard's) |

```
python tests/vetting/oracles/gen_gldzm3d_mirp.py
```

Three MIRP runs, and the generator verifies every number the tree quotes from any of them: the
sixteen pins in `test_3d_gldzm_mirp.h` at `rel<=1e-12`, and the two tables below at `rel<=1e-5`, the
precision they are quoted to. It also rebuilds the compatibility phantom from the rule that defines
it and fails if the checked-in NIfTI pair no longer holds what that rule produces.

MIRP implements IBSI GLDZM. PyRadiomics has no GLDZM at all, so MIRP is the only mainstream oracle
available for this family.

## Result at `gldzm3d.mirp_compat_phantom` -- the vetted cell

Both tools read the phantom's voxel values as grey levels 1..8 directly: MIRP with
`base_discretisation_method="none"`, Nyxus at `IBSI=true`, which switches the family's binning off.
Nothing about the comparison depends on the two agreeing about discretisation, because neither one
discretises.

| feature | Nyxus | MIRP | rel |
|---|---:|---:|---:|
| `3GLDZM_SDE` | 0.8238705738705739 | 0.8238705738705739 | 0.0e+00 |
| `3GLDZM_LDE` | 2.4615384615384617 | 2.4615384615384617 | 0.0e+00 |
| `3GLDZM_LGLZE` | 0.13946884110787175 | 0.13946884110787175 | 0.0e+00 |
| `3GLDZM_HGLZE` | 27.53846153846154 | 27.53846153846154 | 0.0e+00 |
| `3GLDZM_SDLGLE` | 0.12001901217090417 | 0.12001901217090417 | 0.0e+00 |
| `3GLDZM_SDHGLE` | 23.23946886446887 | 23.23946886446887 | 0.0e+00 |
| `3GLDZM_LDLGLE` | 0.3112708535795271 | 0.31127085357952705 | 1.8e-16 |
| `3GLDZM_LDHGLE` | 60.19230769230769 | 60.19230769230769 | 0.0e+00 |
| `3GLDZM_GLNU` | 23.384615384615383 | 23.384615384615383 | 0.0e+00 |
| `3GLDZM_GLNUN` | 0.12848689771766694 | 0.12848689771766694 | 0.0e+00 |
| `3GLDZM_ZDNU` | 121.0 | 121.0 | 0.0e+00 |
| `3GLDZM_ZDNUN` | 0.6648351648351648 | 0.6648351648351648 | 0.0e+00 |
| `3GLDZM_ZP` | 0.12037037037037036 | 0.12037037037037036 | 0.0e+00 |
| `3GLDZM_GLV` | 4.792899408284024 | 4.792899408284024 | 0.0e+00 |
| `3GLDZM_ZDV` | 0.5746890472165199 | 0.5746890472165197 | 3.9e-16 |
| `3GLDZM_ZDE` | 3.7485223944249118 | 3.7485223944249175 | 1.5e-15 |

**Thirteen of the sixteen are bit-identical to MIRP's doubles.** The three that are not differ in
the last unit in the last place, worst 5.7e-15 absolute on `3GLDZM_ZDE` — the only one of the
sixteen whose formula takes a logarithm, which Nyxus evaluates as `log2(p + EPS)`. The assertions
band this at SPEC §7's exact tier, an absolute 1e-9.

**The phantom's zone count follows from how it is built**, so `3GLDZM_ZP` can be checked without any
tool: 189 bricks of one grey level each, less the seven that the corner-touching implant joins into
one zone, is 182 zones over 1512 ROI voxels — 0.12037037037037036, which is the pinned value.

## The phantom, and what each of its three choices separates

`bench_compat_gldzm_3d` is a 12x12x12 cube of 2x2x2 bricks with a 6x6x6 corner cut out of it, inside
a two-voxel background margin in a 16x16x16 volume. A brick's grey level is
`1 + 4*(bz%2) + 2*(by%2) + (bx%2)` over its brick coordinates, except the brick at `(3,3,3)`, which
carries level 1 instead of the 8 that rule gives it. Nothing about it is arbitrary — each choice
separates one of the defects below, and each was negative-controlled by putting the defect back:

| choice | what it separates | control: put the defect back |
|---|---|---|
| the background margin | the mask, not the binned intensity, is what says which voxels are the ROI's | on `bench_ut57_3d`, growing zones over the bounding box gives 127,933 zones against 40,769 |
| the corner cut | the ROI is non-convex, so near the cut the shortest way out is diagonal and a distance measured along the axes overstates it | distance = the shortest of the six axis rays: **9 of the 16 fail**, worst `LDHGLE` 77.12 against 60.19 |
| the `(3,3,3)` implant | that brick touches eight level-1 bricks at a corner and nowhere else | 6-connected zones: **all 16 fail**, `ZP` 0.125 against 0.12037. 18-connected: **all 16 fail**, same 0.125 — a corner is the only contact, so 18-connectivity does not see it either |

The controls were run by patching `3d_gldzm.cpp` and rebuilding, not by reasoning about the
fixture. Under the ray-distance control the seven purely grey-level features still pass, which is
the expected shape: that control moves distances and leaves the zone map alone.

## The four defects

Every one of them is in `prepare_GLDZM_matrix_kit` or in what it called.

**1. Zones were grown over the bounding box, and the background joined them.** The zone search
seeded at every voxel of the box and skipped intensity 0 only when `IBSI=true`. In the default
configuration the family bins MATLAB-style, and **that binning sends intensity 0 to level 1** — so
after binning the background filling the rest of the bounding box was indistinguishable from a
genuine level-1 ROI voxel, and there was no zero left to skip. On `bench_ut57_3d` the box is 551,040
voxels against the ROI's 274,432.

The fix marks the ROI from `r.raw_pixels_3D`, which is the only thing that knows. Masking by
"intensity is zero" instead would allocate nothing, but it is exactly what was wrong. The 2D NGLDM
implementation `ngldm.cpp` builds its mask from the pixel cloud for the same reason. **2D GLDZM does
not** — `gldzm.cpp` still reads its grey levels off the whole bounding box and still tests
`intensity == 0` in its own `dist2border`, so it carries defects 1 and 2; see "What is left open".

### What the family allocates, and what the admission gate actually covers

`LR::get_ram_footprint_estimate_3D` budgets `sizeof(Pixel2)` per bounding-box voxel — 12 bytes where
`StatsInt` is 4 — and is the only gate deciding whether a ROI is processed in RAM. Against that
budget, per bounding-box voxel:

| buffer | bytes | notes |
|---|---:|---|
| `aux_image_cube` | 4 | the ROI's cache, allocated before this family runs |
| the binned cube `D` | 4 | |
| the distance cube | 4 | **also the ROI mask** — nonzero exactly at the ROI's voxels before the transform runs, and zeroed again as each zone consumes one — so nothing is allocated to hold a mask separately |

Twelve bytes of cubes against a twelve-byte budget, and then three buffers that scale with the ROI
rather than with the box and are **not** in that budget:

| buffer | worst case | |
|---|---|---|
| `frontier` + `next_frontier` | 4 bytes per ROI voxel | a voxel enters exactly one of the two over the whole walk, so together they never hold more than the ROI. They hold packed indices rather than `{x,y,z}` triples, which is what takes this from 12 bytes to 4 |
| the zone stack | 4 bytes per voxel of the largest zone | packed the same way, and cleared rather than freed between zones so the allocation is made once |
| `Z`, the zone list | 12 bytes per zone | predates this branch |

**So the estimate does not bound this family's peak, and it never did.** `Z` and the zone-growing
stack are in the pre-existing implementation and in the 2D twin, and the same is true of every
texture family here: the estimate models the cubes a ROI needs, not the per-feature scratch on top of
them. What this branch changed is the size of the scratch — the two index vectors are a third of what
`{x,y,z}` triples would cost — not whether it is accounted for.

Accounting for per-feature scratch in the admission gate is a repo-wide change: the estimate is
per-ROI, every family would have to declare its own worst case, and the trivial/out-of-core decision
would move for all of them. Deliberately not taken here, and filed instead.

**2. `dist2border` measured the distance to the bounding box, not to the ROI.** It scanned rays until
it hit a voxel of intensity 0 *or the margin of the box*. Since after MATLAB binning no voxel is 0,
**no ray ever stopped at the ROI**, and the function reduced to `min(x+1, W-x, y+1, H-y)` — a
property of the box's geometry with the ROI nowhere in it. That is the whole of `LDE` 314 against
11.2 and `ZDV` 79.7 against 3.25.

**3. `dist2border` never scanned the z axis.** Left, right, up and down, all within one z-slice, and
the minimum of four. The 2D implementation scans the same four, which in 2D is both axes and
therefore complete; the 3D version was that function with a `z` threaded through and never scanned.

Defects 2 and 3 are both gone because the ray scan is gone. **The distance is now the city-block
distance transform**, breadth-first from the ROI's surface inwards, which is the distance IBSI
defines and the one MIRP computes. Six rays would have fixed defect 3 and left an approximation:
the shortest of the six rays is an upper bound on the transform, it differs from it on 40 of the
phantom's 1512 voxels, and that is enough to move `LDE` by 15% — far outside any band this family
could be vetted at. On `bench_ut57_3d` the two metrics give mean distances of 5.5418 and 5.5198.

**4. Zones were 6-connected.** IBSI defines a GLDZM zone at 26-connectivity in 3D, which is the
connectivity MIRP implements and the one Nyxus' own 3D GLSZM already used for the same notion of a
zone. The search now walks all 26 offsets.

The zone search itself was rewritten around the mask: an explicit stack of voxels rather than the
parent-stack walk, taking a voxel into a zone by clearing its mask bit, and reading the zone's
distance out of the precomputed transform. The old walk also read `dist2border` off the cube it was
destructively marking as it went, which the precomputed transform ends as a class of bug.

## What the values do

All eighteen move. On `bench_ut57_3d`:

| feature | before | after |
|---|---:|---:|
| `3GLDZM_LDHGLE` | 734618.35720259824 | 14203.744217420099 |
| `3GLDZM_LDE` | 314.01248309662088 | 7.4337854742574017 |
| `3GLDZM_ZDV` | 79.723412707174901 | 2.0941027837664841 |
| `3GLDZM_ZP` | 0.46617376982276121 | 0.14855774836753732 |
| `3GLDZM_SDE` | 0.022387420258025731 | 0.47006537949579702 |

Zones on that phantom go from **127,933 to 40,769**, and `ZP` from 0.466 to 0.1486.

## The drift guard's configuration cannot be compared with MIRP, and why

`gldzm3d.regression_ut_phantom` is `GREYDEPTH=64`, `IBSI=false` on the segmented phantom — the
family's default mode. Running MIRP on the same fixture at `fixed_bin_number` n=64 does **not**
measure the GLDZM, because the two tools do not put this ROI on the same grey levels:

| | levels the ROI reaches |
|---|---|
| MIRP, `fixed_bin_number` n=64 | 1-64 |
| Nyxus, `GREYDEPTH=64` | 22-64, 43 distinct |

Nyxus bins `floor(64 * i / ROI max) + 1` over a volume the loader has already shifted by its minimum
(−1024 on this CT phantom), so the ROI occupies the upper two thirds of the shifted range. Two
independent causes: the lower bin edge is 0 rather than the ROI minimum, and the volume-wide shift.
Neither is a GLDZM question. It is the same gap every Nyxus texture family meets on a CT-like
fixture, and it is why the vetted cell uses a phantom on which neither tool discretises at all.

## Result at `gldzm3d.mirp_fbn64` -- MIRP discretises

| feature | Nyxus | MIRP | Nyxus / MIRP |
|---|---:|---:|---:|
| `3GLDZM_SDE` | 0.470065379495797 | 0.3819882926480209 | 1.23 |
| `3GLDZM_LDE` | 7.433785474257402 | 11.231508735923956 | 0.662 |
| `3GLDZM_LGLZE` | 0.0004349083674222441 | 0.0014952243584138115 | 0.291 |
| `3GLDZM_HGLZE` | 2685.0693909588167 | 1920.8249709702682 | 1.4 |
| `3GLDZM_SDLGLE` | 0.00015407786939386234 | 0.0002162322321202084 | 0.713 |
| `3GLDZM_SDHGLE` | 1540.784139550179 | 1098.1125072487994 | 1.4 |
| `3GLDZM_LDLGLE` | 0.004580296296000341 | 0.03479098207358527 | 0.132 |
| `3GLDZM_LDHGLE` | 14203.744217420099 | 10881.709594663931 | 1.31 |
| `3GLDZM_GLNU` | 1349.3969192278446 | 1433.445626637142 | 0.941 |
| `3GLDZM_GLNUN` | 0.03309860235050761 | 0.01935467077093708 | 1.71 |
| `3GLDZM_ZDNU` | 10424.1403272094 | 14488.2767411088 | 0.719 |
| `3GLDZM_ZDNUN` | 0.25568790814612574 | 0.19562362265546165 | 1.31 |
| `3GLDZM_ZP` | 0.14855774836753732 | 0.2698737756529851 | 0.55 |
| `3GLDZM_GLV` | 84.66230769118728 | 227.68948600566614 | 0.372 |
| `3GLDZM_ZDV` | 2.094102783766484 | 3.2456304818289197 | 0.645 |
| `3GLDZM_ZDE` | 6.469785865699117 | 7.504620664827702 | 0.862 |

The residual has the shape a discretisation gap has: the grey-level-weighted features are the ones
far from 1 (`LDLGLE` 0.132x, `LGLZE` 0.291x, `GLV` 0.372x) and `GLNU`, which sums a marginal over
grey levels but weights nothing by them, is 0.941x. **`INVALID` as a vetting cell**, and
`matrix/gldzm3d.md` records it as such with this measurement beside it.

## Result at `gldzm3d.mirp_samelevels` -- the same grey levels

The same fixture with MIRP handed the grey levels Nyxus bins to, and told
`base_discretisation_method="none"`. Now the GLDZM itself is what is compared, on a 274,432-voxel
ROI rather than the compatibility phantom's 1512:

| feature | Nyxus | MIRP | rel |
|---|---:|---:|---:|
| `3GLDZM_SDE` | 0.470065379495797 | 0.4700653794957969 | 2.4e-16 |
| `3GLDZM_LDE` | 7.433785474257402 | 7.433785474257402 | 0.0e+00 |
| `3GLDZM_LGLZE` | 0.0004349083674222441 | 0.0004349083674222441 | 0.0e+00 |
| `3GLDZM_HGLZE` | 2685.0693909588167 | 2685.0693909588167 | 0.0e+00 |
| `3GLDZM_SDLGLE` | 0.00015407786939386234 | 0.00015407786939386232 | 1.8e-16 |
| `3GLDZM_SDHGLE` | 1540.784139550179 | 1540.7841395501787 | 1.5e-16 |
| `3GLDZM_LDLGLE` | 0.004580296296000341 | 0.004580296296000339 | 3.8e-16 |
| `3GLDZM_LDHGLE` | 14203.744217420099 | 14203.744217420099 | 0.0e+00 |
| `3GLDZM_GLNU` | 1349.3969192278446 | 1349.3969192278446 | 0.0e+00 |
| `3GLDZM_GLNUN` | 0.03309860235050761 | 0.03309860235050761 | 0.0e+00 |
| `3GLDZM_ZDNU` | 10424.1403272094 | 10424.1403272094 | 0.0e+00 |
| `3GLDZM_ZDNUN` | 0.25568790814612574 | 0.25568790814612574 | 0.0e+00 |
| `3GLDZM_ZP` | 0.14855774836753732 | 0.14855774836753732 | 0.0e+00 |
| `3GLDZM_GLV` | 84.66230769118728 | 84.66230769118725 | 3.4e-16 |
| `3GLDZM_ZDV` | 2.094102783766484 | 2.094102783766484 | 0.0e+00 |
| `3GLDZM_ZDE` | 6.469785865699117 | 6.469785865699184 | 1.0e-14 |

Worst **1.0e-14**, on `3GLDZM_ZDE` again. This is not a pinned cell — it takes a MIRP invocation the
Nyxus command line cannot produce, so it is a measurement rather than a configuration a user can
reach — but it is what says the agreement on the compatibility phantom is not an artifact of that
phantom's size.

## What is left open

- **The discretisation gap** is not this family's to close. It is the same
  `floor(n * i / ROI max) + 1` over a zero lower edge that every Nyxus texture family bins with, so
  any family whose oracle recipe is `fixed_bin_number` on a CT-like fixture is measuring it rather
  than itself.
- **`3GLDZM_GLM` and `3GLDZM_ZDM`** have no counterpart in any tool — MIRP's GLDZM emits no
  `dzm_gl_mean` / `dzm_zd_mean` column and IBSI defines neither — so they stay drift guards at every
  recipe.
- **The admission gate does not model per-feature scratch**, for this family or any other. Sized
  above; filed as its own item, since fixing it means every family declaring a worst case and the
  trivial/out-of-core decision moving repo-wide.
- **The 2D twin carries defects 1 and 2**, and is vetted anyway because its recipe hides them.
  `gldzm.cpp` has the same "intensity is zero means outside the ROI" test in its `dist2border` and
  the same zero-skip only under `IBSI=true`; `gldzm.ibsi_phantom_2d` runs at `IBSI=true`, where the
  binning is off, background stays 0 and both readings are correct. At a positive `GREYDEPTH` the 2D
  family has the same two defects the 3D one had, and no cell of `matrix/gldzm.md` is vetted there.
  Recorded rather than fixed here: it moves 2D goldens this branch does not own.
