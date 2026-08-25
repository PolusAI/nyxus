# 2D neighbour config matrix

Axes = the settings `NeighborsFeature` actually reads; verdicts are measured on
`bench_scene7_5roi_enclosed` unless a cell says otherwise (SPEC §5.1).

The family has **one** knob. `NeighborsFeature::manual_reduce()` reads `PIXELDISTANCE` and nothing
else — no angle set, no binning, no `IBSI` branch, no grey depth. `PIXELSIZEUM` and `XYRES` are set
by the shared fixture because `BasicMorphologyFeatures` and `ContourFeature` want them, but the
neighbour quantities are computed from raw pixel coordinates: centroids are the plain mean of
`px.x` / `px.y`, and adjacency is a squared distance between contour pixels. So the cross-product
is one axis wide, and the matrix is short for a real reason rather than for want of looking.

| pixel distance | verdict | recipe / oracle |
|---|---|---|
| 1 | VALID | `neighbor.scene2d_radius1` — cellprofiler (2 features) + analytic (6 features) |
| ≥ 2 | NOT MEASURED | reachable and meaningful; see below |
| 0 | INVALID | degenerate — the radius gate is `mind > radius²`, so nothing is anyone's neighbour |

## Why pixel distance 1 is the one measured point

It is the production default and the only distance CellProfiler's `Adjacent` method corresponds to,
which is what makes a CP comparison a same-quantity comparison rather than a near-miss. At that
setting the scene gives label 1 four neighbours and labels 2–5 one each, so both branches of every
closed form are exercised: `CLOSEST_NEIGHBOR2_*` is a real second-neighbour value on label 1 and the
structural zero on the other four, and `ANG_BW_NEIGHBORS_STDDEV` is a sample standard deviation over
four angles on label 1 and the n<2 zero elsewhere.

## PERCENT_TOUCHING is off this axis by construction

It is *physical* 8-adjacency — each contour pixel is marked if it is within squared distance 2 of
another ROI's contour, and that mark is taken **before** the radius gate (`neighbors.cpp`, "PERCENT
TOUCHING is physical contact, independent of the neighbor search radius"). Raising the pixel
distance therefore moves `NUM_NEIGHBORS` and the closest-neighbour quantities and leaves
`PERCENT_TOUCHING` exactly where it was. A matrix that lists it under the distance axis would be
describing a dependence the code does not have.

## Not measured

**Pixel distance ≥ 2.** The graph the oracle recomputes on this scene goes 4/1/1/1/1 neighbours at
distance 1, 4/2/2/2/2 at 2, 4/3/3/3/3 at 3 and complete at 4 — so the axis is live, the scene
discriminates along it, and a second recipe would be cheap. No assertion in the tree runs on it and
this vetting did not measure it: it is recorded here as an open point rather than omitted, so the
next revisit knows it was never triaged rather than assuming it was ruled out. CellProfiler's
`Within` method takes a distance parameter and would be the oracle for it.

**The multi-threaded harvest.** `NeighborsFeature::parallel_process_1_batch()` is a second
implementation of the same collision harvest, and `manual_reduce()` carries an `n_threads` variable
that is assigned 1 and never anything else. Nothing calls the batch version today:
`reduce_trivial_rois.cpp` dispatches every other family through
`runParallel(X::parallel_process_1_batch, …)` and reaches this one only through `manual_reduce`, and
the neighbour overload could not be dispatched that way anyway — it takes five parameters where
`runParallel` passes six. So there is no second config point here. But the two bodies are not
obviously equivalent (the batch one increments only `r1`), and if the parallel path is ever wired up
it needs its own row rather than inheriting this one's evidence.
