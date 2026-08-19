# 2D NGLDM vs mirp — vetting report

The registry has claimed `oracle=mirp` for all 19 2D NGLDM rows since the tracker was imported, with
`target_test=test_2d_ngldm_mirp.h` — **a file that did not exist**. This report records running that
oracle for the first time.

## Tool and configuration

| | |
|---|---|
| Tool | mirp 2.6.0, numpy 2.4.6, env `nyxus_mirp` (conda) |
| Generator | `tests/vetting/oracles/gen_ngldm_mirp.py` |
| Recipe | `ngldm.ibsi_phantom_2d` |
| Fixture | the four IBSI digital-phantom slices, read out of `tests/test_data.h` by `oracles/ibsi_phantom.py` |
| Nyxus config | `IBSI=true`, `GREYDEPTH=128`, each slice featurised on its own and the four averaged |
| mirp config | `by_slice=True`, `base_discretisation_method="none"`, `ngldm_distance=1`, `ngldm_difference_level=0` |
| Test | `test_2d_ngldm_mirp.h` |
| Tolerance | `rel=1e-9` (SPEC §7 exact tier) |

```
python tests/vetting/oracles/gen_ngldm_mirp.py
```

Verifies every golden pinned in the header and exits non-zero on mismatch. Current run:
**17 verified, 0 failed, 0 unproducible**, every one at `rel = 0`.

## Results — mirp reproduces Nyxus to machine precision

| feature | mirp (= pinned golden) | Nyxus | rel |
|---|---|---|---|
| NGLDM_LDE | 0.15807024738501638 | 0.158070247385016 | 0 |
| NGLDM_HDE | 19.173821809425526 | 19.1738218094255 | 0 |
| NGLDM_LGLCE | 0.7017531915300232 | 0.701753191530023 | 0 |
| NGLDM_HGLCE | 7.486949604403165 | 7.48694960440316 | 0 |
| NGLDM_LDLGLE | 0.047290498640367454 | 0.0472904986403675 | 0 |
| NGLDM_LDHGLE | 3.064914180133555 | 3.06491418013355 | 2.9e-16 |
| NGLDM_HDLGLE | 17.59968920804189 | 17.5996892080419 | 0 |
| NGLDM_HDHGLE | 49.477721878224976 | 49.477721878225 | 0 |
| NGLDM_GLNU | 10.24637942896457 | 10.2463794289646 | 0 |
| NGLDM_GLNUN | 0.5618604963062601 | 0.56186049630626 | 0 |
| NGLDM_DCNU | 3.9646456828345373 | 3.96464568283454 | 0 |
| NGLDM_DCNUN | 0.21177218060411693 | 0.211772180604117 | 0 |
| NGLDM_GLV | 2.7037332451477987 | 2.7037332451478 | 0 |
| NGLDM_DCP | 1.0 | 1 | 0 |
| NGLDM_DCV | 2.729504577399913 | 2.72950457739991 | 0 |
| NGLDM_DCENT | 2.7142924232815497 | 2.71429242328155 | 0 |
| NGLDM_DCENE | 0.17025209750162384 | 0.170252097501624 | 0 |

Worst residual **2.9e-16**. The two implement the same definition over the same neighbourhood, so
this is pinned at the exact tier rather than as a cross-tool band.

## The assertion tolerance was 50%

`assert_ngldm_feature_against_golden_values()` ended in
`agrees_gt(aveTotal, reference, 2.)`. `agrees_gt` computes `tolerance = ground_truth /
frac_tolerance`, so a factor of 2 is a **±50% band** — applied to all 17 IBSI assertions and both
regression assertions.

That is roughly 110× looser than the data requires and it is the "tolerance loose enough to pass a
known-bad value" SPEC §7 calls a test bug: at ±50% these assertions could not have detected a
doubled or halved feature.

Measured agreement against the published IBSI consensus values, same aggregation:

| | worst residual |
|---|---|
| Nyxus vs IBSI consensus | 0.45% |
| mirp vs IBSI consensus | 0.45% |
| Nyxus vs mirp | 2.9e-16 |

The 0.45% is `NGLDM_GLNU`, and it is not disagreement — the IBSI consensus table publishes 10.2
where both tools compute 10.2464. Every other feature lands under 0.2%. That is the precision of the
published value, so the IBSI file now asserts at `rel=1e-2` and the mirp file at `rel=1e-9`: **IBSI
fixes the definition, mirp fixes the digits.** The regression assertions moved to `rel=1e-9` as well
— they pin Nyxus' own output to 17 digits, so a drift guard should catch any change at all.

Tightening 50% → 1% / 1e-9 broke nothing: the suite goes from 785 to 802 tests, all passing.

## NGLDM_GLM and NGLDM_DCM stay regression

mirp exposes no grey-level-mean or dependence-count-mean column, because neither is an IBSI NGLDM
feature — the IBSI table has no counterpart and `test_2d_ngldm_ibsi.h` marks both `--not in IBSI--`.
No oracle can reproduce them, so they keep `status=regression` and their rows stop claiming
`oracle=mirp`.

## The non-IBSI matrix check was never an IBSI test

`test_2d_ngldm_ibsi.h` held two NGLD-matrix checks. The first compares against IBSI's published
Fig. 3.19 matrix and belongs there. The second does not: its ground truth is the NGLDM worked out in
a StackOverflow answer — `test_data.h` names the URL beside the image — and it runs with IBSI mode
**off**, which is the mode the IBSI definition does not describe. It was nonetheless called
`test_2d_ngldm_matrix_correctness_nonibsi_mode_ibsi()`, so both the file and the `_ibsi` suffix
claimed an oracle the assertion does not use.

It moves to `test_2d_ngldm_regression.h` as
`test_2d_ngldm_matrix_correctness_nonibsi_mode_regression()`. Regression rather than an oracle test
because the reference cannot be regenerated from this tree: a forum post is not a tool that can be
run, there is no version to record and no generator to write (SPEC 6.4). It still earns its place —
it holds the non-IBSI matrix, which uses unique grey tones rather than the 0..max IBSI range, to the
shape a second implementation arrived at, so a change in the dependence counting is caught. Turning
it into an oracle claim means running MATLAB or Octave, pinning what that produces, and shipping the
generator beside it.

No values change and no registry row moves: the matrix checks back no feature row, and the suite
stays at 802 tests.

## Registry corrections

Beyond the status and oracle columns, every one of the 19 rows carried:

- `target_test=test_2d_ngldm_mirp.h`, a file that did not exist until this PR — now created, so the
  target is met and cleared;
- `current_test` naming **`test_3d_ngldm_regression.h`**, a 3D file, for 2D features;
- `config_recipe` holding the generic "Not mode-specific…" blurb rather than a recipe id — now
  `ngldm.ibsi_phantom_2d`;
- `source=tracker`, i.e. the verdict rested on an offline harness run nobody could repeat. All 19
  now read `audit`.
