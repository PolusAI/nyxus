# `tests/vetting/` — Nyxus oracle-vetting framework

The goal: every number Nyxus produces is verified against an independent tool (or a closed-form
value) to close — ideally exact — agreement. This directory holds the **framework** for that;
the actual test migration/rollout happens after the spec is approved and merged.

## Contents

| Path | What |
|------|------|
| [`SPEC.md`](SPEC.md) | The framework spec — assertion-based vetting model, the four test kinds (oracle / regression / invariant / mechanics), the per-assertion coverage registry with the "vetted by ≥1 oracle" metric, config-matching recipes, the config-matrix → valid/invalid triage, naming conventions, tolerance policy, authoring checklist. |
| [`TOOLS.md`](TOOLS.md) | How to stand up each oracle tool locally (Docker or Python venv; MATLAB excluded) — per-tool setup, feasibility, coverage-by-family, rollout order. From one research pass per tool. |
| [`check_test_names.py`](check_test_names.py) | Enforces the SPEC §6.1 file-name and §6.2 function-name conventions over the whole test tree, including gtest case names. `--check` fails in CI. |
| [`report_feature_tests.py`](report_feature_tests.py) | Scans the test tree for the assertion behind every registry feature and regenerates [`feature_test_report.md`](feature_test_report.md) (`--write`; `--csv` adds the full-notes twin, `--check` fails when the report is stale). |
| [`audit/`](audit/) | Baseline coverage audit of the current test tree (see below). Seeds the registry. |
| [`oracles/`](oracles/) | Oracle golden-generators. Currently `fraclac/` (headless shifting-grid box-count macro). Grows per tool during rollout. |

## `audit/` — baseline snapshot

A point-in-time audit of what the test tree vets today, produced by four parallel review agents
plus an authoritative token scan. **Baseline caveat:** it reflects the *merged* tree, so it predates
the GLCM entropy/hom2 normalization fix and the histogram/GLCM oracle tests that are still on their
own branches (PRs). The registry is maintained forward from this baseline.

Tracked artifacts (the rest are regenerable — see below):

- `vetting_report.csv` — master, all 758 real features: `dim, group, family, feature,
  vetting_status, oracle_used, test_files, candidate_oracle, how_to_vet`
- `TEST_VETTING_REPORT.md` — narrative findings (758 features: vetted / claimed-3p / regression /
  not-tested; the `#if 0` "planned for a future PR" 3D-feature correction; includes the group×status
  pivot as a table)
- `extract_features.py` → `scan_tests.py` → `merge.py` → `report.py` (+ `candidate_oracles.py`) —
  the pipeline (repo-relative paths); `audit_1..4.txt` are the raw per-feature agent-evidence inputs.

Regenerable byproducts (git-ignored, rebuilt by the pipeline): `features.csv`, `audit_scan.txt`,
`vetting_pivot.csv`, `gap_not_tested.csv`, `gap_claimed_3p.csv`, `gap_regression.csv`. Run
`extract_features.py && scan_tests.py && merge.py && report.py` from this folder to rebuild them
(and to refresh `vetting_report.csv`).

## `feature_test_report.md` — registry claim vs. tree

`oracle_coverage.csv` records the verdict a feature *carries*; the report records the assertions the
tree *contains*, one row per feature: `Dim, Family, FeatureName, List_of_Oracles, Test_Names,
Regression, Reg_Test_Name, Notes`. The two are not the same and are not meant to be — a row can be
vetted from an offline harness run (`source=tracker`) whose assertion has not been migrated into the
tree yet, and its `target_test` names where that assertion belongs. Only the dim/family/feature
columns come from the registry; everything else is scanned, so the report is what shows the gap.

It also names the assertions that have never executed — a header outside `test_all.cc`'s `#include`
closure, or a `test_*` function no `TEST()` case calls (`not_covered.md` §B.1/§B.2).

Regenerate from the repo root after changing the registry or the tests:

```
python tests/vetting/report_feature_tests.py --write --csv
```

`test_vetting_mechanics.py` fails if the checked-in report is stale.

## `oracles/` — golden generators

- `fraclac/shiftgrid_boxcount.ijm` — headless ImageJ macro reproducing FracLac's shifting-grid
  box-counting (the FracLac plugin itself is GUI-only). `ref_boxcount.py` is a numpy cross-check.
  See TOOLS.md → `fraclac` for the reconciliation (from-method reimplementation vs. true-tool run).

## Status

Documentation + data only — nothing here changes the test tree. Rollout (registry CSV, config
recipes, per-family config matrices, per-tool generators, and the test renames in SPEC.md §6) begins
after the spec is merged.
