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
| [`benchmarks.md`](benchmarks.md) | The fixtures assertions run on (SPEC §6.3): what each one is, why it exists, and which recipes and tests use it. A `benchmark` id in the registry must be defined here, which `check_coverage.py` enforces — as it does for `test_name`, which must resolve to a gtest case in `test_all.cc`. |
| [`matrix/`](matrix/) | Per-family config matrices (SPEC §5.1): the settings a family actually reads, the config points they produce, and each point's verdict. |
| [`report_features.py`](report_features.py) | Generates [`features.csv`](features.csv) and [`features.md`](features.md) — the whole report, one row per (dim, feature, oracle, config_recipe), joining what the registry CLAIMS to what the tree ASSERTS. Its `verdict` column is the join and `verdict_scope` says how far that join reached — `row+config` compared the row's own assertion at its own config, `row` its own assertion, `feature` only the feature, which is all a row with no `test_name` can be asked. `--check` fails in CI on any row that disagrees without an allowlisted reason. **This is the artifact to read.** |
| [`features.md`](features.md) | The generated report: coverage, verdicts, the families no scanner covers, the family × oracle matrix, the dimensionality split. Never edit by hand. |
| [`audit/`](audit/) | Baseline coverage audit of the current test tree (see below), and the per-family coverage scanners that keep it current. |
| [`audit/scanlib.py`](audit/scanlib.py) | The scanners' shared machinery: the coverage rule, the acceptance checks and the artifact rendering, in one place. Each `audit/scan_*_coverage.py` is the family's declaration on top of it — which files to read, how the family spells its feature names in C++, which checks apply. |
| [`audit/scanlib_selftest.py`](audit/scanlib_selftest.py) | Negative controls for the coverage rule itself, on fixtures rather than on the tree: a literal the case never loops, a returned value the caller discards, a gtest case registered on one line, and a registry row whose `config_recipe` has been swapped for another of its family's. The per-family `--check` cannot catch these — it compares an artifact to the tree through the same rule that produced it. Runs in CI. |
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

### The per-family scanners

`audit/scan_<family>_coverage.py` regenerates `audit/<family>_<dim>_coverage.csv` by reading the test
sources, so a coverage artifact cannot drift from the tree. `--check` reports drift instead of
rewriting and runs that family's acceptance checks against `oracle_coverage.csv`. All twenty share
`scanlib.py`; a family whose tests genuinely read differently (2D moments resolves golden tables, not
assertion lines) or which judges differently (2D NGTDM, IMQ, 3D GLDM and 3D GLSZM check each row
against the tests of its own KIND) overrides that part in its own file, where the difference is
visible.

All twenty run in CI through [`audit/run_scanners.py`](audit/run_scanners.py), which runs
every family before failing and names the ones that failed, beside `check_coverage.py`,
`check_test_names.py`, `scanlib_selftest.py` and `report_features.py --check`.

Regenerable byproducts of the retired pipeline, all git-ignored and all under `audit/`:
`audit/features.csv`, `audit_scan.txt`, `vetting_pivot.csv`, `gap_not_tested.csv`,
`gap_claimed_3p.csv`, `gap_regression.csv`. `audit/features.csv` is not the tracked
`tests/vetting/features.csv` the table above links; the two share a basename and nothing else. Run
`extract_features.py && scan_tests.py && merge.py && report.py` from this folder to rebuild them
(and to refresh `vetting_report.csv`).

## `oracles/` — golden generators

- `fraclac/shiftgrid_boxcount.ijm` — headless ImageJ macro reproducing FracLac's shifting-grid
  box-counting (the FracLac plugin itself is GUI-only). `ref_boxcount.py` is a numpy cross-check.
  See TOOLS.md → `fraclac` for the reconciliation (from-method reimplementation vs. true-tool run).

## Status

Documentation + data only — nothing here changes the test tree. Rollout (registry CSV, config
recipes, per-family config matrices, per-tool generators, and the test renames in SPEC.md §6) begins
after the spec is merged.
