# `tests/vetting/` — Nyxus oracle-vetting framework

The goal: every number Nyxus produces is verified against an independent tool (or a closed-form
value) to close — ideally exact — agreement. This directory holds the framework that governs the
test tree: the spec, the coverage registry, the config recipes and matrices, the oracle golden
generators, and the checkers CI runs to keep the tree and the registry in agreement.

## Contents

| Path | What |
|------|------|
| [`SPEC.md`](SPEC.md) | The framework spec — assertion-based vetting model, the four test kinds (oracle / regression / invariant / mechanics), the per-assertion coverage registry with the "vetted by ≥1 oracle" metric, config-matching recipes, the config-matrix → valid/invalid triage, naming conventions, tolerance policy, authoring checklist, and what CI enforces. Any new or changed test follows it. |
| [`oracle_coverage.csv`](oracle_coverage.csv) | The registry, the single source of truth: one row per assertion (`dim × feature × oracle × config_recipe`) with its `status` (`vetted` / `regression` / `invariant` / `untested`), oracle, tolerance, backing test and benchmark. Column meanings are in SPEC §3. |
| [`coverage_report.md`](coverage_report.md) | The headline "features vetted by ≥1 oracle", overall and per family. Generated from the registry by `check_coverage.py --write`; never edit by hand. |
| [`check_coverage.py`](check_coverage.py) | Validates the registry (columns, `status`/`oracle`/`source` values, and that every `benchmark`, `config_recipe` and `test_name` resolves) and regenerates `coverage_report.md`. `--check` also fails when the committed report is stale or missing. Stdlib only. |
| [`check_test_names.py`](check_test_names.py) | Enforces the SPEC §6.1 file-name, §6.2 function-name and gtest case-name conventions, and the §6.3.1 rules for golden reference tables and family-header includes, over the whole test tree. `--check` fails in CI. |
| [`config_recipes.md`](config_recipes.md) | The config recipes (SPEC §5): the exact Nyxus and reference-tool settings that make a feature directly comparable, one `## <id>` section each, referenced by id from the registry's `config_recipe` column. |
| [`matrix/`](matrix/) | Per-family config matrices (SPEC §5.1): the settings a family actually reads, the config points they produce, and each point's measured verdict. |
| [`benchmarks.md`](benchmarks.md) | The fixtures assertions run on (SPEC §6.3): what each one is, why it exists, and which recipes and tests use it. A `benchmark` id in the registry must be defined here, which `check_coverage.py` enforces — as it does for `test_name`, which must resolve to a gtest case in `test_all.cc`. |
| [`oracles/`](oracles/) | The offline golden generators, `gen_<family>_<oracle>.*` (Python, or `.m` for `matlab`), one per family and reference tool, plus the IBSI digital phantom (`ibsi_phantom.py`) and the `fraclac/` box-count macro. Reference tools are never CI dependencies: the generators are run by hand and their output is pinned in the tests. |
| [`TOOLS.md`](TOOLS.md) | How to stand up each oracle tool locally (Docker, conda or a Python venv; licensed MATLAB or GNU Octave for `matlab`) — per-tool setup, feasibility, coverage by family, and notable findings. |
| [`report_features.py`](report_features.py) | Generates [`report_output.csv`](report_output.csv) and [`report_output.md`](report_output.md) — the whole report, one row per (dim, feature, oracle, config_recipe), joining what the registry CLAIMS to what the tree ASSERTS. Its `verdict` column is the join and `verdict_scope` says how far that join reached — `row+config` compared the row's own assertion at its own config, `row` its own assertion, `feature` only the feature, which is all a row with no `test_name` can be asked. `report_output.csv` also carries each feature's scanner note (`scan_notes`). `--check` fails in CI when either artifact is stale, on any row that disagrees without an allowlisted reason, on an unscanned family not recorded in `KNOWN_UNSCANNED`, and on two rows that share a key. **This is the artifact to read.** |
| [`report_output.md`](report_output.md) | The generated report: coverage, verdicts, the families no scanner covers, the family × oracle matrix, the dimensionality split. Never edit by hand. |
| [`not_covered.md`](not_covered.md) | Tests the registry does not reference, and tests that never run (not included, not registered, or excluded by a build flag). |
| [`MIGRATION.md`](MIGRATION.md) | The record of how the test tree was reorganized onto the SPEC taxonomy: the original migration map and gaps register, then each rename and placement wave. |
| [`audit/`](audit/) | Baseline coverage audit of the test tree (see below), and the per-family coverage scanners that keep it current. |
| [`audit/scanlib.py`](audit/scanlib.py) | The scanners' shared machinery: the coverage rule and the acceptance checks, in one place. Each `audit/scan_*_coverage.py` is the family's declaration on top of it — which files to read, how the family spells its feature names in C++, which checks apply. |
| [`audit/run_scanners.py`](audit/run_scanners.py) | Runs every family scanner's `--check`, reports all of them, then fails once. Runs in CI. |
| [`audit/scanlib_selftest.py`](audit/scanlib_selftest.py) | Negative controls for the coverage rule itself, on fixtures rather than on the tree: a literal the case never loops, a returned value the caller discards, a gtest case registered on one line, and a registry row whose `config_recipe` has been swapped for another of its family's. The per-family `--check` cannot catch these — it reads the tree through the very rule being tested. Runs in CI. |

## `audit/` — baseline snapshot

A point-in-time audit of what the test tree vetted when the registry was seeded, produced by four
parallel review passes plus an authoritative token scan. **Baseline caveat:** it reflects the tree
as merged at that time, so it predates the GLCM entropy/hom2 normalization fix and the histogram
and GLCM oracle tests that landed afterwards. The registry has been maintained forward from this
baseline; the audit is kept as it was.

Tracked artifacts (the rest are regenerable — see below):

- `vetting_report.csv` — master, all 758 real features: `dim, group, family, feature,
  vetting_status, oracle_used, test_files, candidate_oracle, how_to_vet`
- `TEST_VETTING_REPORT.md` — narrative findings (758 features: vetted / claimed-3p / regression /
  not-tested; the `#if 0` "planned for a future PR" 3D-feature correction; includes the group×status
  pivot as a table)
- `extract_features.py` → `scan_tests.py` → `merge.py` → `report.py` (+ `candidate_oracles.py`) —
  the pipeline (repo-relative paths); `audit_1..4.txt` are the raw per-feature review inputs.
- `*_vetting_report.md` (e.g. `glcm_2d_mirp_vetting_report.md`) and `*_golden_regen.md` (e.g.
  `glcm_2d_golden_regen.md`) — the record of each family's vetting runs: what the oracle was run
  on, what it measured, and how the pinned goldens were regenerated.

### The per-family scanners

`audit/scan_<family>_coverage.py` reads the family's feature -> test mapping out of the test
sources, so it cannot drift from the tree, and runs that family's acceptance checks against
`oracle_coverage.csv`. A scanner writes no file of its own: `report_features.py` imports every
declaration and joins the scan into `report_output.csv`. Every family in the registry has one, and all
twenty-five share `scanlib.py`; a family whose tests genuinely read differently (2D moments resolves golden tables, not
assertion lines) or which judges differently (2D NGTDM, IMQ, 3D GLDM and 3D GLSZM check each row
against the tests of its own KIND) overrides that part in its own file, where the difference is
visible.

All twenty-five run in CI through [`audit/run_scanners.py`](audit/run_scanners.py), which runs
every family before failing and names the ones that failed, beside `check_coverage.py`,
`check_test_names.py`, `scanlib_selftest.py` and `report_features.py --check`.

Regenerable byproducts of the retired pipeline, all git-ignored and all under `audit/`:
`audit/features.csv`, `audit_scan.txt`, `vetting_pivot.csv`, `gap_not_tested.csv`,
`gap_claimed_3p.csv`, `gap_regression.csv`. `audit/features.csv` is not the tracked
`tests/vetting/report_output.csv` the table above links. Run
`extract_features.py && scan_tests.py && merge.py && report.py` from this folder to rebuild them
(and to refresh `vetting_report.csv`).

## `oracles/` — golden generators

- `gen_<family>_<oracle>.py` / `.m` — one generator per family and reference tool (for example
  `gen_glcm_mirp.py`, `gen_firstorder2d_matlab.m`, `gen_imq_opencv.py`). Each one produces the
  goldens a `test_<dim>_<family>_<oracle>` file pins, and states the tool version and settings
  the test's provenance header records (SPEC §6.4).
- `ibsi_phantom.py` — the IBSI digital phantom, shared by the generators that run on it.
- `fraclac/shiftgrid_boxcount.ijm` — headless ImageJ macro reproducing FracLac's shifting-grid
  box-counting (the FracLac plugin itself is GUI-only). `ref_boxcount.py` is a numpy cross-check.
  See TOOLS.md → `fraclac` for the reconciliation (from-method reimplementation vs. true-tool run).

## Status

In force. The test tree follows SPEC.md, and CI (`.github/workflows/build_and_test_ubuntu.yml`)
runs `check_coverage.py --check`, `check_test_names.py --check`, `audit/scanlib_selftest.py`,
`audit/run_scanners.py` and `report_features.py --check` on every change. The current coverage
figure is the headline of [`coverage_report.md`](coverage_report.md).
