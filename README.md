# Issue-Time ANPR Forecasting

Code for the experiments in "Issue-Time Probabilistic Travel Time
Forecasting under Completion Delay for an ANPR-Instrumented Urban Arterial"
(Hanwu Qin, Dianhai Wang, Meng Zhang, Shenao Bao; College of Civil
Engineering and Architecture, Zhejiang University), published in
*Transportation Research Part C: Emerging Technologies* 195, 106177
(2027), https://doi.org/10.1016/j.trc.2026.106177
(manuscript TRC-26-01835R3, accepted 2026-11-18).

This repository was formerly named `availability-aware-anpr-forecasting`;
its new name is `issue-time-anpr-forecasting`
(`https://github.com/chin1hanwu/issue-time-anpr-forecasting`).
The old URL, `https://github.com/chin1hanwu/availability-aware-anpr-forecasting`,
redirects here after the rename; the data-availability statement in the
published paper points to this repository.

## What the paper finds

Using 113,006 matched trajectories from a Hangzhou signalized arterial, the
paper asks whether a probabilistic forecast built only from information
available at the issue time can be accurate and reliable, and which
components achieve this. Headline results:

- Across seven cadence/window settings (2- to 10-minute update intervals,
  5- to 60-minute entry windows), separately tuned comparisons detect no
  look-ahead CRPS improvement from adding completions up to 30 minutes
  later; 95% block intervals bound the gain at 0.272 s, as expected for a
  corridor travel time of about 81 s.
- Explicit-lag quantile boosting improves CRPS over a recurrent mixture
  density network by 1.487 s (Holm-adjusted p = 0.003), and mixture-density
  training improves a shared recurrent encoder by 1.630 s.
- Directly fitted 0.05/0.95 interval endpoints with uniform scaling attain
  period-weighted coverage 0.908 at width 41.9 s, and at a common
  validation target give intervals 27.3 s narrower than an availability
  gate.

The implementation covers completion-aware ANPR preprocessing,
complete-period chronological splitting, snapshot and recurrent mixture
density networks, quantile gradient-boosted trees, validation-only model
and policy selection, positive-support diagnostics, moving-period-block
inference, frozen test evaluation, service-level reliability metrics, CPU
inference benchmarking, and the second-revision analyses (directly fitted
endpoints, service-policy comparison, cadence/window sensitivity, and
variant-specific feature ablations).

## Privacy and release boundary

The original ANPR event records and metadata remain governed by the data
provider under applicable privacy constraints. The configuration files use
generic camera identifiers and relative input paths, and this repository
contains no original or derived ANPR records.

The paper's exact split counts and experiment grids are retained in
`experiment_config.yaml`. Reproducing the reported numerical results
requires authorized access to the original records. The included unit tests
and synthetic event generator exercise the public interfaces with synthetic
records only.

## Repository contents

- `data_processing.py`: event matching, availability-aware features,
  trajectory alignment, and complete-period splitting.
- `forecast_models.py`: snapshot and recurrent neural models and training code.
- `probabilistic_metrics.py`: probabilistic scores, support corrections, block
  resampling, and multiplicity-aware utilities.
- `experiment_runner.py`: training/validation selection stages and
  selection freezing.
- `frozen_evaluation.py`: one-shot frozen test evaluation and service KPIs.
- `r2_experiment_runner.py`: second-revision analyses — directly fitted
  interval endpoints, service-policy comparison, and cadence/window
  sensitivity. Reads only the frozen artifacts under
  `outputs/paper_results/`, except for the sensitivity stage, which re-reads
  the configured input tables.
- `ablation_retuned.py`: feature ablations with variant-specific tuning on
  validation mean pinball loss.
- `gate_selectivity.py`: diagnostic that re-aggregates the published
  per-case service file to test whether the selected gate identifies
  unreliable periods or only widens intervals. No model is refitted.
- `r2_window5.py`: the seventh cadence/window setting (5-minute update
  interval, 5-minute entry window), run with the same procedure as the six
  frozen settings.
- `scripts/benchmark_inference.py`: CPU inference benchmark used in
  the computational-efficiency experiment.
- `scripts/r2_supplement.py`: recomputation of the validation ablation and
  endpoint differences, full-support block resampling, and the
  second-revision figures.
- `docs/experiment_protocol.md`: execution order and data-use boundaries,
  including the second-revision additions.
- `tests/`: deterministic synthetic unit tests, including the
  second-revision boundary tests (`test_r2_experiments.py`,
  `test_service_kpi_granularity.py`).

## Environment

The reported experiments used Python 3.9, scikit-learn 1.6.1, and PyTorch
2.7.1. Install the platform-appropriate PyTorch build when GPU training is
required. `threadpoolctl` is required by `r2_experiment_runner.py` and
`matplotlib` only by the figure step in `scripts/r2_supplement.py` and
`r2_window5.py`.

```powershell
conda activate tf_env
python -m pip install -r requirements.txt
```

## Input schema

Set `plate1_path`, `plate2_path`, `camera_up`, and `camera_down` in
`study_config.yaml`. Relative input paths resolve under `data/` (see
`data_processing._resolve_data_path`); absolute paths are used as given.
Each event table must contain:

| Column | Meaning |
| --- | --- |
| `leave_time` | Event timestamp parseable by pandas |
| `vehicle_id` | Identifier used only for upstream/downstream matching |
| `camera_id` | Camera identifier used by the configured corridor filter |
| `turn_id` | Movement identifier; the study filter uses value `2` |

Vehicle and camera identifiers must be handled according to the applicable
data-governance agreement. Keep event tables and generated outputs outside
version control.

The synthetic generator illustrates this schema with fictional identifiers:

```powershell
conda activate tf_env
python examples/generate_synthetic_events.py
```

The synthetic files support interface checks; authorized original records are
required to reproduce the paper's sample size and results.

## Tests

```powershell
conda activate tf_env
python -m unittest discover -s tests -p "test_*.py"
```

## Reproducing the experiment workflow

After placing authorized input tables at the configured paths, execute the
development stages in order. All model, threshold, and block-length choices use
training and validation data only.

```powershell
conda activate tf_env
python experiment_runner.py prepare
python experiment_runner.py stage-a --family snapshot
python experiment_runner.py stage-a --family recurrent
python experiment_runner.py stage-b
python experiment_runner.py k-scan
python experiment_runner.py fixed-references
python experiment_runner.py trees --kind snapshot
python experiment_runner.py trees --kind lagged
python experiment_runner.py ap03
python experiment_runner.py ap01
python experiment_runner.py ap06
python experiment_runner.py final-models
python experiment_runner.py freeze
```

Open the frozen test set once, then run the CPU benchmark:

```powershell
conda activate tf_env
python frozen_evaluation.py
python scripts/benchmark_inference.py
```

The workflow writes generated artifacts under `outputs/paper_results/`, an
ignored output directory.

### Second-revision analyses

Run these after the frozen evaluation. The runner freezes its plan and
input manifest under `outputs/paper_results_r2/` on the first run and
refuses to reuse artifacts if either changes; see
`docs/experiment_protocol.md` for the data-use boundary.

```powershell
conda activate tf_env
python r2_experiment_runner.py endpoints
python r2_experiment_runner.py service
python r2_experiment_runner.py sensitivity
python ablation_retuned.py
python scripts/r2_supplement.py
python gate_selectivity.py outputs/paper_results/ap10/service_kpi_per_case.csv outputs/paper_results
python r2_window5.py
```

`r2_experiment_runner.py` also accepts `all` to run the endpoints, service,
and sensitivity stages in one go. `r2_window5.py` accepts `audit` (check the
extension against the frozen manifest only) and `figure` (redraw the
cadence/window figure with all seven settings only). The second-revision
outputs go to `outputs/paper_results_r2/` and `outputs/paper_results_r2_w5/`,
both ignored output directories.

## Citation

If you use this code, please cite the associated paper:

> Qin, H., Wang, D., Zhang, M., & Bao, S. (2027). Issue-Time Probabilistic
> Travel Time Forecasting under Completion Delay for an ANPR-Instrumented
> Urban Arterial. *Transportation Research Part C: Emerging Technologies*,
> *195*, 106177. https://doi.org/10.1016/j.trc.2026.106177

See `CITATION.cff` for the machine-readable citation.

## License

The code is released under the Apache License 2.0. See `LICENSE`.
