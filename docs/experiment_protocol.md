# Experiment protocol

## Data-use boundary

The workflow uses a chronological train/validation/test split at complete
five-minute update-period boundaries. Training and validation data determine
features, scalers, model architecture, mixture count, quantile-tree settings,
block length, interval construction, and publication thresholds. The test set
is scored only after these choices are written to `selection_lock.json`.

## Development stages

1. `prepare` constructs completion-aware features and materializes the frozen
   complete-period split.
2. `stage-a` and `stage-b` compare snapshot and recurrent density-model
   configurations on training and validation data.
3. `k-scan` applies the validation one-standard-error rule to mixture count.
4. `fixed-references` fits the naive and ridge reference distributions.
5. `trees` selects snapshot and explicit-lag quantile GBDTs on validation data.
6. `ap03` evaluates native quantiles, induced Gaussian distributions, and
   positive-support corrections on development data.
7. `ap01` runs completion-delay and feature-block diagnostics.
8. `ap06` selects publication policies on complete validation periods.
9. `final-models` materializes the selected four-seed neural components.
10. `freeze` hashes all selection artifacts and records the test-access guard.

## Confirmatory evaluation

`frozen_evaluation.py` performs one frozen test batch for predictive scores,
paired representation effects, support diagnostics, moving-period-block
intervals, Holm-adjusted families, service policies, and service KPIs. The
inference benchmark uses one prepared update state per call and excludes model
loading and feature construction, matching the paper's computational protocol.

## Second-revision additions

The analyses added in the second revision were specified after the original
test outputs had been reported. They reuse the frozen artifacts only, except
for the cadence/window stage, which re-reads the configured input tables to
rebuild features at other update intervals. Run them after the frozen
evaluation, in this order:

1. `r2_experiment_runner.py endpoints` fits directly fitted 0.05/0.95
   interval endpoints for the snapshot and explicit-lag quantile GBDTs,
   reusing each model's original validation hyperparameters, and compares
   them against the induced-Gaussian endpoints on the frozen test batch.
2. `r2_experiment_runner.py service` freezes gate rules and uniform-scaling
   factors on validation data, then compares the service policies
   (no-gate, balanced and reliability-first gates, uniform scaling at the
   nominal and gate-matched targets) on the test batch with the original
   5000 moving-block indices.
3. `r2_experiment_runner.py sensitivity` reruns the snapshot quantile-GBDT
   comparison at six frozen cadence/window settings (update intervals 2, 5
   and 10 minutes; entry windows 5, 10, 15, 30 and 60 minutes), selecting
   separately for the deployable and 30-minute look-ahead information sets
   on validation mean pinball loss, with 150-minute gap-respecting blocks.
   It re-reads only the input tables inside `data/`.
4. `ablation_retuned.py` retunes each feature-ablation variant of the
   snapshot quantile GBDT on validation mean pinball loss from the same
   24-configuration grid, instead of reusing the full-feature
   configuration, and reports fixed-configuration and retuned effects side
   by side with Holm adjustment within the three feature comparisons.
5. `scripts/r2_supplement.py` recomputes the validation ablation and the
   endpoint differences from the stored artifacts, adds the full-support
   block resampling for the cadence/window comparisons, and draws the
   service-comparison and cadence/window figures into
   `outputs/paper_results_r2/figures/`.
6. `r2_window5.py` adds the seventh cadence/window setting (5-minute
   update interval, 5-minute entry window) with the same procedure as the
   six frozen settings, and redraws the cadence/window figure with all
   seven settings. It writes to `outputs/paper_results_r2_w5/` so the
   frozen plan and results are untouched.
7. `gate_selectivity.py` is a diagnostic, not a model run: it re-aggregates
   the published per-case service file
   (`outputs/paper_results/ap10/service_kpi_per_case.csv`) with the frozen
   bootstrap indices to test whether the selected gate identifies
   unreliable periods or only widens intervals. Without a run directory it
   regenerates the indices and skips the validation-based sections.

The second-revision runner freezes its plan (`plan.json`) and input manifest
(`input_manifest.json`) under `outputs/paper_results_r2/` on the first run
and refuses to reuse artifacts if either changes. It requires the frozen
run's `selection_lock.json` to exist. The frozen plan and the input manifest
are the public counterparts of the private run's anonymized-data
verification step, which has no counterpart in this release.
