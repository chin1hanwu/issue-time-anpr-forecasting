"""Recompute the validation ablation and endpoint differences, and generate the second-revision figures.

Reads only local derived artifacts, except for `sensitivity_full_support`,
which re-reads the configured input tables through `data_processing` to map
test cases back to the full update-period sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import r2_experiment_runner as r2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probabilistic_metrics import moving_block_period_indices, pinball_loss


def validation_ablation():
    split = r2.load_split("val")
    periods = split["meta"].update_period_id.drop_duplicates().to_numpy()
    mask = ~split["meta"].update_period_id.isin(periods[-6:]).to_numpy()
    meta, y = split["meta"].loc[mask].reset_index(drop=True), split["y"][mask]
    order, indices = moving_block_period_indices(meta, 30, 5000, r2.SEED)
    counts = meta.groupby("update_period_id").size().to_numpy()
    base = np.load(r2.OLD / "ap01/validation_predictions/full_h0.npz")["pinball"].astype(float).mean(axis=1)
    rows = []
    for variant in ["full_h0", "full_h1800", "no_rho_carrier", "no_completion_block", "no_moment_block", "base"]:
        path = r2.OLD / f"ap01/validation_predictions/{variant}.npz"
        z = np.load(path)
        predictions = z["quantiles"].astype(float)
        loss = z["pinball"].astype(float).mean(axis=1)
        reconstructed = np.column_stack([pinball_loss(y, predictions[:, j], q) for j, q in enumerate([0.1, 0.5, 0.9])]).mean(axis=1)
        if not np.allclose(loss, reconstructed, atol=2e-5):
            raise RuntimeError("Stored losses disagree with the recomputation from predictions.")
        period_difference = r2.pm(loss - base, meta).reshape(-1)
        boot = (period_difference[indices] * counts[indices]).sum(axis=1) / counts[indices].sum(axis=1)
        lo, hi = np.minimum(predictions[:, 0], predictions[:, 2]), np.maximum(predictions[:, 0], predictions[:, 2])
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({"variant": variant, "pinball": loss.mean(), "difference": (loss - base).mean(),
                     "low": low, "high": high, "coverage80": np.mean((y >= lo) & (y <= hi)),
                     "n_cases": len(y), "n_periods": len(order), "source_sha256": r2.digest(path)})
    folder = r2.OUT / "ablation"
    folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "validation_summary.csv", index=False)
    np.savez_compressed(folder / "bootstrap_indices.npz", update_period_id=order.update_period_id, indices=indices)


def endpoint_effects():
    test = r2.load_split("test")
    boot = np.load(r2.OLD / "ap09/bootstrap_indices.npz")["indices"]
    rows = []
    for kind in ["snapshot", "lagged"]:
        pred = np.load(r2.OUT / f"endpoints/{kind}_test.npz")
        direct = r2.interval_values(test["y"], pred["lo"], pred["hi"])
        gaussian = r2.interval_values(test["y"], pred["original_lo"], pred["original_hi"])
        differences = r2.pm(direct - gaussian, test["meta"])
        limits = r2.ci(differences, boot)
        for j, name in enumerate(r2.METRICS):
            rows.append({"kind": kind, "metric": name, "direct_minus_gaussian": differences[:, j].mean(),
                         "low": limits[0, j], "high": limits[1, j]})
    pd.DataFrame(rows).to_csv(r2.OUT / "endpoints/paired_effects.csv", index=False)


def sensitivity_full_support():
    """Block resampling covering all periods as a supplement; the original moving-block results are kept for comparison."""
    import yaml
    from data_processing import load_matched_data
    config = yaml.safe_load((ROOT / "study_config.yaml").read_text(encoding="utf-8"))
    _, _, matched = load_matched_data(config)
    rows = []
    for folder in sorted((r2.OUT / "sensitivity").glob("d*_w*")):
        if not (folder / "completed.json").exists():
            continue
        delta, window = map(int, folder.name.replace("d", "").split("_w"))
        deployable, ahead = np.load(folder / "deployable_test.npz"), np.load(folder / "lookahead_test.npz")
        ids = deployable["case_id"]
        assert np.array_equal(ids, ahead["case_id"])
        all_times = matched.timestamp_up.dt.floor(f"{delta}min")
        all_periods, _ = pd.factorize(all_times, sort=False)
        meta = pd.DataFrame({"update_period_id": all_periods[ids], "prediction_time": all_times.iloc[ids].to_numpy()})
        period_table = meta.drop_duplicates("update_period_id").reset_index(drop=True)
        original = np.load(folder / "bootstrap_indices.npz")
        assert np.array_equal(period_table.update_period_id, original["update_period_id"])
        step = pd.Timedelta(minutes=delta)
        breaks = np.flatnonzero(period_table.prediction_time.diff().fillna(step) != step)
        blocks = []
        for start, end in zip(np.r_[0, breaks], np.r_[breaks, len(period_table)]):
            for left in range(start, end, 150 // delta):
                blocks.append(np.arange(left, min(left + 150 // delta, end)))
        assert np.array_equal(np.concatenate(blocks), np.arange(len(period_table)))
        difference = ahead["scores"] - deployable["scores"]
        summed = pd.Series(difference).groupby(meta.update_period_id.to_numpy()).sum().to_numpy()
        counts = meta.groupby("update_period_id").size().to_numpy()
        block_sum = np.array([summed[b].sum() for b in blocks])
        block_count = np.array([counts[b].sum() for b in blocks])
        choices = np.random.default_rng(r2.SEED).integers(0, len(blocks), size=(5000, len(blocks)))
        boot = block_sum[choices].sum(axis=1) / block_count[choices].sum(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({"delta_minutes": delta, "window_minutes": window, "metric": "crps",
                     "lookahead_minus_deployable": difference.mean(), "low": low, "high": high,
                     "n_blocks": len(blocks), "n_periods": len(period_table),
                     "moving_block_eligible_periods": len(np.unique(original["indices"]))})
        np.savez_compressed(folder / "full_support_block_indices.npz", block_choices=choices,
                            block_starts=[b[0] for b in blocks], block_lengths=[len(b) for b in blocks])
    pd.DataFrame(rows).to_csv(r2.OUT / "sensitivity/full_support_effects.csv", index=False)
    r2.write_json(r2.OUT / "sensitivity/uncertainty_amendment.json", {
        "reason": "Fixed-length moving blocks cannot cover short consecutive segments; at the 2-minute setting 203 of 958 periods are never sampled.",
        "method": "All consecutive segments are blocked at most 150 minutes, keeping short segments and tail blocks; blocks are drawn in pairs and normalized by actual trajectory counts.",
        "scope": "All six settings use the same supplementary algorithm; point estimates, models, selection and the original moving-block results are unchanged.",
        "interpretation": "95% pointwise block intervals; no multiplicity correction.",
    })


def make_figures():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    folder = r2.OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    service = pd.read_csv(r2.OUT / "service/test_summary.csv")
    effects = pd.read_csv(r2.OUT / "service/paired_effects.csv")
    names = ["no_gate", "balanced", "reliability_first", "uniform_nominal", "uniform_match_gate"]
    labels = ["Direct endpoints", "Balanced gate", "Reliability-first gate", "Uniform: nominal target", "Uniform: gate target"]
    data = service[service.constructor.eq("direct")].set_index("rule").loc[names]
    fig, axes = plt.subplots(2, 1, figsize=(7.3, 5.5), constrained_layout=True)
    colors = ["#15607A", "#B07622", "#AA4D45", "#287B58", "#665291"]
    for i, (_, row) in enumerate(data.iterrows()):
        axes[0].scatter(row.width, row.coverage, s=55, color=colors[i], label=labels[i], zorder=3)
    axes[0].axhline(0.90, color="0.5", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="Mean interval width (s)", ylabel="Period-weighted coverage", title="(a) Test coverage and width")
    axes[0].legend(fontsize=8, loc="lower right", frameon=True,
                   facecolor="white", edgecolor="none", framealpha=1)
    pos = np.arange(len(names))
    error = np.vstack([data.interval_score-data.interval_score_low, data.interval_score_high-data.interval_score])
    axes[1].errorbar(data.interval_score, pos, xerr=error, fmt="none", ecolor="0.55", capsize=3)
    axes[1].scatter(data.interval_score, pos, color=colors, s=45, zorder=3)
    axes[1].set(yticks=pos, yticklabels=labels, xlabel="Interval score (s)", title="(b) Score and 95% block intervals")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "service_comparison_r2.pdf")
    fig.savefig(folder / "service_comparison_r2.png", dpi=160)
    plt.close(fig)
    path = r2.OUT / "sensitivity/full_support_effects.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame[frame.metric.eq("crps")]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), constrained_layout=True)
    for ax, subset, xfield, label, title in [
        (axes[0], frame[frame.window_minutes.eq(30)].sort_values("delta_minutes"), "delta_minutes", "Update interval (min)", "(a) Window fixed at 30 min"),
        (axes[1], frame[frame.delta_minutes.eq(5)].sort_values("window_minutes"), "window_minutes", "Entry-window length (min)", "(b) Update interval fixed at 5 min"),
    ]:
        y = subset.lookahead_minus_deployable
        ax.errorbar(subset[xfield], y, yerr=np.vstack([y-subset.low, subset.high-y]), fmt="o-", capsize=4, color="#15607A")
        ax.axhline(0, linestyle="--", color="0.5", linewidth=0.8)
        ax.set(xlabel=label, ylabel="Look-ahead minus deployable CRPS (s)", title=title, xticks=subset[xfield])
        ax.grid(alpha=0.2)
    fig.savefig(folder / "cadence_window_r2.pdf")
    fig.savefig(folder / "cadence_window_r2.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    validation_ablation()
    endpoint_effects()
    if (r2.OUT / "sensitivity/summary.csv").exists():
        sensitivity_full_support()
    make_figures()
    print("Supplementary computations and second-revision figures generated.")
