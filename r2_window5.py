"""Additional cadence-window setting: update interval 5 min, entry window 5 min.

Runs the same procedure as the six frozen settings of r2_experiment_runner.py
(separate validation selection for deployable and 30-min look-ahead inputs on the
same four-candidate grid, 30-min end-of-segment exclusion, induced-Gaussian CRPS)
and the same gap-respecting 150-min block intervals as scripts/r2_supplement.py.
Output goes to outputs/paper_results_r2_w5 so the frozen plan and results are untouched.

Usage (tf_env):  python r2_window5.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import r2_experiment_runner as r2

r2.OUT = ROOT / "outputs/paper_results_r2_w5"
import r2_supplement  # noqa: E402  (uses the patched r2.OUT through the shared module)


def main():
    frozen = r2.read_json(ROOT / "outputs/paper_results_r2/plan.json")
    plan = {"sensitivity_settings": [[5, 5]], "sensitivity_grid": frozen["sensitivity_grid"]}
    r2.write_json(r2.OUT / "plan_extension.json", {
        "setting": [5, 5], "grid": frozen["sensitivity_grid"], "seed": r2.SEED,
        "note": "Reported regardless of outcome; same procedure as the six frozen settings.",
    })
    r2.sensitivity_stage(plan)
    r2_supplement.sensitivity_full_support()


def audit():
    """Record that the extension used the same inputs and environment as the frozen run."""
    import platform
    import numpy, pandas, scipy, sklearn
    frozen = r2.read_json(ROOT / "outputs/paper_results_r2/input_manifest.json")
    current = {name: r2.digest(ROOT / name) for name in frozen}
    mismatched = [name for name in frozen if current[name] != frozen[name]]
    r2.write_json(r2.OUT / "input_check.json", {
        "reference": "outputs/paper_results_r2/input_manifest.json",
        "files": current, "mismatched": mismatched, "passed": not mismatched})
    r2.write_json(r2.OUT / "environment.json", {
        "python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
        "sklearn": sklearn.__version__, "scipy": scipy.__version__})
    r2.write_json(r2.OUT / "sensitivity/uncertainty_amendment.json", {
        "method": "All consecutive segments are blocked at most 150 minutes, keeping short segments and tail blocks; blocks are drawn in pairs and normalized by actual trajectory counts.",
        "scope": "Only this extension setting (5-minute update interval, 5-minute window), using the same algorithm as the original six settings.",
        "interpretation": "95% pointwise block intervals; no multiplicity correction."})
    if mismatched:
        raise RuntimeError(f"Inputs differ from the frozen run: {mismatched}")


def figure_and_supplement():
    """Redraw Figure 5 with all seven settings and export the extension rows."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = ROOT / "outputs/paper_results_r2/sensitivity"
    extension = ROOT / "outputs/paper_results_r2_w5/sensitivity"
    frame = pd.concat([pd.read_csv(original / "full_support_effects.csv"),
                       pd.read_csv(extension / "full_support_effects.csv")], ignore_index=True)
    frame = frame[frame.metric.eq("crps")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), constrained_layout=True)
    for ax, subset, xfield, label, title in [
        (axes[0], frame[frame.window_minutes.eq(30)].sort_values("delta_minutes"), "delta_minutes",
         "Update interval (min)", "(a) Window fixed at 30 min"),
        (axes[1], frame[frame.delta_minutes.eq(5)].sort_values("window_minutes"), "window_minutes",
         "Entry-window length (min)", "(b) Update interval fixed at 5 min"),
    ]:
        y = subset.lookahead_minus_deployable
        ax.errorbar(subset[xfield], y, yerr=np.vstack([y - subset.low, subset.high - y]),
                    fmt="o-", capsize=4, color="#15607A")
        ax.axhline(0, linestyle="--", color="0.5", linewidth=0.8)
        ax.set(xlabel=label, ylabel="Look-ahead minus deployable CRPS (s)", title=title, xticks=subset[xfield])
        ax.grid(alpha=0.2)
    folder = r2.OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "cadence_window_r2.pdf")
    fig.savefig(folder / "cadence_window_r2.png", dpi=160)
    plt.close(fig)

    summary = pd.read_csv(extension / "summary.csv")
    effects = pd.read_csv(extension / "full_support_effects.csv")
    moving = pd.read_csv(extension / "paired_effects.csv")
    supplement = extension
    supplement.mkdir(parents=True, exist_ok=True)
    summary.to_csv(supplement / "cadence_window_w5_metrics.csv", index=False)
    effects.to_csv(supplement / "cadence_window_w5_effects.csv", index=False)
    moving.to_csv(supplement / "cadence_window_w5_moving_block_effects.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figure":
        figure_and_supplement()
    elif len(sys.argv) > 1 and sys.argv[1] == "audit":
        audit()
    else:
        audit()
        main()
        audit()
        figure_and_supplement()
