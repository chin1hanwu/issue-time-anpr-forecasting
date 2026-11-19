"""Feature ablation with variant-specific tuning.

Each ablation variant of the snapshot quantile GBDT is selected on validation mean
pinball loss from the same 24-configuration grid used for the full model, instead of
reusing the full-feature configuration. Support, scoring, block bootstrap, and p-values
follow the original AP-01 protocol (final six periods of each split excluded, induced
Gaussian CRPS, moving blocks of 30 periods, 5000 replicates, test seed 29042).

As a reproduction check, the run with the original fixed configuration must return the
published AP-01 effects.

Holm adjustment is applied to the three feature comparisons of Table 4 as a separate family.
The original AP-01 family also contained the 30-min look-ahead; its separately tuned
counterpart comes from the cadence and window comparisons and is not part of this family.
Validation intervals are conditional on the selected configurations and do not include
the uncertainty of re-tuning.

Usage (tf_env):  python ablation_retuned.py
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from probabilistic_metrics import density_metrics, moving_block_period_indices, pinball_loss

ROOT = Path(__file__).resolve().parent
G0 = ROOT / "outputs/paper_results/g0"
OUT = ROOT / "outputs/paper_results_r2/ablation_retuned_v2"
QUANTILES = [0.1, 0.5, 0.9]
FIXED_ID = "d3_lr0.1_i200_l21.0"  # configuration selected for the full feature set

FEATURE_SETS = {
    "full_h0": list(range(14)),
    "no_rho_carrier": [i for i in range(14) if i != 3],
    "no_completion_block": [i for i in range(14) if i not in (1, 2)],
    "no_moment_block": [i for i in range(14) if i not in (3, 4)],
    "base": [0] + list(range(5, 14)),
}
TABLE4_FAMILY = ["no_rho_carrier", "no_completion_block", "base"]


def load(split):
    meta = pd.read_csv(G0 / f"{split}_metadata.csv")
    X = np.load(G0 / f"{split}_X_raw.npy")[:, -1, :]
    y = np.load(G0 / f"{split}_y_raw.npy").reshape(-1).astype(float)
    ordered = meta["update_period_id"].drop_duplicates().to_numpy()
    keep = ~meta["update_period_id"].isin(set(ordered[-6:])).to_numpy()
    return meta.loc[keep].reset_index(drop=True), X[keep], y[keep]


def grid():
    cfg = yaml.safe_load((ROOT / "experiment_config.yaml").read_text(encoding="utf-8"))["gbdt"]
    for depth, lr, it, l2 in itertools.product(cfg["max_depth"], cfg["learning_rate"],
                                              cfg["max_iter"], cfg["l2_regularization"]):
        yield {"id": f"d{depth}_lr{lr}_i{it}_l2{float(l2)}", "max_depth": depth,
               "learning_rate": float(lr), "max_iter": int(it), "l2": float(l2),
               "min_leaf": int(cfg["min_samples_leaf"])}


def fit_predict(spec, X_train, y_train, X_eval):
    preds = []
    for q in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_depth=spec["max_depth"],
            learning_rate=spec["learning_rate"], max_iter=spec["max_iter"],
            l2_regularization=spec["l2"], min_samples_leaf=spec["min_leaf"],
            max_bins=255, early_stopping=False, random_state=0)
        model.fit(X_train, y_train)
        preds.append([model.predict(X) for X in X_eval])
    return [np.column_stack([p[k] for p in preds]) for k in range(len(X_eval))]


def induced_crps(q, y):
    lower, upper = np.minimum(q[:, 0], q[:, 2]), np.maximum(q[:, 0], q[:, 2])
    mu = np.clip(q[:, 1], lower, upper)
    sigma = np.maximum((upper - lower) / 2.563103, 1e-3)
    return density_metrics(np.ones((len(y), 1)), mu[:, None], sigma[:, None], y)["crps"]


def mean_pinball(q, y):
    return np.column_stack([pinball_loss(y, q[:, j], t) for j, t in enumerate(QUANTILES)]).mean(axis=1)


def block_boot(values, meta, period_index, blocks):
    order = period_index["update_period_id"].to_numpy()
    g = pd.DataFrame({"p": meta["update_period_id"].to_numpy(), "v": values}).groupby("p")["v"]
    sums = g.sum().reindex(order, fill_value=0.0).to_numpy()
    counts = g.count().reindex(order, fill_value=0.0).to_numpy(dtype=float)
    return sums[blocks].sum(axis=1) / counts[blocks].sum(axis=1)


def paired(diff, meta, period_index, blocks):
    boot = block_boot(diff, meta, period_index, blocks)
    point = float(diff.mean())
    centered = boot - boot.mean()
    p = float((1 + np.sum(np.abs(centered) >= abs(point))) / (len(centered) + 1))
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), p


def holm(ps):
    ps = np.asarray(ps, float)
    order, out, running = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_predictions(name, config, fingerprints):
    """Load cached predictions only if their stored fingerprint matches the current run."""
    z = np.load(OUT / f"{name}__{config}.npz")
    stored = json.loads(str(z["fingerprint"]))
    if stored != fingerprints[(name, config)]:
        raise RuntimeError(f"Cache fingerprint mismatch for {name} {config}")
    return z["val"], z["test"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (tm, tX, ty), (vm, vX, vy), (sm, sX, sy) = load("train"), load("val"), load("test")
    data_id = {
        "train": digest(tm["case_id"].to_numpy(), ty),
        "val": digest(vm["case_id"].to_numpy(), vy),
        "test": digest(sm["case_id"].to_numpy(), sy),
    }
    rows, fingerprints = [], {}
    for name, cols in FEATURE_SETS.items():
        for spec in grid():
            fingerprints[(name, spec["id"])] = {
                "variant": name, "columns": cols, "spec": spec, "quantiles": QUANTILES,
                "data": data_id, "features": digest(tX[:, cols], vX[:, cols], sX[:, cols])}
            cache = OUT / f"{name}__{spec['id']}.npz"
            if not cache.exists():
                print(f"[fit] {name} {spec['id']}", flush=True)
                vq, sq = fit_predict(spec, tX[:, cols], ty, [vX[:, cols], sX[:, cols]])
                np.savez_compressed(cache, val=vq, test=sq,
                                    fingerprint=json.dumps(fingerprints[(name, spec["id"])]))
            vq, _ = load_predictions(name, spec["id"], fingerprints)
            rows.append({"variant": name, "config": spec["id"],
                         "validation_pinball": float(mean_pinball(vq, vy).mean())})
    trials = pd.DataFrame(rows)
    trials.to_csv(OUT / "validation_grid.csv", index=False)
    selected = trials.loc[trials.groupby("variant")["validation_pinball"].idxmin()].set_index("variant")

    v_index, v_blocks = moving_block_period_indices(vm, 30, 5000, seed=20260923)
    s_index, s_blocks = moving_block_period_indices(sm, 30, 5000, seed=29042)

    results = {}
    for mode in ("fixed", "retuned"):
        scores = {}
        for name in FEATURE_SETS:
            config = FIXED_ID if mode == "fixed" else selected.loc[name, "config"]
            vq, sq = load_predictions(name, config, fingerprints)
            scores[name] = {
                "config": config,
                "val_pinball": mean_pinball(vq, vy),
                "val_cov80": float(((vy >= vq[:, 0]) & (vy <= vq[:, 2])).mean()),
                "test_crps": induced_crps(sq, sy),
                "test_cov80": float(((sy >= sq[:, 0]) & (sy <= sq[:, 2])).mean()),
            }
        effects = []
        for name in FEATURE_SETS:
            if name == "full_h0":
                continue
            vd = scores[name]["val_pinball"] - scores["full_h0"]["val_pinball"]
            sd = scores[name]["test_crps"] - scores["full_h0"]["test_crps"]
            v_point, v_lo, v_hi, _ = paired(vd, vm, v_index, v_blocks)
            s_point, s_lo, s_hi, s_p = paired(sd, sm, s_index, s_blocks)
            effects.append({"mode": mode, "variant": name, "config": scores[name]["config"],
                            "val_pinball": scores[name]["val_pinball"].mean(), "val_diff": v_point,
                            "val_low": v_lo, "val_high": v_hi, "val_cov80": scores[name]["val_cov80"],
                            "test_crps": scores[name]["test_crps"].mean(), "test_diff": s_point,
                            "test_low": s_lo, "test_high": s_hi, "test_p": s_p,
                            "test_cov80": scores[name]["test_cov80"]})
        fam = [e for e in effects if e["variant"] in TABLE4_FAMILY]
        for e, adj in zip(fam, holm([e["test_p"] for e in fam])):
            e["test_p_holm_feature_family"] = float(adj)
        full = scores["full_h0"]
        effects.insert(0, {"mode": mode, "variant": "full_h0", "config": full["config"],
                           "val_pinball": full["val_pinball"].mean(), "val_cov80": full["val_cov80"],
                           "test_crps": full["test_crps"].mean(), "test_cov80": full["test_cov80"]})
        results[mode] = pd.DataFrame(effects)

    table = pd.concat(results.values(), ignore_index=True)
    table.to_csv(OUT / "ablation_effects.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    (OUT / "manifest.json").write_text(json.dumps({
        "grid": [s["id"] for s in grid()], "fixed_config": FIXED_ID,
        "selected": selected["config"].to_dict(),
        "validation_seed": 20260923, "test_seed": 29042, "block_length": 30, "replicates": 5000,
        "holm_family": TABLE4_FAMILY,
        "holm_note": "Separate family of the three feature comparisons; the look-ahead comparison "
                     "is taken from the separately tuned cadence-window experiment.",
        "validation_interval_note": "Conditional on the selected configurations; excludes re-tuning uncertainty.",
        "data_fingerprint": data_id}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
