"""Second-revision analyses: directly fitted interval endpoints, service-policy comparison, and fixed-physical-time cadence/window experiments.

This script implements the analyses added in the second revision of the
manuscript (see the paper, Section 5 and the supplementary note on post hoc
analyses). It reads only the frozen artifacts written by
``experiment_runner.py`` (``prepare`` through ``freeze``) and
``frozen_evaluation.py`` under ``outputs/paper_results/`` -- except for the
cadence/window stage, which re-reads the configured input tables through
``data_processing`` to rebuild features at other update intervals.

Usage (``tf_env``)::

    python r2_experiment_runner.py endpoints
    python r2_experiment_runner.py service
    python r2_experiment_runner.py sensitivity
    python r2_experiment_runner.py all

The frozen plan (``plan.json``) and the input manifest
(``input_manifest.json``) are written on the first run and must match on
later runs; a mismatch stops the run instead of silently reusing artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

# Load PyTorch first on Windows to avoid DLL load-order conflicts with the
# scientific libraries loaded below.
import torch
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import yaml

from probabilistic_metrics import gaussian_crps, interval_score, pinball_loss


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "outputs/paper_results"
OUT = ROOT / "outputs/paper_results_r2"
SEED = 20260923
METRICS = ["coverage", "width", "interval_score", "lower_miss", "upper_miss"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def log(event, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with (OUT / "attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(event, flush=True)


def freeze_plan():
    plan = {
        "endpoint_levels": [0.05, 0.95], "endpoint_selection": "Reuse each endpoint's original validation hyperparameters; sort the two endpoints",
        "service_constructor": "snapshot", "service_grid": "Reuse the 49-point validation grid and the three coverage floors",
        "uniform_scale": "Equal-period validation coverage reaches the gated validation combined coverage; also report the nominal 0.90 target",
        "service_inference": "Original 5000 test block indices; pointwise 95% intervals; no equivalence or non-inferiority claims",
        "random_fallback": "Fixed fallback-period count; 20000 draws; conditional random reference distribution, not a confidence interval",
        "sensitivity_settings": [[2, 30], [5, 30], [10, 30], [5, 10], [5, 15], [5, 60]],
        "sensitivity_selection": "Each setting and each of the two information sets selects independently on validation mean pinball from the same four-point grid",
        "sensitivity_grid": [
            {"max_depth": depth, "learning_rate": rate, "max_iter": 200, "l2_regularization": 1.0}
            for depth, rate in itertools.product([3, 5], [0.05, 0.1])
        ],
        "sensitivity_support": "Fixed original three-segment physical times; 30-minute look-ahead guard at the end of each segment; snapshot model",
        "sensitivity_bootstrap": "150-minute blocks; 5000 replicates; blocks never cross a gap in the current update cadence",
        "sensitivity_scores": "CRPS from the 0.1/0.5/0.9 induced Gaussian; also report the native 80% interval",
        "ablation": "Recompute the stored validation predictions; reuse the full-feature hyperparameters, retaining the selection-bias caveat",
        "seed": SEED,
    }
    path = OUT / "plan.json"
    if path.exists() and read_json(path) != plan:
        raise RuntimeError("The experiment plan is frozen and differs from the saved version.")
    write_json(path, plan)
    write_json(OUT / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "scipy": scipy.__version__, "threads": 4,
    })
    # Provenance anchor: the frozen run itself. The second-revision stages
    # reuse the frozen split, selections and bootstrap indices, so the
    # selection lock written by `experiment_runner.py freeze` (and required
    # by `frozen_evaluation.py` before any test scoring) is the check that
    # the inputs are the frozen ones. The private working run additionally
    # verified anonymized input copies against the frozen split before
    # running; that check is specific to the private data handling and has
    # no counterpart in this public release.
    lock = OLD / "freeze" / "selection_lock.json"
    if not lock.exists():
        raise RuntimeError(
            "The frozen run is required before the second-revision analyses: "
            "run experiment_runner.py through 'freeze' and frozen_evaluation.py first, "
            "so outputs/paper_results/freeze/selection_lock.json exists.")
    sources = [ROOT / "data_processing.py", ROOT / "probabilistic_metrics.py",
               ROOT / "experiment_config.yaml", OLD / "g0/split_manifest.json",
               OLD / "ap09/bootstrap_indices.npz", lock]
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    manifest_path = OUT / "input_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != provenance:
        raise RuntimeError("Input sources differ from the first frozen run; refusing to reuse run artifacts.")
    write_json(manifest_path, provenance)
    return plan


def load_split(name):
    return {
        "X": np.load(OLD / f"g0/{name}_X_raw.npy", mmap_mode="r"),
        "y": np.asarray(np.load(OLD / f"g0/{name}_y_raw.npy"), dtype=float).reshape(-1),
        "meta": pd.read_csv(OLD / f"g0/{name}_metadata.csv", parse_dates=["prediction_time"]),
    }


def pm(values, meta):
    return pd.DataFrame(np.asarray(values)).groupby(meta.update_period_id.to_numpy(), sort=True).mean().to_numpy()


def interval_values(y, lo, hi):
    y, lo, hi = np.broadcast_arrays(y, lo, hi)
    if not np.all(np.isfinite([y, lo, hi])) or np.any(lo > hi):
        raise ValueError("Invalid interval endpoints.")
    return np.column_stack([
        ((y >= lo) & (y <= hi)).astype(float), hi - lo,
        interval_score(y, lo, hi, alpha=0.1), (y < lo).astype(float), (y > hi).astype(float),
    ])


def ci(values, indices):
    values = np.asarray(values)
    boot = values[indices].mean(axis=1)
    return np.quantile(boot, [0.025, 0.975], axis=0)


def block_indices(meta, step_minutes, block_length, repeats=5000):
    periods = meta[["update_period_id", "prediction_time"]].drop_duplicates("update_period_id").sort_values("prediction_time").reset_index(drop=True)
    step = pd.Timedelta(minutes=step_minutes)
    breaks = np.flatnonzero(periods.prediction_time.diff().fillna(step) != step)
    starts, ends = np.r_[0, breaks], np.r_[breaks, len(periods)]
    valid = np.array([j for a, b in zip(starts, ends) for j in range(a, b - block_length + 1)])
    if not len(valid):
        raise ValueError("Not enough consecutive periods to form blocks of the requested physical length.")
    rng = np.random.default_rng(SEED)
    blocks = int(np.ceil(len(periods) / block_length))
    draws = rng.choice(valid, size=(repeats, blocks))
    indices = (draws[:, :, None] + np.arange(block_length)).reshape(repeats, -1)[:, :len(periods)]
    return periods, indices


def fit_models(X, y, spec, levels):
    models = []
    for q in levels:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_depth=spec["max_depth"],
            learning_rate=spec["learning_rate"], max_iter=spec["max_iter"],
            l2_regularization=spec["l2_regularization"], min_samples_leaf=20,
            max_bins=255, early_stopping=False, random_state=0,
        )
        model.fit(X, y)
        models.append(model)
    return models


def induced_interval(pred):
    lo = np.minimum(pred[:, 0], pred[:, 2])
    hi = np.maximum(pred[:, 0], pred[:, 2])
    mu = np.clip(pred[:, 1], lo, hi)
    sigma = np.maximum((hi - lo) / 2.563103, 0.001)
    return mu, sigma, mu - norm.ppf(0.95) * sigma, mu + norm.ppf(0.95) * sigma


def endpoint_stage():
    train, val = load_split("train"), load_split("val")
    folder = OUT / "endpoints"
    folder.mkdir(parents=True, exist_ok=True)
    selections = {}
    for kind, relative in [("snapshot", "ap03"), ("lagged", "ap02")]:
        selection_path = OLD / relative / f"{kind}_quantile_gbdt/selected_model.json"
        selection = read_json(selection_path)
        spec = selection["spec"]
        history = spec["history_length"]
        X = np.asarray(train["X"][:, -history:, :]).reshape(len(train["y"]), -1)
        path = folder / f"{kind}_models.joblib"
        selections[kind] = {"spec": spec, "selection_sha256": digest(selection_path)}
        write_json(folder / "frozen_endpoint_settings.json", selections)
        models = fit_models(X, train["y"], spec, [0.05, 0.95])
        joblib.dump(models, path)
        old_models = joblib.load(OLD / selection["models"])
        for name in ["val", "test"]:
            split = val if name == "val" else load_split(name)
            xx = np.asarray(split["X"][:, -history:, :]).reshape(len(split["y"]), -1)
            raw = np.column_stack([m.predict(xx) for m in models])
            direct = np.sort(raw, axis=1)
            original = np.column_stack([m.predict(xx) for m in old_models])
            _, _, old_lo, old_hi = induced_interval(original)
            np.savez_compressed(folder / f"{kind}_{name}.npz", lo=direct[:, 0], hi=direct[:, 1],
                                original_lo=old_lo, original_hi=old_hi, raw=raw,
                                case_id=split["meta"].case_id.to_numpy())
        log("Direct endpoints fitted", kind=kind)
    meta = load_split("test")["meta"]
    y = load_split("test")["y"]
    original_boot = np.load(OLD / "ap09/bootstrap_indices.npz")
    assert np.array_equal(np.sort(meta.update_period_id.unique()), original_boot["update_period_id"])
    rows = []
    for kind in selections:
        pred = np.load(folder / f"{kind}_test.npz")
        for constructor, lo, hi in [("direct", pred["lo"], pred["hi"]), ("gaussian", pred["original_lo"], pred["original_hi"])]:
            values = interval_values(y, lo, hi)
            periods = pm(values, meta)
            limits = ci(periods, original_boot["indices"])
            row = {"kind": kind, "constructor": constructor, "crossing_rate": float(np.mean(pred["raw"][:, 0] > pred["raw"][:, 1]))}
            for j, metric in enumerate(METRICS):
                row.update({metric + "_case": values[:, j].mean(), metric + "_period": periods[:, j].mean(),
                            metric + "_low": limits[0, j], metric + "_high": limits[1, j]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(folder / "test_metrics.csv", index=False)


def scale_for_coverage(y, lo, hi, meta, target):
    half = (hi - lo) / 2
    if np.any(half <= 0):
        raise ValueError("Uniform scaling requires positive interval widths.")
    ratios = np.abs(y - (lo + hi) / 2) / half
    counts = meta.groupby("update_period_id").case_id.transform("count").to_numpy()
    weights = 1.0 / counts / meta.update_period_id.nunique()
    order = np.argsort(ratios, kind="stable")
    index = min(np.searchsorted(np.cumsum(weights[order]), target), len(order) - 1)
    # Leave a rounding margin so an observation exactly on an endpoint counts as covered.
    return float(ratios[order[index]] * (1 + 1e-12))


def gate_mask(state, rule):
    mode = rule["mode"]
    result = np.ones(len(state), dtype=bool)
    if mode == "fallback_only":
        return ~result
    if mode in ["joint_gate", "R_comp_only", "label_age_R_comp"]:
        result &= state.R_comp.to_numpy() >= rule["R_low"]
    if mode in ["joint_gate", "rho_only"]:
        result &= state.rho_t.to_numpy() <= rule["rho_high"]
    if mode in ["label_age_only", "label_age_R_comp"]:
        result &= state.label_age_seconds.to_numpy() <= rule["label_age_high"]
    return result


def state_table(split, name):
    ids = np.sort(split["meta"].update_period_id.unique())
    x = pm(split["X"][:, -1, :5], split["meta"])
    state = pd.DataFrame({"update_period_id": ids, "R_comp": x[:, 2], "rho_t": x[:, 3] / 300})
    source = "test" if name == "test" else "development"
    delay = pd.read_csv(OLD / f"ap01/delay_per_period_{source}.csv")
    delay = delay[["update_period_id", "label_age_seconds"]].drop_duplicates("update_period_id")
    return state.merge(delay, on="update_period_id", how="left", validate="one_to_one")


def service_stage():
    cfg = yaml.safe_load((ROOT / "experiment_config.yaml").read_text())
    original = read_json(OLD / "ap06/selected_policies.json")
    fallback = original["fallback"]
    val = load_split("val")
    vs = state_table(val, "val")
    vpred = np.load(OUT / "endpoints/snapshot_val.npz")
    vf = pm(interval_values(val["y"], fallback["lo90"], fallback["hi90"]), val["meta"])
    frozen = {}
    folder = OUT / "service"
    folder.mkdir(parents=True, exist_ok=True)
    for constructor in ["direct", "gaussian"]:
        vlo, vhi = (vpred["lo"], vpred["hi"]) if constructor == "direct" else (vpred["original_lo"], vpred["original_hi"])
        vm = pm(interval_values(val["y"], vlo, vhi), val["meta"])
        grid = []
        for qR, qrho in itertools.product(cfg["ap06"]["q_R"], cfg["ap06"]["q_rho"]):
            rule = {"mode": "joint_gate", "q_R": qR, "q_rho": qrho,
                    "R_low": float(np.quantile(vs.R_comp, qR)), "rho_high": float(np.quantile(vs.rho_t, qrho))}
            mask = gate_mask(vs, rule)
            combined = np.where(mask[:, None], vm, vf)
            grid.append({**rule, "publication": mask.mean(), "broadcast_coverage": vm[mask, 0].mean(),
                         "score": combined[:, 2].mean(), "width": vm[mask, 1].mean(),
                         "eligible": bool(min(mask.sum(), (~mask).sum()) >= 60)})
        grid = pd.DataFrame(grid)
        grid.to_csv(folder / f"{constructor}_validation_grid.csv", index=False)
        rules = {k: v for k, v in read_json(OLD / "ap01/service_rules.json").items() if isinstance(v, dict)}
        for name, floor in [("reliability_first", 0.90), ("balanced", 0.87), ("information_first", 0.85)]:
            feasible = grid[grid.eligible & (grid.broadcast_coverage >= floor)]
            if len(feasible):
                winner = feasible.sort_values(["publication", "score", "width", "q_R", "q_rho"], ascending=[False, True, True, False, True]).iloc[0]
                rules[name] = {k: winner[k] for k in ["mode", "q_R", "q_rho", "R_low", "rho_high"]}
            else:
                rules[name] = {"mode": "fallback_only"}
        selected_mask = gate_mask(vs, rules["reliability_first"])
        target = float(np.where(selected_mask, vm[:, 0], vf[:, 0]).mean())
        scales = {"uniform_match_gate": scale_for_coverage(val["y"], vlo, vhi, val["meta"], target),
                  "uniform_nominal": scale_for_coverage(val["y"], vlo, vhi, val["meta"], 0.90)}
        frozen[constructor] = {"rules": rules, "scales": scales, "gate_validation_coverage": target,
                               "gate_validation_publication": float(selected_mask.mean())}
    write_json(folder / "frozen_validation_policies.json", frozen)
    log("Service rules frozen on validation results")
    test = load_split("test")
    state = state_table(test, "test")
    pred = np.load(OUT / "endpoints/snapshot_test.npz")
    bootfile = np.load(OLD / "ap09/bootstrap_indices.npz")
    assert np.array_equal(state.update_period_id, bootfile["update_period_id"])
    indices = bootfile["indices"]
    fm = pm(interval_values(test["y"], fallback["lo90"], fallback["hi90"]), test["meta"])
    rows, effects, selectivity, random_rows, per_period = [], [], [], [], []
    for constructor, frozen_one in frozen.items():
        lo, hi = (pred["lo"], pred["hi"]) if constructor == "direct" else (pred["original_lo"], pred["original_hi"])
        model = pm(interval_values(test["y"], lo, hi), test["meta"])
        outputs = {}
        for name, rule in frozen_one["rules"].items():
            mask = gate_mask(state, rule)
            outputs[name] = (np.where(mask[:, None], model, fm), mask)
        for name, scale in frozen_one["scales"].items():
            center, half = (lo + hi) / 2, (hi - lo) / 2
            outputs[name] = (pm(interval_values(test["y"], center - scale * half, center + scale * half), test["meta"]), np.ones(len(state), bool))
        for name, (values, mask) in outputs.items():
            limits = ci(values, indices)
            row = {"constructor": constructor, "rule": name, "publication": mask.mean(), "n_fallback": int((~mask).sum())}
            for j, metric in enumerate(METRICS):
                row.update({metric: values[:, j].mean(), metric + "_low": limits[0, j], metric + "_high": limits[1, j]})
            rows.append(row)
            frame = pd.DataFrame(values, columns=METRICS)
            frame["update_period_id"] = state.update_period_id
            frame["constructor"], frame["rule"], frame["broadcast"] = constructor, name, mask
            per_period.append(frame)
            comparison = "reliability_first" if name.startswith("uniform") else "no_gate"
            if name == comparison:
                continue
            difference = values - outputs[comparison][0]
            limits = ci(difference, indices)
            for j, metric in enumerate(METRICS):
                effects.append({"constructor": constructor, "rule": name, "reference": comparison,
                                "metric": metric, "difference": difference[:, j].mean(), "low": limits[0, j], "high": limits[1, j]})
        selected = outputs["reliability_first"][1]
        if 0 < selected.sum() < len(selected):
            sampled = selected[indices]
            sampled_values = model[indices]
            retained = np.sum(sampled_values * sampled[:, :, None], axis=1) / sampled.sum(axis=1)[:, None]
            rejected = np.sum(sampled_values * (~sampled)[:, :, None], axis=1) / (~sampled).sum(axis=1)[:, None]
            limits = np.nanquantile(rejected - retained, [0.025, 0.975], axis=0)
            for j, metric in enumerate(METRICS):
                selectivity.append({"constructor": constructor, "metric": metric,
                                    "retained": model[selected, j].mean(), "fallback_periods": model[~selected, j].mean(),
                                    "difference": model[~selected, j].mean() - model[selected, j].mean(),
                                    "low": limits[0, j], "high": limits[1, j]})
        rng = np.random.default_rng(SEED)
        random_values = np.empty((20000, len(METRICS)))
        for i in range(len(random_values)):
            mask = np.ones(len(state), bool)
            mask[rng.choice(len(state), (~selected).sum(), replace=False)] = False
            random_values[i] = np.where(mask[:, None], model, fm).mean(axis=0)
        for j, metric in enumerate(METRICS):
            q = np.quantile(random_values[:, j], [0.025, 0.975])
            actual = outputs["reliability_first"][0][:, j].mean()
            random_rows.append({"constructor": constructor, "metric": metric, "mean": random_values[:, j].mean(),
                                "reference_low": q[0], "reference_high": q[1], "selected": actual,
                                "fraction_random_le_selected": np.mean(random_values[:, j] <= actual)})
    for name, frame in [("test_summary", pd.DataFrame(rows)), ("paired_effects", pd.DataFrame(effects)),
                        ("selectivity", pd.DataFrame(selectivity)), ("random_reference", pd.DataFrame(random_rows)),
                        ("per_period", pd.concat(per_period, ignore_index=True))]:
        frame.to_csv(folder / f"{name}.csv", index=False)
    log("Service comparison of direct and Gaussian endpoints complete")


def sensitivity_stage(plan):
    from data_processing import _resolve_data_path, compute_rolling_features, create_trajectory_indexed_dataset, load_matched_data
    config_path = ROOT / "study_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Only read input tables inside the configured data directory, using the
    # same path resolution as the main pipeline.
    data_dir = (ROOT / "data").resolve()
    for key in ["plate1_path", "plate2_path"]:
        if _resolve_data_path(config["data"][key]).resolve().parent != data_dir:
            raise RuntimeError("Cadence/window experiments only read input tables inside data/.")
    upstream, _, matched = load_matched_data(config)
    source_manifest = read_json(OLD / "g0/split_manifest.json")
    defaults = source_manifest["training_defaults"]
    original_meta = {name: load_split(name)["meta"] for name in ["train", "val", "test"]}
    starts = {name: frame.prediction_time.min() for name, frame in original_meta.items()}
    ends = {"train": starts["val"], "val": starts["test"], "test": original_meta["test"].prediction_time.max() + pd.Timedelta(minutes=5)}
    write_json(OUT / "sensitivity/physical_boundaries.json", {"starts": starts, "ends_exclusive": ends, "guard_minutes": 30})
    summaries, paired = [], []
    for delta, window in plan["sensitivity_settings"]:
        folder = OUT / f"sensitivity/d{delta}_w{window}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "completed.json").exists():
            summaries.extend(read_json(folder / "completed.json")["summary"])
            paired.extend(read_json(folder / "completed.json")["paired"])
            continue
        log("Starting cadence/window setting", delta=delta, window=window)
        samples = {}
        grids = []
        for variant, horizon in [("deployable", 0), ("lookahead", 1800)]:
            features = compute_rolling_features(upstream, matched, delta, window,
                        defaults["travel_time_median"], defaults["travel_time_variance"], horizon)
            X, y, meta = create_trajectory_indexed_dataset(matched, features, delta, 1, return_metadata=True)
            X, y = X.numpy()[:, -1, :], y.numpy().astype(float)
            data = {}
            for name in starts:
                mask = (meta.prediction_time >= starts[name]) & (meta.prediction_time < ends[name] - pd.Timedelta(minutes=30))
                data[name] = {"X": X[mask], "y": y[mask], "meta": meta.loc[mask].reset_index(drop=True)}
            samples[variant] = data
            trials = []
            best = None
            for index, spec in enumerate(plan["sensitivity_grid"]):
                trial = folder / f"{variant}_trial{index}.json"
                model_path = folder / f"{variant}_trial{index}.joblib"
                if trial.exists() and model_path.exists():
                    record = read_json(trial)
                    if record["spec"] != spec:
                        raise RuntimeError("A saved trial disagrees with the frozen parameters.")
                else:
                    models = fit_models(data["train"]["X"], data["train"]["y"], spec, [0.1, 0.5, 0.9])
                    valpred = np.column_stack([m.predict(data["val"]["X"]) for m in models])
                    loss = np.column_stack([pinball_loss(data["val"]["y"], valpred[:, j], q) for j, q in enumerate([0.1, 0.5, 0.9])]).mean()
                    record = {"spec": spec, "variant": variant, "trial": index, "validation_pinball": float(loss)}
                    joblib.dump(models, model_path)
                    write_json(trial, record)
                trials.append(record)
                if best is None or (record["validation_pinball"], index) < (best["validation_pinball"], best["trial"]):
                    best = record
            grids.extend(trials)
            write_json(folder / f"{variant}_selected.json", best)
        pd.DataFrame(grids).to_csv(folder / "validation_grid.csv", index=False)
        log("Cadence/window validation selection complete", delta=delta, window=window)
        full_rows, paired_rows, results = [], [], {}
        for variant, data in samples.items():
            selected = read_json(folder / f"{variant}_selected.json")
            models = joblib.load(folder / f"{variant}_trial{selected['trial']}.joblib")
            test = data["test"]
            pred = np.column_stack([m.predict(test["X"]) for m in models])
            mu, sigma, _, _ = induced_interval(pred)
            lo, hi = np.minimum(pred[:, 0], pred[:, 2]), np.maximum(pred[:, 0], pred[:, 2])
            scores = gaussian_crps(mu, sigma, test["y"])
            values = np.column_stack([scores, ((test["y"] >= lo) & (test["y"] <= hi)), hi - lo])
            period_values = pm(values, test["meta"])
            periods, indices = block_indices(test["meta"], delta, 150 // delta)
            assert np.array_equal(periods.update_period_id, np.sort(test["meta"].update_period_id.unique()))
            counts = test["meta"].groupby("update_period_id").size().to_numpy()
            boot_case = (period_values[indices] * counts[indices, None]).sum(axis=1) / counts[indices].sum(axis=1)[:, None]
            limits = np.quantile(boot_case, [0.025, 0.975], axis=0)
            row = {"delta_minutes": delta, "window_minutes": window, "variant": variant,
                   "n_test_cases": len(test["y"]), "n_test_periods": len(periods),
                   "n_train_cases": len(data["train"]["y"]), "n_validation_cases": len(data["val"]["y"]),
                   "validation_pinball": selected["validation_pinball"], "block_periods": 150 // delta}
            for j, metric in enumerate(["crps", "coverage80", "width80"]):
                row.update({metric: values[:, j].mean(), metric + "_low": limits[0, j], metric + "_high": limits[1, j]})
            full_rows.append(row)
            np.savez_compressed(folder / f"{variant}_test.npz", predictions=pred, scores=scores,
                                case_id=test["meta"].case_id, period_values=period_values)
            results[variant] = (values, period_values, test, indices, counts)
        va, pa, ta, indices, counts = results["deployable"]
        vb, pb, tb, _, _ = results["lookahead"]
        assert ta["meta"].equals(tb["meta"]) and np.array_equal(ta["y"], tb["y"])
        diff = pb - pa
        boot = (diff[indices] * counts[indices, None]).sum(axis=1) / counts[indices].sum(axis=1)[:, None]
        limits = np.quantile(boot, [0.025, 0.975], axis=0)
        for j, metric in enumerate(["crps", "coverage80", "width80"]):
            paired_rows.append({"delta_minutes": delta, "window_minutes": window, "metric": metric,
                                "lookahead_minus_deployable": (vb - va)[:, j].mean(), "low": limits[0, j], "high": limits[1, j]})
        test = ta
        state = test["meta"].copy()
        state["N_complete"] = test["X"][:, 1]
        periods = state.groupby("update_period_id", sort=True).first().reset_index()
        up = matched.timestamp_up.to_numpy(dtype="datetime64[ns]")
        down = matched.timestamp_down.to_numpy(dtype="datetime64[ns]")
        diagnostic = []
        for row in periods.itertuples():
            t = np.datetime64(row.prediction_time, "ns")
            left, right = np.searchsorted(up, [t - np.timedelta64(window, "m"), t], side="right")
            complete = down[left:right] <= t
            age = float((t - up[left:right][complete].max()) / np.timedelta64(1, "s")) if complete.any() else np.nan
            diagnostic.append({"update_period_id": row.update_period_id, "inflight_share": 1 - row.N_complete / max(right-left, 1), "label_age_seconds": age})
        diagnostics = pd.DataFrame(diagnostic)
        diagnostics.to_csv(folder / "delay_per_period.csv", index=False)
        np.savez_compressed(folder / "bootstrap_indices.npz", update_period_id=periods.update_period_id, indices=indices)
        for row in full_rows:
            row["mean_inflight_share"] = diagnostics.inflight_share.mean()
            row["median_label_age_seconds"] = diagnostics.label_age_seconds.median()
        write_json(folder / "completed.json", {"summary": full_rows, "paired": paired_rows})
        summaries.extend(full_rows)
        paired.extend(paired_rows)
        log("Cadence/window setting complete", delta=delta, window=window)
    pd.DataFrame(summaries).to_csv(OUT / "sensitivity/summary.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "sensitivity/paired_effects.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["endpoints", "service", "sensitivity", "all"])
    args = parser.parse_args()
    plan = freeze_plan()
    log("Run started", stage=args.stage, source_sha256=digest(__file__), plan_sha256=digest(OUT / "plan.json"))
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        if args.stage in ["endpoints", "all"]:
            endpoint_stage()
        if args.stage in ["service", "all"]:
            service_stage()
        if args.stage in ["sensitivity", "all"]:
            sensitivity_stage(plan)
    log("Run complete", stage=args.stage, elapsed_seconds=time.perf_counter() - started)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("Run failed", exception_type=type(error).__name__)
        raise
