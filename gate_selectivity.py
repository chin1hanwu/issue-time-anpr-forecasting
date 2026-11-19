"""R4-2 diagnostic: does the selected gate identify unreliable periods, or only widen?

Pure re-aggregation of stored predictions. No model is refitted.
Period weighting follows the manuscript: average within period, equal weight across periods.

Inputs
  per-case test service file (the supplementary service file written by
  frozen_evaluation.py, ap10/service_kpi_per_case.csv), and the local run
  directory outputs/paper_results, which supplies
    ap09/bootstrap_indices.npz          original gap-aware test moving-block indices
    ap03/validation_predictions.npz     snapshot quantile-induced Gaussian, validation segment
    g0/val_metadata.csv, g0/val_y_raw.npy, g0/val_X_raw.npy

Usage:
    python gate_selectivity.py outputs/paper_results/ap10/service_kpi_per_case.csv [outputs/paper_results]

Without the run directory, the test bootstrap indices are regenerated with the manuscript's
gap-aware moving-block function (same block length and replicate count, different random draws),
and the validation-based sections 4 and 5 are skipped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ALPHA = 0.10
FALLBACK_MU, FALLBACK_SD = 81.782, 35.465
R_LOW, RHO_HIGH = 0.5749265968799591, 0.37360023558139815
DELTA_OBS = 300.0

# Validation-derived constants of the published study, needed to reproduce
# this diagnostic on the published supplementary file: the fallback mean and
# standard deviation are the training-marginal moments, R_LOW / RHO_HIGH are
# the validation quantiles of the selected gate, and DELTA_OBS is the
# 5-minute update interval in seconds. They are study results, not raw data.

case_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
HAS_RUN = run_dir is not None and (run_dir / "ap09" / "bootstrap_indices.npz").exists()


def interval_score(lo, hi, obs):
    return (hi - lo) + (2 / ALPHA) * (lo - obs) * (obs < lo) + (2 / ALPHA) * (obs - hi) * (obs > hi)


def period_mean(values, period_ids, order):
    s = pd.Series(values).groupby(np.asarray(period_ids)).mean()
    return s.reindex(order).to_numpy()


def ci(boot):
    return np.nanpercentile(boot, [2.5, 97.5])


# ---------------------------------------------------------------- test data
d = pd.read_csv(case_path)
if HAS_RUN:
    boot_file = np.load(run_dir / "ap09" / "bootstrap_indices.npz")
    periods = boot_file["update_period_id"]      # chronological period order used by the paper
    BOOT = boot_file["indices"]                   # (5000, 402), blocks never cross time gaps
    boot_source = "original gap-aware indices"
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probabilistic_metrics import moving_block_period_indices
    table, BOOT = moving_block_period_indices(d, 30, 5000, 20260923)
    periods = table["update_period_id"].to_numpy()
    boot_source = "regenerated gap-aware indices"
P = len(periods)
assert set(periods) == set(d["update_period_id"])

y = d["y_seconds"].to_numpy()
ml, mu = d["component_lo90"].to_numpy(), d["component_hi90"].to_numpy()
fl, fu = d["fallback_lo90"].to_numpy(), d["fallback_hi90"].to_numpy()
pid = d["update_period_id"].to_numpy()


def pm(v):
    return period_mean(v, pid, periods)


m_cov, m_is, m_w = pm(((y >= ml) & (y <= mu)).astype(float)), pm(interval_score(ml, mu, y)), pm(mu - ml)
f_cov, f_is, f_w = pm(((y >= fl) & (y <= fu)).astype(float)), pm(interval_score(fl, fu, y)), pm(fu - fl)
m_lo, m_hi = pm((y < ml).astype(float)), pm((y > mu).astype(float))

RULES = ["no_gate", "reliability_first", "anchor", "rho_only", "R_comp_only", "label_age_only", "label_age_R_comp"]
pub = {r: pm(d["broadcast_" + r].to_numpy(float)) > 0.5 for r in RULES}


def combined(rule):
    b = pub[rule]
    return np.where(b, m_cov, f_cov), np.where(b, m_is, f_is), np.where(b, m_w, f_w)


print("test periods %d, bootstrap replicates %d (%s)\n" % (P, len(BOOT), boot_source))

print("=== 1. Service rules: period-weighted coverage, width, interval score with 95% block intervals ===")
_, is0, _ = combined("no_gate")
for r in RULES:
    c, s, w = combined(r)
    bs = np.array([s[i].mean() for i in BOOT])
    bd = np.array([(s - is0)[i].mean() for i in BOOT])
    print("  %-18s pub %.3f  cov %.3f  width %7.3f  IS %7.3f [%7.3f,%7.3f]  dIS vs no-gate %+7.3f [%+7.3f,%+7.3f]" % (
        r, pub[r].mean(), c.mean(), w.mean(), s.mean(), *ci(bs), (s - is0).mean(), *ci(bd)))

print("\n=== 2. Selectivity of the selected gate: MODEL interval in fallback vs retained periods ===")
g = ~pub["reliability_first"]
for name, v in (("coverage", m_cov), ("interval score", m_is), ("width", m_w), ("upper miss", m_hi), ("lower miss", m_lo)):
    boot = []
    for i in BOOT:
        gi, vi = g[i], v[i]
        boot.append(vi[gi].mean() - vi[~gi].mean() if 0 < gi.sum() < len(gi) else np.nan)
    print("  %-15s retained %8.4f  fallback %8.4f  diff %+8.4f  95%% [%+8.4f, %+8.4f]" % (
        name, v[~g].mean(), v[g].mean(), v[g].mean() - v[~g].mean(), *ci(np.array(boot))))

print("\n=== 3. Matched-budget random gate (same number of fallback periods) ===")
rng = np.random.default_rng(20260923)
k, R = int(g.sum()), 20000
rc, ri = np.empty(R), np.empty(R)
for b in range(R):
    fb = np.zeros(P, bool)
    fb[rng.choice(P, k, replace=False)] = True
    rc[b], ri[b] = np.where(fb, f_cov, m_cov).mean(), np.where(fb, f_is, m_is).mean()
gc, gs, _ = combined("reliability_first")
print("  random cov %.3f [%.3f, %.3f]  IS %.3f [%.3f, %.3f]" % (rc.mean(), *ci(rc), ri.mean(), *ci(ri)))
print("  selected gate cov %.3f IS %.3f; share of random draws with IS <= gate %.4f, with cov >= gate %.4f" % (
    gc.mean(), gs.mean(), (ri <= gs.mean()).mean(), (rc >= gc.mean()).mean()))

if not HAS_RUN:
    print("\nRun directory not supplied: validation sections 4 and 5 skipped.")
    sys.exit(0)

# ---------------------------------------------------------------- validation data
vz = np.load(run_dir / "ap03" / "validation_predictions.npz")
vmeta = pd.read_csv(run_dir / "g0" / "val_metadata.csv")
vy = np.load(run_dir / "g0" / "val_y_raw.npy").reshape(-1)
vX = np.load(run_dir / "g0" / "val_X_raw.npy")[:, -1, :]
vlo, vhi = vz["raw_lo90"].astype(float), vz["raw_hi90"].astype(float)
vpid = vmeta["update_period_id"].to_numpy()
vperiods = np.sort(np.unique(vpid))
assert len(vy) == len(vlo) == len(vmeta)


def vpm(v):
    return period_mean(v, vpid, vperiods)


vR, vrho = vpm(vX[:, 2]), vpm(vX[:, 3]) / DELTA_OBS
vpub = (vR >= R_LOW) & (vrho <= RHO_HIGH)
z = norm.ppf(0.95)
vfl, vfu = FALLBACK_MU - z * FALLBACK_SD, FALLBACK_MU + z * FALLBACK_SD
vm_cov = vpm(((vy >= vlo) & (vy <= vhi)).astype(float))
vf_cov = vpm(((vy >= vfl) & (vy <= vfu)).astype(float))
v_gate_cov = np.where(vpub, vm_cov, vf_cov).mean()
print("\n=== 4. Validation reproduction of the selected gate ===")
print("  validation periods %d, publication %.4f (manifest 0.8000), broadcast coverage %.4f (manifest 0.9153)" % (
    len(vperiods), vpub.mean(), vm_cov[vpub].mean()))
print("  validation combined coverage of the selected gate %.4f; no-gate validation coverage %.4f" % (
    v_gate_cov, vm_cov.mean()))

print("\n=== 5. Uniform widening, scale chosen on VALIDATION, evaluated on test ===")
vc, vh = (vlo + vhi) / 2, (vhi - vlo) / 2
tc, th = (ml + mu) / 2, (mu - ml) / 2


def v_cov_at(s):
    return vpm(((vy >= vc - s * vh) & (vy <= vc + s * vh)).astype(float)).mean()


def scale_for(target):
    lo, hi = 0.5, 4.0
    for _ in range(60):
        s = (lo + hi) / 2
        lo, hi = (s, hi) if v_cov_at(s) < target else (lo, s)
    return hi


for label, target in (("match gate validation coverage", v_gate_cov), ("nominal 0.90", 0.90)):
    s = scale_for(target)
    lo_t, hi_t = tc - s * th, tc + s * th
    w_cov, w_is, w_w = pm(((y >= lo_t) & (y <= hi_t)).astype(float)), pm(interval_score(lo_t, hi_t, y)), pm(hi_t - lo_t)
    dd = w_is - gs
    bd = np.array([dd[i].mean() for i in BOOT])
    bc = np.array([(w_cov - gc)[i].mean() for i in BOOT])
    print("  %-31s scale %.3f  test cov %.3f  width %7.3f  IS %7.3f | widen - gate: dIS %+7.3f [%+7.3f,%+7.3f]  dcov %+.4f [%+.4f,%+.4f]" % (
        label, s, w_cov.mean(), w_w.mean(), w_is.mean(), dd.mean(), *ci(bd), (w_cov - gc).mean(), *ci(bc)))

print("\n=== 6. Tail misses of the Gaussian-extrapolated 90% interval, no gate (nominal 0.05 each) ===")
bl = np.array([m_lo[i].mean() for i in BOOT])
bh = np.array([m_hi[i].mean() for i in BOOT])
print("  test lower %.4f [%.4f, %.4f]  upper %.4f [%.4f, %.4f]" % (m_lo.mean(), *ci(bl), m_hi.mean(), *ci(bh)))
print("  validation lower %.4f  upper %.4f" % (vpm((vy < vlo).astype(float)).mean(), vpm((vy > vhi).astype(float)).mean()))
