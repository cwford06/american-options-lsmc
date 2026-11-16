"""Run every experiment, write CSVs to results/ and charts to figures/.

    python -m amerop.experiments            # full run (about 10 minutes)
    python -m amerop.experiments --quick    # small smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .basis import FAMILIES, design
from .binomial import bbsr, bermudan_tree, crr
from .longstaff_schwartz import LSMCConfig, evaluate_policy, exercise_boundary, lsmc
from .models import OptionSpec, bs_price, simulate_gbm
from .rl import fqi, lspi

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
BASE = OptionSpec()                          # S0=36, K=40, r=6%, sigma=20%, T=1 put (Longstaff-Schwartz)
ATM = OptionSpec(S0=40.0)
OOS_SEED = 10_000


def dates(spec, per_year=50):
    return max(int(round(per_year * spec.T)), 5)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 1. benchmark table
def benchmark_table(q):
    rows = []
    for S0 in (36, 38, 40, 42, 44):
        for sigma in (0.2, 0.4):
            for T in (1, 2):
                spec = BASE.with_(S0=S0, sigma=sigma, T=T)
                M = dates(spec)
                amer, eur = bbsr(spec, 2000), bs_price(spec)
                berm = bermudan_tree(spec, M)
                oos = simulate_gbm(spec, q["oos"], M, np.random.default_rng(OOS_SEED))
                r = lsmc(spec, LSMCConfig(n_paths=q["paths"], n_steps=M, seed=0))
                v_oos = evaluate_policy(spec, r.continuation, oos)[0]
                rl = lspi(spec, simulate_gbm(spec, q["paths"], M, np.random.default_rng(1)))
                rows.append(dict(S0=S0, sigma=sigma, T=T, european=eur, american=amer, bermudan=berm,
                                 premium=amer - eur, lsmc=r.price, lsmc_se=r.stderr, lsmc_oos=v_oos,
                                 lspi_oos=evaluate_policy(spec, rl.continuation, oos)[0],
                                 lsmc_err_vs_berm=r.price - berm, lsmc_err_vs_amer=r.price - amer))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "benchmark_table.csv", index=False)
    return df


# ---------------------------------------------------------------- 2. convergence
def tree_convergence(q):
    bench = bbsr(BASE, 4000)
    ns = list(range(10, q["tree_max"] + 1))
    df = pd.DataFrame({"n": ns, "crr": [crr(BASE, n).price for n in ns],
                       "bbs": [crr(BASE, n, smooth=True).price for n in ns],
                       "bbsr": [bbsr(BASE, n) for n in ns]})
    df.to_csv(RES / "tree_convergence.csv", index=False)
    plots.tree_convergence(df, bench, FIG / "tree_convergence.png")
    return df, bench


def lsmc_convergence(q, berm):
    oos = simulate_gbm(BASE, q["oos"], 50, np.random.default_rng(OOS_SEED))
    rows = []
    for n in q["path_grid"]:
        ins, outs = [], []
        for seed in range(q["seeds"]):
            r = lsmc(BASE, LSMCConfig(n_paths=n, seed=seed))
            ins.append(r.price)
            outs.append(evaluate_policy(BASE, r.continuation, oos)[0])
        ins, outs = np.array(ins), np.array(outs)
        rows.append(dict(n_paths=n, mean_in=ins.mean(), std_in=ins.std(ddof=1), bias_in=ins.mean() - berm,
                         rmse_in=np.sqrt(np.mean((ins - berm) ** 2)), mean_oos=outs.mean(),
                         bias_oos=outs.mean() - berm, rmse_oos=np.sqrt(np.mean((outs - berm) ** 2))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "lsmc_convergence.csv", index=False)
    plots.lsmc_convergence(df, FIG / "lsmc_convergence.png")
    return df


def exercise_date_study(q, amer):
    rows = []
    for M in q["date_grid"]:
        k = max(20, 5000 // M)
        prices = [lsmc(BASE, LSMCConfig(n_paths=q["paths"] // 2, n_steps=M, seed=s)).price for s in range(q["seeds_small"])]
        rows.append(dict(n_dates=M, bermudan_tree=bermudan_tree(BASE, M, k), lsmc_mean=np.mean(prices),
                         lsmc_std=np.std(prices, ddof=1)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "exercise_dates.csv", index=False)
    plots.exercise_dates(df, amer, FIG / "exercise_dates.png")
    return df


# ---------------------------------------------------------------- 3. basis / regression study
def basis_study(q, berm):
    specs = {"OLS, in-the-money paths": dict(regressor="ols", itm_only=True),
             "OLS, all paths": dict(regressor="ols", itm_only=False),
             "Ridge, in-the-money paths": dict(regressor="ridge", itm_only=True, alpha=1e-4)}
    oos = simulate_gbm(BASE, q["oos"] // 2, 50, np.random.default_rng(OOS_SEED))
    train = [simulate_gbm(BASE, q["basis_paths"], 50, np.random.default_rng(s)) for s in range(q["seeds_small"] * 2)]
    rows = []
    for sp_name, sp in specs.items():
        for fam in FAMILIES:
            for deg in range(1, 7):
                ins, outs = [], []
                for paths in train:
                    r = lsmc(BASE, LSMCConfig(basis=fam, degree=deg, **sp), paths=paths)
                    ins.append(r.price)
                    outs.append(evaluate_policy(BASE, r.continuation, oos)[0])
                ins, outs = np.array(ins), np.array(outs)
                rows.append(dict(spec=sp_name, family=fam, degree=deg, mean_in=ins.mean(), std_in=ins.std(ddof=1),
                                 bias_in=ins.mean() - berm, rmse_in=np.sqrt(np.mean((ins - berm) ** 2)),
                                 mean_oos=outs.mean(), bias_oos=outs.mean() - berm))
    df = pd.DataFrame(rows)
    # conditioning of each design matrix on in-the-money paths at mid-life
    x = train[0][:, 25] / BASE.K
    x = x[x < 1]
    df["cond"] = [np.linalg.cond(design(x, f, d)[0]) for f, d in zip(df["family"], df["degree"])]
    df.to_csv(RES / "basis_study.csv", index=False)
    plots.basis_study(df, FIG / "basis_study.png")
    plots.basis_oos(df, FIG / "basis_in_vs_oos.png")
    return df


# ---------------------------------------------------------------- 4. variance reduction
def variance_reduction(q, berm):
    methods = {"Plain Monte Carlo": dict(antithetic=False),
               "Moment matching only": dict(antithetic=False, moment_match=True),
               "European control variate only": dict(antithetic=False, control_variate=True),
               "Antithetic paths only": dict(antithetic=True),
               "Antithetic + moment matching": dict(antithetic=True, moment_match=True),
               "Antithetic + control variate": dict(antithetic=True, control_variate=True),
               "All three": dict(antithetic=True, moment_match=True, control_variate=True)}
    rows = []
    for name, kw in methods.items():
        t0 = time.perf_counter()
        runs = [lsmc(BASE, LSMCConfig(n_paths=10_000, seed=s, **kw)) for s in range(q["seeds_vr"])]
        p = [r.price for r in runs]
        rows.append(dict(method=name, mean=np.mean(p), std=np.std(p, ddof=1), bias=np.mean(p) - berm,
                         within_run_se=np.mean([r.stderr for r in runs]),
                         seconds=(time.perf_counter() - t0) / len(p)))
    df = pd.DataFrame(rows)
    df["variance_reduction"] = (df["std"].iloc[0] / df["std"]) ** 2
    df["sampling_variance_reduction"] = (df["within_run_se"].iloc[0] / df["within_run_se"]) ** 2
    df.to_csv(RES / "variance_reduction.csv", index=False)
    plots.variance_reduction(df, FIG / "variance_reduction.png")
    return df


# ---------------------------------------------------------------- 5. ML / RL comparison
def ml_comparison(q):
    cases = {"S0=36, σ=20%, T=1": BASE, "S0=40, σ=20%, T=1": ATM, "S0=44, σ=40%, T=2": BASE.with_(S0=44, sigma=0.4, T=2)}
    rows = []
    for cname, spec in cases.items():
        M = dates(spec)
        berm = bermudan_tree(spec, M)
        oos = simulate_gbm(spec, q["oos"], M, np.random.default_rng(OOS_SEED))
        for seed in range(q["seeds_ml"]):
            paths = simulate_gbm(spec, q["ml_paths"], M, np.random.default_rng(seed))
            fits = {
                "OLS · Laguerre 3": lambda: lsmc(spec, LSMCConfig(n_steps=M, seed=seed), paths=paths),
                "Ridge · power 6": lambda: lsmc(spec, LSMCConfig(n_steps=M, basis="power", degree=6, regressor="ridge", seed=seed), paths=paths),
                "Neural net (MLP)": lambda: lsmc(spec, LSMCConfig(n_steps=M, regressor="mlp", seed=seed), paths=paths),
                "Gradient boosting": lambda: lsmc(spec, LSMCConfig(n_steps=M, regressor="hgb", seed=seed), paths=paths),
                "RL · LSPI": lambda: lspi(spec, paths),
                "RL · FQI": lambda: fqi(spec, paths),
            }
            for mname, fit in fits.items():
                t0 = time.perf_counter()
                r = fit()
                secs = time.perf_counter() - t0
                ins = getattr(r, "price", None)
                ins = r.in_sample if ins is None else ins
                v = evaluate_policy(spec, r.continuation, oos)[0]
                rows.append(dict(case=cname, method=mname, seed=seed, bermudan=berm, in_sample=ins, oos=v,
                                 err_in=ins - berm, err_oos=v - berm, seconds=secs))
        log(f"    {cname} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(RES / "ml_comparison_runs.csv", index=False)
    df = raw.groupby(["case", "method"], sort=False).agg(bermudan=("bermudan", "first"), in_sample=("in_sample", "mean"),
                                                          oos=("oos", "mean"), err_in=("err_in", "mean"),
                                                          err_oos=("err_oos", "mean"), oos_sd=("oos", "std"),
                                                          seconds=("seconds", "mean")).reset_index()
    df.to_csv(RES / "ml_comparison.csv", index=False)
    plots.ml_comparison(df, FIG / "ml_comparison.png")
    return df


# ---------------------------------------------------------------- 6. exercise boundaries
def boundary_study(q):
    M = 50
    tree = crr(BASE, 2000, boundary=True)
    paths = simulate_gbm(BASE, q["paths"], M, np.random.default_rng(0))
    curves = {"Binomial tree (2,000 steps)": (tree.boundary_t, tree.boundary_S)}
    r = lsmc(BASE, LSMCConfig(n_paths=q["paths"]), paths=paths)
    curves["LSMC · OLS Laguerre 3"] = exercise_boundary(BASE, r.continuation, M)
    rm = lsmc(BASE, LSMCConfig(regressor="mlp"), paths=paths[: q["ml_paths"]])
    curves["LSMC · neural net"] = exercise_boundary(BASE, rm.continuation, M)
    rl = lspi(BASE, paths)
    curves["RL · LSPI"] = exercise_boundary(BASE, rl.continuation, M)
    out = []
    for name, (t, s) in curves.items():
        out += [dict(method=name, t=a, S_star=b) for a, b in zip(t, s)]
    pd.DataFrame(out).to_csv(RES / "exercise_boundaries.csv", index=False)
    plots.boundaries(curves, BASE, FIG / "exercise_boundary.png")

    # sample paths with the LSMC policy's exercise points
    show = paths[:40]
    alive = np.ones(len(show), bool)
    ex_t, ex_S = [], []
    for t in range(1, M):
        g = BASE.payoff(show[:, t])
        cand = np.flatnonzero(alive & (g > 0))
        if len(cand):
            stop = cand[g[cand] > r.continuation(t, show[cand, t] / BASE.K)]
            ex_t += [t / M] * len(stop)
            ex_S += list(show[stop, t])
            alive[stop] = False
    tt = np.linspace(0, BASE.T, M + 1)
    plots.sample_paths(show, tt, ex_t, ex_S, tree.boundary_t, tree.boundary_S, BASE.K, FIG / "sample_paths.png")
    return curves


# ---------------------------------------------------------------- 7. sensitivity & scenarios
def sensitivity(q):
    grids = {"sigma": np.round(np.linspace(0.05, 0.6, 12), 3), "r": np.round(np.linspace(0.0, 0.12, 13), 3),
             "T": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], "K": list(range(30, 51, 2))}
    sweeps = {}
    for k, vals in grids.items():
        rows = []
        for v in vals:
            spec = ATM.with_(**{k: float(v)})
            M = dates(spec)
            r = lsmc(spec, LSMCConfig(n_paths=q["paths"] // 2, n_steps=M, seed=0))
            amer, eur = bbsr(spec, 1000), bs_price(spec)
            rows.append({k: float(v), "american": amer, "european": eur, "premium": amer - eur,
                         "lsmc": r.price, "lsmc_se": r.stderr})
        sweeps[k] = pd.DataFrame(rows)
        sweeps[k].to_csv(RES / f"sensitivity_{k}.csv", index=False)
    plots.sensitivity(sweeps, FIG / "sensitivity.png")

    # exercise boundary under scenarios, against time to maturity
    def bnd(spec):
        tr = crr(spec, 1500, boundary=True)
        return spec.T - tr.boundary_t, tr.boundary_S
    panels = {"Volatility": {f"σ = {s:.0%}": bnd(ATM.with_(sigma=s)) for s in (0.1, 0.2, 0.3, 0.4)},
              "Interest rate": {f"r = {r:.0%}": bnd(ATM.with_(r=r)) for r in (0.02, 0.06, 0.10)},
              "Maturity": {f"T = {T:g}y": bnd(ATM.with_(T=T)) for T in (0.5, 1.0, 2.0)}}
    plots.boundary_scenarios(panels, ATM.K, FIG / "boundary_scenarios.png")

    # early-exercise premium heatmap over volatility x rates
    sig = np.round(np.linspace(0.1, 0.5, 9), 2)
    rr = np.round(np.linspace(0.0, 0.12, 7), 2)
    Z = np.array([[bbsr(ATM.with_(sigma=s, r=r), 800) - bs_price(ATM.with_(sigma=s, r=r)) for s in sig] for r in rr])
    pd.DataFrame(Z, index=rr, columns=sig).to_csv(RES / "premium_heatmap.csv")
    plots.heatmap(Z, sig, rr, "Volatility σ", "Interest rate r", "Early-exercise premium of the ATM put ($)",
                  "American − European ($)", FIG / "premium_heatmap.png")
    def s_star(spec):
        b = crr(spec, 1000, boundary=True).boundary_S
        return b[0] if len(b) else np.nan  # no early-exercise region (e.g. r = 0)
    Zs = np.array([[s_star(ATM.with_(sigma=s, r=r)) for s in sig] for r in rr])
    pd.DataFrame(Zs, index=rr, columns=sig).to_csv(RES / "critical_price_heatmap.csv")
    plots.heatmap(Zs, sig, rr, "Volatility σ", "Interest rate r", "Critical price S* today (exercise below it)",
                  "S* ($)", FIG / "critical_price_heatmap.png", fmt="{:.1f}")

    # stress scenarios
    scen = {"Base (ATM, σ 20%, r 6%, 1y)": {}, "Volatility +50% (σ 30%)": dict(sigma=0.30),
            "Volatility −50% (σ 10%)": dict(sigma=0.10), "Rates +300bp (r 9%)": dict(r=0.09),
            "Rates to zero": dict(r=0.0), "Short maturity (3 months)": dict(T=0.25),
            "Long maturity (3 years)": dict(T=3.0), "Deep in the money (K 48)": dict(K=48.0),
            "Out of the money (K 34)": dict(K=34.0), "Dividend yield 3%": dict(q=0.03)}
    rows = []
    for name, kw in scen.items():
        spec = ATM.with_(**kw)
        tr = crr(spec, 1500, boundary=True)
        r = lsmc(spec, LSMCConfig(n_paths=q["paths"] // 2, n_steps=dates(spec), seed=0))
        amer, eur = bbsr(spec, 1000), bs_price(spec)
        rows.append(dict(scenario=name, american=amer, european=eur, premium=amer - eur,
                         premium_pct=(amer - eur) / amer * 100, critical_price_today=tr.boundary_S[0] if len(tr.boundary_S) else np.nan,
                         delta=tr.delta, gamma=tr.gamma, theta=tr.theta, lsmc=r.price, lsmc_se=r.stderr))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "scenarios.csv", index=False)

    # Greeks across spot
    Ss = np.arange(26, 56, 1.0)
    rows = []
    for S in Ss:
        spec = ATM.with_(S0=float(S))
        tr = crr(spec, 1000)
        ld = np.nan
        if S % 4 == 2:
            h = 0.5
            base_paths = simulate_gbm(spec.with_(S0=1.0), q["paths"] // 4, 50, np.random.default_rng(3))
            up = lsmc(spec.with_(S0=S + h), LSMCConfig(), paths=base_paths * (S + h)).price
            dn = lsmc(spec.with_(S0=S - h), LSMCConfig(), paths=base_paths * (S - h)).price
            ld = (up - dn) / (2 * h)
        rows.append(dict(S=S, american=tr.price, european=bs_price(spec), intrinsic=max(ATM.K - S, 0),
                         delta=tr.delta, gamma=tr.gamma, theta=tr.theta, lsmc_delta=ld))
    g = pd.DataFrame(rows)
    g.to_csv(RES / "greeks.csv", index=False)
    plots.greeks(g, FIG / "greeks.png")
    return sweeps, df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    args = ap.parse_args()
    q = dict(paths=100_000, oos=100_000, seeds=25, seeds_small=5, seeds_vr=40, seeds_ml=3, ml_paths=20_000,
             basis_paths=20_000, tree_max=500, path_grid=[1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000],
             date_grid=[5, 10, 25, 50, 100, 200])
    if args.quick:
        q.update(paths=4000, oos=4000, seeds=3, seeds_small=2, seeds_vr=4, seeds_ml=1, ml_paths=2000,
                 basis_paths=2000, tree_max=60, path_grid=[1000, 4000], date_grid=[5, 25])
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    summary = {}

    log("1/7 Benchmark table (20 Longstaff-Schwartz cases)")
    bt = benchmark_table(q)
    summary["benchmark_mean_abs_err_vs_bermudan_cents"] = float(bt["lsmc_err_vs_berm"].abs().mean() * 100)
    summary["benchmark_max_abs_err_vs_bermudan_cents"] = float(bt["lsmc_err_vs_berm"].abs().max() * 100)

    log("2/7 Convergence")
    _, amer = tree_convergence(q)
    berm = bermudan_tree(BASE, 50)
    summary.update(base_american=amer, base_bermudan_50=berm, base_european=bs_price(BASE))
    conv = lsmc_convergence(q, berm)
    summary["rmse_100k_paths_cents"] = float(conv["rmse_in"].iloc[-1] * 100)
    exercise_date_study(q, amer)

    log("3/7 Basis functions and regression specifications")
    bs = basis_study(q, berm)
    best = bs.sort_values("rmse_in").iloc[0]
    summary["best_basis"] = dict(spec=best["spec"], family=best["family"], degree=int(best["degree"]),
                                 rmse_cents=float(best["rmse_in"] * 100))

    log("4/7 Variance reduction")
    vr = variance_reduction(q, berm)
    summary["variance_reduction"] = dict(zip(vr["method"], vr["variance_reduction"].round(2)))

    log("5/7 Machine-learning and RL continuation values")
    ml = ml_comparison(q)
    summary["ml_mean_oos_error_cents"] = (ml.groupby("method", sort=False)["err_oos"].mean() * 100).round(2).to_dict()

    log("6/7 Exercise boundaries")
    boundary_study(q)

    log("7/7 Sensitivity and scenario analysis")
    sensitivity(q)

    summary["seconds"] = round(time.perf_counter() - t0, 1)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
