"""Matplotlib figures for the experiments. One consistent style across all charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1d2433"
MUTED = "#6b7280"
GRID = "#e5e7eb"
PALETTE = ["#1f4e79", "#d1495b", "#e09f3e", "#00798c", "#5a8a3c", "#7b5ea7", "#9c6644"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "legend.frameon": False, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "lines.linewidth": 1.8,
})


def save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)


def tree_convergence(df, bench, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(df["n"], df["crr"], color=PALETTE[0], lw=1.0, label="CRR tree")
    ax.plot(df["n"], df["bbs"], color=PALETTE[2], label="BBS (Black-Scholes last step)")
    ax.plot(df["n"], df["bbsr"], color=PALETTE[1], label="BBSR (BBS + Richardson)")
    ax.axhline(bench, color=INK, ls="--", lw=1, label=f"Benchmark {bench:.4f}")
    ax.set_ylim(bench - 0.03, bench + 0.03)
    ax.set_xlabel("Tree steps")
    ax.set_ylabel("American put price")
    ax.set_title("Binomial tree convergence")
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, path)


def lsmc_convergence(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    for col, lab, c in [("rmse_in", "In-sample", PALETTE[0]), ("rmse_oos", "Out-of-sample policy", PALETTE[1])]:
        ax.loglog(df["n_paths"], df[col], "o-", color=c, label=lab)
    ref = df["std_in"].iloc[0] * np.sqrt(df["n_paths"].iloc[0] / df["n_paths"])
    ax.loglog(df["n_paths"], ref, ":", color=MUTED, label=r"$1/\sqrt{N}$ reference")
    ax.set_xlabel("Simulated paths")
    ax.set_ylabel("RMSE vs Bermudan benchmark")
    ax.set_title("LSMC error shrinks with more paths")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    ax.semilogx(df["n_paths"], df["bias_in"], "o-", color=PALETTE[0], label="In-sample bias")
    ax.semilogx(df["n_paths"], df["bias_oos"], "o-", color=PALETTE[1], label="Out-of-sample bias")
    ax.fill_between(df["n_paths"], df["bias_in"] - df["std_in"], df["bias_in"] + df["std_in"], color=PALETTE[0], alpha=0.12)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("Simulated paths")
    ax.set_ylabel("Mean price − benchmark")
    ax.set_title("Bias: in-sample high, out-of-sample low")
    ax.legend(fontsize=8.5)
    save(fig, path)


def exercise_dates(df, amer, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(df["n_dates"], df["bermudan_tree"], "s-", color=INK, label="Bermudan tree (exact for that grid)")
    ax.errorbar(df["n_dates"], df["lsmc_mean"], yerr=2 * df["lsmc_std"], fmt="o-", color=PALETTE[0], capsize=3,
                label="LSMC (mean ± 2 sd)")
    ax.axhline(amer, color=PALETTE[1], ls="--", lw=1.2, label=f"American (continuous) {amer:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Exercise dates per year")
    ax.set_ylabel("Price")
    ax.set_title("More exercise dates close the gap to the American price")
    ax.legend(fontsize=8.5)
    save(fig, path)


def basis_study(df, path):
    fams = list(dict.fromkeys(df["family"]))
    specs = list(dict.fromkeys(df["spec"]))
    fig, axes = plt.subplots(1, len(specs), figsize=(12, 3.8), sharey=True)
    for ax, sp in zip(np.atleast_1d(axes), specs):
        sub = df[df["spec"] == sp]
        for i, f in enumerate(fams):
            g = sub[sub["family"] == f].sort_values("degree")
            ax.plot(g["degree"], g["rmse_in"] * 100, "o-", color=PALETTE[i], label=f)
        ax.set_title(sp)
        ax.set_xlabel("Polynomial degree")
        ax.set_yscale("log")
    np.atleast_1d(axes)[0].set_ylabel("In-sample RMSE vs benchmark (¢)")
    np.atleast_1d(axes)[0].legend(fontsize=8.5)
    fig.suptitle("Basis functions and regression specification", x=0.01, ha="left", fontweight="bold")
    save(fig, path)


def basis_oos(df, path):
    sub = df[df["spec"] == "OLS, in-the-money paths"]
    fams = list(dict.fromkeys(sub["family"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, f in enumerate(fams):
        g = sub[sub["family"] == f].sort_values("degree")
        ax.plot(g["degree"], g["bias_in"] * 100, "o-", color=PALETTE[i], label=f"{f} in-sample")
        ax.plot(g["degree"], g["bias_oos"] * 100, "s--", color=PALETTE[i], alpha=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("Polynomial degree (solid = in-sample, dashed = out-of-sample)")
    ax.set_ylabel("Mean error vs benchmark (¢)")
    ax.set_title("Richer bases cut out-of-sample error until degree 3–4")
    ax.legend(fontsize=8, ncol=2)
    save(fig, path)


def variance_reduction(df, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(df))
    h = 0.38
    ax.barh(y - h / 2, df["std"] * 100, height=h, color=PALETTE[0], label="Spread of the price across 40 runs")
    ax.barh(y + h / 2, df["within_run_se"] * 100, height=h, color=PALETTE[2], label="Standard error reported by each run")
    for i, (s, e) in enumerate(zip(df["std"] * 100, df["variance_reduction"])):
        ax.text(s, i - h / 2, f"  {e:.1f}×", va="center", fontsize=8, color=INK)
    for i, (s, e) in enumerate(zip(df["within_run_se"] * 100, df["sampling_variance_reduction"])):
        ax.text(s, i + h / 2, f"  {e:.1f}×", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, df["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Cents, at 10,000 paths (labels: variance reduction vs plain Monte Carlo)")
    ax.set_title("Variance reduction")
    ax.set_xlim(0, max(df["std"].max(), df["within_run_se"].max()) * 100 * 1.25)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, path)


def ml_comparison(df, path):
    methods = list(dict.fromkeys(df["method"]))
    cases = list(dict.fromkeys(df["case"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.8 / len(cases)
    for j, c in enumerate(cases):
        g = df[df["case"] == c].set_index("method").loc[methods]
        axes[0].bar(np.arange(len(methods)) + j * w, g["err_oos"] * 100, width=w, color=PALETTE[j], label=c)
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xticks(np.arange(len(methods)) + w * (len(cases) - 1) / 2, methods, rotation=20)
    axes[0].set_ylabel("Out-of-sample value − benchmark (¢)")
    axes[0].set_title("Exercise-policy quality on fresh paths")
    axes[0].legend(fontsize=8)
    t = df.groupby("method", sort=False)["seconds"].mean().loc[methods]
    axes[1].bar(methods, t, color=PALETTE[0])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Seconds per fit (log)")
    axes[1].set_title("Training time")
    axes[1].tick_params(axis="x", rotation=20)
    save(fig, path)


def boundaries(curves: dict, spec, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    styles = {"Binomial tree (2,000 steps)": dict(color=INK, lw=2.2)}
    for i, (name, (t, s)) in enumerate(curves.items()):
        st = styles.get(name, dict(color=PALETTE[i % len(PALETTE)], lw=1.4, alpha=0.9))
        ax.plot(t, s, label=name, **st)
    ax.axhline(spec.K, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Critical stock price S*(t)")
    ax.set_title("Optimal exercise boundary: exercise the put when S falls below the curve")
    ax.legend(fontsize=8.5)
    save(fig, path)


def boundary_scenarios(panels: dict, K, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3.6), sharey=True)
    for ax, (title, curves) in zip(axes, panels.items()):
        for i, (lab, (ttm, s)) in enumerate(curves.items()):
            ax.plot(ttm, s, color=PALETTE[i], label=lab)
        ax.axhline(K, color=MUTED, ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Time to maturity (years)")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Critical stock price S*")
    fig.suptitle("How the exercise boundary moves under each scenario", x=0.01, ha="left", fontweight="bold")
    save(fig, path)


def sensitivity(sweeps: dict, path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    labels = {"sigma": "Volatility σ", "r": "Interest rate r", "T": "Maturity T (years)", "K": "Strike K"}
    for ax, (k, df) in zip(axes, sweeps.items()):
        ax.plot(df[k], df["american"], color=PALETTE[0], label="American (tree)")
        ax.plot(df[k], df["european"], color=PALETTE[2], label="European (Black-Scholes)")
        ax.errorbar(df[k], df["lsmc"], yerr=2 * df["lsmc_se"], fmt="o", ms=3, color=PALETTE[1], label="LSMC ± 2 se")
        ax2 = ax.twinx()
        ax2.fill_between(df[k], 0, df["premium"], color=PALETTE[3], alpha=0.15)
        ax2.set_ylim(0, df["premium"].max() * 3 if df["premium"].max() > 0 else 1)
        ax2.grid(False)
        ax2.tick_params(colors=PALETTE[3])
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(PALETTE[3])
        ax.set_xlabel(labels[k])
    axes[0].set_ylabel("Put price")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.text(0.995, 0.5, "Early-exercise premium (shaded)", rotation=270, va="center", color=PALETTE[3], fontsize=9)
    fig.suptitle("Sensitivity of the at-the-money American put (S0 = K = 40)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def heatmap(Z, xs, ys, xlabel, ylabel, title, cbar, path, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = Z[i, j]
            ax.text(j, i, "none" if np.isnan(v) else fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="white" if np.isfinite(v) and v > np.nanmax(Z) * 0.6 else INK)
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar)
    save(fig, path)


def greeks(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(df["S"], df["american"], color=PALETTE[0], label="American")
    axes[0].plot(df["S"], df["european"], color=PALETTE[2], label="European")
    axes[0].plot(df["S"], df["intrinsic"], color=MUTED, ls=":", label="Intrinsic K − S")
    axes[0].set_title("Price")
    axes[1].plot(df["S"], df["delta"], color=PALETTE[0], label="Tree")
    m = df["lsmc_delta"].notna()
    axes[1].plot(df.loc[m, "S"], df.loc[m, "lsmc_delta"], "o", color=PALETTE[1], ms=4, label="LSMC bump (common paths)")
    axes[1].set_title("Delta")
    axes[2].plot(df["S"], df["gamma"], color=PALETTE[0])
    axes[2].set_title("Gamma")
    for ax in axes:
        ax.set_xlabel("Stock price S")
        ax.legend(fontsize=8) if ax.get_legend_handles_labels()[0] else None
    save(fig, path)


def sample_paths(S, t, ex_t, ex_S, bt, bs, K, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for row in S:
        ax.plot(t, row, color=PALETTE[0], lw=0.6, alpha=0.35)
    ax.plot(bt, bs, color=PALETTE[1], lw=2.2, label="Exercise boundary (tree)")
    ax.scatter(ex_t, ex_S, color=PALETTE[2], s=22, zorder=5, edgecolor=INK, lw=0.4, label="LSMC exercises here")
    ax.axhline(K, color=MUTED, ls=":", lw=1, label="Strike")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock price")
    ax.set_title("Simulated risk-neutral paths and where the learned policy exercises")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, path)
