"""Elsevier-style convergence comparison: SBS vs Selector vs VBS.

Two-panel figure:
  (a) FE vs terminal gap (log scale)
  (b) Wall-clock time vs terminal gap (log scale)
"""
from __future__ import annotations

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# ── Elsevier style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 0.8,
    "lines.markersize": 2.5,
})

COLORS = {
    "SBS (skip)": "#2166ac",
    "Selector (query)": "#b2182b",
    "VBS (oracle)": "#1a9850",
}
MARKERS = {
    "SBS (skip)": "o",
    "Selector (query)": "s",
    "VBS (oracle)": "D",
}

def load_data():
    sr = pq.read_table(
        "results/pilot_v2/selection_reference/selection_reference.parquet"
    ).to_pandas()

    # ── SBS (skip path) ──
    sbs = pd.DataFrame({
        "gap": sr["loss_skip"].astype(float).values,
        "first_hit_FE": pd.to_numeric(sr["skip_first_hit_FE"], errors="coerce").values,
        "success": sr["skip_success"].astype(bool).values,
        "runtime": sr["runtime_no_query_optimization"].astype(float).values,
        "effective_FE": sr["skip_effective_FE"].astype(int).values,
        "prefix_algorithm": sr["prefix_algorithm"].astype(str).values,
        "function_id": sr["function_id"].astype(str).values,
        "FE_prefix": sr["FE_prefix"].astype(int).values,
    })

    # ── Selector (query path) ──
    sel = pd.DataFrame({
        "gap": sr["selected_action_loss"].astype(float).values,
        "first_hit_FE": pd.to_numeric(sr["selected_action_first_hit_FE"], errors="coerce").values,
        "success": sr["selected_action_success"].astype(bool).values,
        "runtime": (
            sr["runtime_query"].astype(float).values
            + sr["runtime_selection"].astype(float).values
            + sr["runtime_handoff"].astype(float).values
            + sr["runtime_selected_action_optimization"].astype(float).values
        ),
        "effective_FE": sr["selected_action_effective_FE"].astype(int).values,
        "prefix_algorithm": sr["prefix_algorithm"].astype(str).values,
        "function_id": sr["function_id"].astype(str).values,
        "FE_prefix": sr["FE_prefix"].astype(int).values,
    })

    # ── VBS (oracle best) ──
    # Per-algorithm losses
    algo_gaps = {}
    algo_runtimes = {}
    algo_first_hits = {}
    algo_success = {}
    for algo in ["de", "pso", "cmaes", "shade"]:
        gap_col = f"observed_loss_{algo}"
        algo_gaps[algo] = sr[gap_col].astype(float).values
    # VBS gap = best_observed_loss
    vbs_gap = sr["best_observed_loss"].astype(float).values
    vbs_algo = sr["best_observed_algorithm"].astype(str).values

    # VBS runtime: for each state, use the runtime of the best algorithm
    # We need per-algorithm runtime from action losses
    # For now, approximate from selector runtime + skip runtime based on which algo is best
    # Load action losses to get per-algo runtime
    rt_frames = []
    for f in [1, 3, 15, 24]:
        al = pq.read_table(
            f"results/pilot_v2/action_losses/query_adjusted_f{f}.parquet"
        ).to_pandas()
        rt_frames.append(al)
    al_all = pd.concat(rt_frames, ignore_index=True)

    STATE_KEY = [
        "split", "problem_id", "function_id", "family", "cv_group_id",
        "dimension", "prefix_algorithm", "seed", "FE",
    ]

    # Build per-state per-algo runtime lookup
    al_all["runtime_total"] = (
        al_all["runtime_handoff"].astype(float)
        + al_all["runtime_action_optimization"].astype(float)
    )
    # For VBS, runtime = query + selection + handoff(best_algo) + action_opt(best_algo)
    # handoff and action_opt depend on target_algorithm
    rt_lookup = {}
    for _, row in al_all.iterrows():
        key = tuple(str(row[c]) for c in STATE_KEY)
        target = str(row["target_algorithm"])
        rt_total = float(row["runtime_total"])
        rt_lookup[(key, target)] = rt_total

    vbs_runtime = np.zeros(len(sr))
    for i in range(len(sr)):
        key = tuple(str(sr.iloc[i][c]) for c in STATE_KEY)
        best_a = vbs_algo[i]
        vbs_runtime[i] = rt_lookup.get((key, best_a), 0.0)

    # VBS first_hit: use per-algo first_hit from action losses
    fh_lookup = {}
    for _, row in al_all.iterrows():
        key = tuple(str(row[c]) for c in STATE_KEY)
        target = str(row["target_algorithm"])
        fh = row["first_hit_FE"]
        fh_lookup[(key, target)] = fh

    vbs_fh = np.full(len(sr), np.nan)
    vbs_succ = np.zeros(len(sr), dtype=bool)
    for i in range(len(sr)):
        key = tuple(str(sr.iloc[i][c]) for c in STATE_KEY)
        best_a = vbs_algo[i]
        fh_val = fh_lookup.get((key, best_a))
        if pd.notna(fh_val) and int(fh_val) > 0:
            vbs_fh[i] = int(fh_val)
            vbs_succ[i] = True

    vbs = pd.DataFrame({
        "gap": vbs_gap,
        "first_hit_FE": vbs_fh,
        "success": vbs_succ,
        "runtime": vbs_runtime,
        "effective_FE": sr["selected_action_effective_FE"].astype(int).values,  # same planned FE
        "prefix_algorithm": sr["prefix_algorithm"].astype(str).values,
        "function_id": sr["function_id"].astype(str).values,
        "FE_prefix": sr["FE_prefix"].astype(int).values,
    })

    return {"SBS (skip)": sbs, "Selector (query)": sel, "VBS (oracle)": vbs}


def plot_convergence(data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ── Panel (a): FE vs gap ──
    ax = axes[0]
    for name, df in data.items():
        # Sort by effective_FE (which equals FE_prefix + continuation)
        fe = df["effective_FE"].values
        gap = df["gap"].values
        # Sort by FE
        order = np.argsort(fe)
        fe_sorted = fe[order]
        gap_sorted = gap[order]
        # Cumulative: at each FE threshold, what fraction of states have gap <= threshold?
        # Plot as scatter: each state is one point
        ax.scatter(fe_sorted, gap_sorted + 1e-12,
                   c=COLORS[name], marker=MARKERS[name], alpha=0.3, s=3, label=name)

    ax.set_yscale("log")
    ax.set_xlabel("Effective function evaluations")
    ax.set_ylabel("Optimality gap (log scale)")
    ax.set_title("(a) Convergence: FE vs gap")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.5")
    ax.set_ylim(1e-12, 1e3)
    ax.grid(True, alpha=0.2, linewidth=0.3)

    # ── Panel (b): Wall-clock time vs gap ──
    ax = axes[1]
    for name, df in data.items():
        rt = df["runtime"].values
        gap = df["gap"].values
        order = np.argsort(rt)
        rt_sorted = rt[order]
        gap_sorted = gap[order]
        ax.scatter(rt_sorted, gap_sorted + 1e-12,
                   c=COLORS[name], marker=MARKERS[name], alpha=0.3, s=3, label=name)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Optimality gap (log scale)")
    ax.set_title("(b) Convergence: time vs gap")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.5")
    ax.set_ylim(1e-12, 1e3)
    ax.grid(True, alpha=0.2, linewidth=0.3)

    plt.tight_layout(w_pad=0.5)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_cdf(data, output_path):
    """CDF of gap for each method, Elsevier style."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ── Panel (a): gap CDF ──
    ax = axes[0]
    for name, df in data.items():
        gaps = df["gap"].values
        gaps_sorted = np.sort(gaps)
        cdf = np.arange(1, len(gaps_sorted) + 1) / len(gaps_sorted)
        ax.plot(gaps_sorted + 1e-12, cdf, color=COLORS[name], linewidth=0.8,
                label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Optimality gap")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("(a) Gap CDF")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.5")
    ax.set_xlim(1e-12, 1e3)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linewidth=0.3)

    # ── Panel (b): first_hit_FE CDF (convergence speed) ──
    ax = axes[1]
    for name, df in data.items():
        succ = df[df["success"]]
        if len(succ) == 0:
            continue
        fh = succ["first_hit_FE"].dropna().values
        if len(fh) == 0:
            continue
        fh_sorted = np.sort(fh)
        cdf = np.arange(1, len(fh_sorted) + 1) / len(fh_sorted)
        ax.plot(fh_sorted, cdf, color=COLORS[name], linewidth=0.8,
                marker=MARKERS[name], markevery=max(1, len(fh_sorted)//10),
                markersize=2.5, label=name)

    ax.set_xlabel("First-hit FE (among successful states)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("(b) Convergence speed CDF")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.5")
    ax.grid(True, alpha=0.2, linewidth=0.3)

    plt.tight_layout(w_pad=0.5)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_by_function(data, output_path):
    """Per-function comparison: grouped bar chart of mean gap."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    functions = sorted(data["SBS (skip)"]["function_id"].unique())
    x = np.arange(len(functions))
    width = 0.25

    for i, (name, df) in enumerate(data.items()):
        means = [float(df[df["function_id"] == f]["gap"].mean()) for f in functions]
        ax.bar(x + i * width, means, width, label=name, color=COLORS[name],
               edgecolor="0.3", linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f.replace("bbob_", "") for f in functions], rotation=30, ha="right")
    ax.set_ylabel("Mean optimality gap")
    ax.set_title("Mean gap by function")
    ax.legend(framealpha=0.9, edgecolor="0.5")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_pareto(data, output_path):
    """Pareto front: mean gap vs mean wall-clock time."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for name, df in data.items():
        mean_gap = float(df["gap"].mean())
        mean_rt = float(df["runtime"].mean())
        ax.scatter(mean_rt, mean_gap + 1e-12, c=COLORS[name], s=40,
                   marker=MARKERS[name], zorder=5, label=name)
        ax.annotate(name, (mean_rt, mean_gap + 1e-12),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, color=COLORS[name])

    ax.set_yscale("log")
    ax.set_xlabel("Mean wall-clock time (s)")
    ax.set_ylabel("Mean optimality gap (log)")
    ax.set_title("Pareto: gap vs time")
    ax.grid(True, alpha=0.2, linewidth=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    data = load_data()

    # Summary table
    print("\n" + "=" * 70)
    print("SBS / Selector / VBS 综合对比")
    print("=" * 70)
    print(f"{'':20s} {'SBS (skip)':>12s} {'Selector':>12s} {'VBS':>12s}")
    print("-" * 60)

    for label, key in [
        ("n states", None),
        ("success rate", "success"),
        ("gap mean", "gap"),
        ("gap median", "gap"),
        ("gap std", "gap"),
        ("first_hit mean", "first_hit_FE"),
        ("first_hit median", "first_hit_FE"),
        ("runtime mean (s)", "runtime"),
        ("runtime median (s)", "runtime"),
        ("effective FE mean", "effective_FE"),
    ]:
        vals = []
        for name in ["SBS (skip)", "Selector (query)", "VBS (oracle)"]:
            df = data[name]
            if key is None:
                vals.append(f"{len(df):>12d}")
            elif key == "success":
                vals.append(f"{df[key].mean()*100:>11.1f}%")
            elif key == "first_hit_FE":
                succ = df[df["success"]]
                if len(succ) == 0 or succ[key].dropna().empty:
                    vals.append(f"{'N/A':>12s}")
                else:
                    vals.append(f"{float(succ[key].dropna().mean()):>12.1f}" if "mean" in label else f"{float(succ[key].dropna().median()):>12.1f}")
            else:
                vals.append(f"{float(df[key].mean()):>12.6f}" if "mean" in label else f"{float(df[key].median()):>12.6f}")
        print(f"{label:20s} {vals[0]} {vals[1]} {vals[2]}")

    # Per-function
    print(f"\n{'By function':20s} {'SBS':>12s} {'Selector':>12s} {'VBS':>12s} {'Sel/SBS':>8s} {'VBS/SBS':>8s}")
    print("-" * 72)
    for fid in sorted(data["SBS (skip)"]["function_id"].unique()):
        sbs_gap = float(data["SBS (skip)"][data["SBS (skip)"]["function_id"] == fid]["gap"].mean())
        sel_gap = float(data["Selector (query)"][data["Selector (query)"]["function_id"] == fid]["gap"].mean())
        vbs_gap = float(data["VBS (oracle)"][data["VBS (oracle)"]["function_id"] == fid]["gap"].mean())
        print(f"{fid:20s} {sbs_gap:>12.4f} {sel_gap:>12.4f} {vbs_gap:>12.4f} {sel_gap/sbs_gap:>8.2f} {vbs_gap/sbs_gap:>8.2f}")

    # Per-prefix
    print(f"\n{'By prefix':20s} {'SBS':>12s} {'Selector':>12s} {'VBS':>12s} {'Sel/SBS':>8s}")
    print("-" * 64)
    for algo in sorted(data["SBS (skip)"]["prefix_algorithm"].unique()):
        sbs_gap = float(data["SBS (skip)"][data["SBS (skip)"]["prefix_algorithm"] == algo]["gap"].mean())
        sel_gap = float(data["Selector (query)"][data["Selector (query)"]["prefix_algorithm"] == algo]["gap"].mean())
        vbs_gap = float(data["VBS (oracle)"][data["VBS (oracle)"]["prefix_algorithm"] == algo]["gap"].mean())
        print(f"{algo:20s} {sbs_gap:>12.4f} {sel_gap:>12.4f} {vbs_gap:>12.4f} {sel_gap/sbs_gap:>8.2f}")

    # Runtime
    print(f"\n{'Runtime (s)':20s} {'SBS':>12s} {'Selector':>12s} {'VBS':>12s}")
    print("-" * 60)
    for name in ["SBS (skip)", "Selector (query)", "VBS (oracle)"]:
        df = data[name]
        print(f"{name:20s} {float(df['runtime'].mean()):>12.6f} {float(df['runtime'].median()):>12.6f} {float(df['runtime'].std()):>12.6f}")

    # Generate figures
    plot_convergence(data, "figures/pilot_convergence_scatter.pdf")
    plot_cdf(data, "figures/pilot_convergence_cdf.pdf")
    plot_by_function(data, "figures/pilot_by_function.pdf")
    plot_pareto(data, "figures/pilot_pareto.pdf")


if __name__ == "__main__":
    main()
