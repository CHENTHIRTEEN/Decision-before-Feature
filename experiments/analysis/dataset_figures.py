"""Dataset-quality figures for the predefined train/validation corpora.

Renders four Elsevier-contract figures plus a machine-readable summary:

Fig 1  composition and sampling-protocol conformity
Fig 2  portfolio heterogeneity and per-problem decision space
Fig 3  behavior-feature (B3) space health
Fig 4  landscape-descriptor coverage of the MA-BBOB augmentation

All rows of every shard are used; no exclusions. Display-axis truncations
are annotated on the figure and quantiles are kept in the summary JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from behavior.features import BEHAVIOR_FEATURE_GROUPS  # noqa: E402

GROUPS = {
    "bbob_train": REPO / "results/phase1_refined_sampling/bbob_train",
    "mabbob_formal": REPO / "results/phase1_mabbob/mabbob_formal",
    "bbob_validation": REPO / "results/phase1_refined_sampling/bbob_validation",
    "mabbob_validation": REPO / "results/phase1_mabbob_validation/mabbob_validation",
}
GROUP_ORDER = ["bbob_train", "mabbob_formal", "bbob_validation", "mabbob_validation"]
GROUP_LABEL = {
    "bbob_train": "BBOB train",
    "mabbob_formal": "MA-BBOB train",
    "bbob_validation": "BBOB validation",
    "mabbob_validation": "MA-BBOB validation",
}
DATASET = {g: g.split("_", 1)[1].replace("formal", "train") for g in GROUP_ORDER}
SUITE = {g: "BBOB" if g.startswith("bbob") else "MA-BBOB" for g in GROUP_ORDER}
ALGOS = ["de", "pso", "cmaes", "shade"]
ALGO_LABEL = {"de": "DE", "pso": "PSO", "cmaes": "CMA-ES", "shade": "SHADE"}
ALGO_COLOR = {"de": "#0072B2", "pso": "#E69F00", "cmaes": "#009E73", "shade": "#CC79A7"}
DIM_COLOR = {10: "#BDBDBD", 20: "#7F7F7F", 40: "#404040"}
PHASE_COLOR = {"early": "#C6DBEF", "mid": "#6BAED6", "late": "#2171B5"}
SUITE_COLOR = {"BBOB": "#0072B2", "MA-BBOB": "#E69F00"}

OUT = REPO / "results/dataset_analysis"
FIGDIR = OUT / "figures"
SRCDIR = OUT / "source_data"
B3 = list(BEHAVIOR_FEATURE_GROUPS["B3"])
DESCRIPTOR_COLS = [
    f"descriptor_{name}" for name in (
        "y_min", "y_max", "y_mean", "y_std", "y_skew", "y_kurtosis",
        "x_mean_pairwise", "x_std_pairwise", "x_best_dist_center",
        "x_mean_dist_center", "corr_y_dist_center", "corr_y_nn_dist",
        "linear_r2", "linear_gradient_norm",
    )
]

TRAJ_COLS = [
    "problem_id", "algorithm", "seed", "FE", "FE_total", "dimension",
    "sampling_phase", "is_budget_milestone", "is_event_sample",
]
FP_COLS = [
    "problem_id", "function_id", "family", "dimension",
    "algorithm", "seed", "log10_gap", "success", "path_completed",
]


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fp_parts, traj_parts, beh_parts = [], [], []
    for group, root in GROUPS.items():
        for shard in sorted(root.glob("*/dimension_*")):
            fp = pq.read_table(shard / "final_performance.parquet", columns=FP_COLS).to_pandas()
            fp["group"] = group
            fp_parts.append(fp)
            traj = pq.read_table(shard / "trajectories.parquet", columns=TRAJ_COLS).to_pandas()
            traj["group"] = group
            traj_parts.append(traj)
            beh = pq.read_table(
                shard / "behavior.parquet",
                columns=["problem_id", "algorithm", "seed", "FE", "dimension"] + B3,
            ).to_pandas()
            beh["group"] = group
            beh_parts.append(beh)
    fp = pd.concat(fp_parts, ignore_index=True)
    traj = pd.concat(traj_parts, ignore_index=True)
    beh = pd.concat(beh_parts, ignore_index=True)
    for frame in (fp, traj, beh):
        frame["dataset"] = frame["group"].map(DATASET)
        frame["suite"] = frame["group"].map(SUITE)
    return fp, traj, beh


def load_descriptors() -> pd.DataFrame:
    parts = []
    for split in ["bbob_train", "mabbob_formal", "bbob_validation", "mabbob_validation"]:
        table = pq.read_table(
            REPO / f"results/landscape_queries/features/descriptor_cheap_invariant/{split}/features.parquet",
            columns=["problem_id", "dimension"] + DESCRIPTOR_COLS,
        ).to_pandas()
        table["group"] = split
        table["dataset"] = DATASET[split]
        table["suite"] = SUITE[split]
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def panel_label(axis, letter: str) -> None:
    axis.text(
        -0.09, 1.06, letter, transform=axis.transAxes,
        fontsize=8, fontweight="bold", va="top", ha="left",
    )


def style_axes() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


def save_pub(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_source(stem: str, frame: pd.DataFrame) -> None:
    SRCDIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SRCDIR / f"{stem}.csv", index=False)


MM = 1.0 / 25.4


def figure1(traj: pd.DataFrame, fp: pd.DataFrame, stats: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(190 * MM, 125 * MM))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) runs per group x dimension
    counts = fp.groupby(["group", "dimension"]).size().unstack(fill_value=0)
    x = np.arange(len(GROUP_ORDER))
    width = 0.26
    for offset, dim in zip([-width, 0.0, width], [10, 20, 40]):
        values = [counts.loc[g, dim] for g in GROUP_ORDER]
        ax_a.bar(x + offset, values, width=width, color=DIM_COLOR[dim], label=f"{dim}D", linewidth=0)
        for xi, value in zip(x + offset, values):
            ax_a.text(xi, value + 40, str(value), ha="center", va="bottom", fontsize=5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER], rotation=16, ha="right", rotation_mode="anchor")
    ax_a.set_ylabel("Runs")
    ax_a.set_ylim(0, counts.to_numpy().max() * 1.16)
    ax_a.legend(title="Dimension", ncols=3, loc="upper right", title_fontsize=6)

    # (b) states per run, per group
    states_per_run = traj.groupby(["group", "problem_id", "algorithm", "seed"]).size().reset_index(name="n")
    for group in GROUP_ORDER:
        values = states_per_run.loc[states_per_run["group"] == group, "n"]
        suite = SUITE[group]
        hist = values.value_counts().reindex(range(12, 19), fill_value=0).sort_index()
        style = "-" if DATASET[group] == "train" else "--"
        ax_b.step(hist.index, hist.values, where="mid",
                  linestyle=style, color=SUITE_COLOR[suite], linewidth=1.1,
                  label=f"{suite} {DATASET[group]}")
    ax_b.axvspan(12, 18, color="#F5F5F5", zorder=0)
    ax_b.set_xlim(11.5, 18.5)
    ax_b.set_xticks(range(12, 19))
    ax_b.set_xlabel("Decision states per run")
    ax_b.set_ylabel("Runs")
    ax_b.legend(ncols=2, loc="upper left")

    # (c) state composition per group
    milestone_flag = traj["is_budget_milestone"].astype(bool)
    event_flag = traj["is_event_sample"].astype(bool)
    comp = pd.DataFrame({
        "group": traj["group"],
        "milestone_only": milestone_flag & ~event_flag,
        "merged": milestone_flag & event_flag,
        "event_only": ~milestone_flag & event_flag,
    }).groupby("group").sum().loc[GROUP_ORDER]
    totals = comp.sum(axis=1)
    fractions = comp.div(totals, axis=0) * 100
    bottom = np.zeros(len(GROUP_ORDER))
    for column, color, label in [
        ("milestone_only", "#969696", "Milestone only"),
        ("merged", "#525252", "Milestone + event"),
        ("event_only", "#D9D9D9", "Event only"),
    ]:
        ax_c.bar(x, fractions[column], bottom=bottom, color=color, label=label, linewidth=0, width=0.62)
        bottom += fractions[column].to_numpy()
    for xi, group in zip(x, GROUP_ORDER):
        ax_c.text(xi, 102, f"{int(totals[group]):,}", ha="center", va="bottom", fontsize=5.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER], rotation=16, ha="right", rotation_mode="anchor")
    ax_c.set_ylabel("Share of decision states (%)")
    ax_c.set_ylim(0, 118)
    ax_c.legend(ncols=3, loc="lower right")
    ax_c.text(0.02, 1.005, "total states above bars", transform=ax_c.transAxes, fontsize=5.5, color="#404040")

    # (d) event-only states by monitoring phase
    event_only = traj[(~traj["is_budget_milestone"]) & (traj["is_event_sample"])]
    phase = event_only.groupby(["group", "sampling_phase"]).size().unstack(fill_value=0)
    phase = phase.reindex(GROUP_ORDER).reindex(columns=["early", "mid", "late"], fill_value=0)
    bottom = np.zeros(len(GROUP_ORDER))
    for phase_name in ["early", "mid", "late"]:
        values = phase[phase_name].to_numpy(dtype=float)
        ax_d.bar(x, values, bottom=bottom, color=PHASE_COLOR[phase_name],
                 label=phase_name.capitalize(), linewidth=0, width=0.62)
        bottom += values
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER], rotation=16, ha="right", rotation_mode="anchor")
    ax_d.set_ylabel("Event-only states")
    ax_d.legend(ncols=3, loc="upper right")

    stats["fig1"] = {
        "runs_per_group_dimension": {
            g: counts.loc[g].to_dict() for g in GROUP_ORDER
        },
        "states_per_run": {
            g: {
                "min": int(states_per_run.loc[states_per_run["group"] == g, "n"].min()),
                "max": int(states_per_run.loc[states_per_run["group"] == g, "n"].max()),
                "mean": round(float(states_per_run.loc[states_per_run["group"] == g, "n"].mean()), 3),
            } for g in GROUP_ORDER
        },
        "state_composition": comp.to_dict(),
        "event_only_by_phase": phase.to_dict(),
    }
    save_source("fig1a_runs_per_group_dimension", counts.reset_index())
    save_source("fig1b_states_per_run", states_per_run)
    save_source("fig1c_state_composition", comp.reset_index())
    save_source("fig1d_event_only_by_phase", phase.reset_index())
    for axis, letter in zip(axes.flat, ["a", "b", "c", "d"]):
        panel_label(axis, letter)
    fig.tight_layout(pad=0.6, h_pad=1.4, w_pad=1.6)
    save_pub(fig, "fig1_dataset_composition")


def figure2(fp: pd.DataFrame, stats: dict) -> None:
    fig = plt.figure(figsize=(190 * MM, 138 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.24)
    import seaborn as sns

    # (a)/(b) final clipped log10 gap per algorithm x dimension per dataset
    for axis, dataset, letter in [(fig.add_subplot(grid[0, 0]), "train", "a"),
                                  (fig.add_subplot(grid[0, 1]), "validation", "b")]:
        sub = fp[fp["dataset"] == dataset].copy()
        sub["algo_dim"] = sub["algorithm"].map(ALGO_LABEL) + " " + sub["dimension"].astype(str) + "D"
        order = [f"{ALGO_LABEL[a]} {d}D" for d in (10, 20, 40) for a in ALGOS]
        colors = [ALGO_COLOR[a] for d in (10, 20, 40) for a in ALGOS]
        sns.violinplot(
            data=sub, x="algo_dim", y="log10_gap", hue="algo_dim", order=order,
            hue_order=order, palette=dict(zip(order, colors)), legend=False,
            cut=0, inner="quartile", linewidth=0.5, saturation=0.85, ax=axis,
        )
        for artist in axis.collections:
            artist.set_alpha(0.85)
        axis.set_ylim(-12.6, 6.0)
        axis.set_xlabel("")
        axis.set_ylabel("Final log10 gap")
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels([label.split()[0] for label in order], rotation=90, rotation_mode="anchor")
        for dim_index, dim in enumerate((10, 20, 40)):
            axis.text(dim_index * 4 + 1.5, 5.6, f"{dim}D", ha="center", va="top", fontsize=6.5)
        axis.set_title(f"{dataset.capitalize()} corpus (n = {len(sub):,} runs)", loc="left", fontsize=7)
        beyond = float((sub["log10_gap"] > 6.0).mean() * 100)
        if beyond > 0:
            axis.text(0.99, 0.96, f"{beyond:.1f}% above axis cap",
                      transform=axis.transAxes, ha="right", va="top", fontsize=5, color="#404040")
        panel_label(axis, letter)

    # (c) per-problem best-vs-worst algorithm spread
    axis = fig.add_subplot(grid[1, 0])
    problem_alg = fp.groupby(["dataset", "dimension", "problem_id", "algorithm"])["log10_gap"].mean().reset_index()
    spread = problem_alg.groupby(["dataset", "dimension", "problem_id"])["log10_gap"].agg(
        lambda s: s.max() - s.min()
    ).reset_index(name="spread")
    sns.violinplot(
        data=spread, x="dimension", y="spread", hue="dataset", order=[10, 20, 40],
        palette={"train": "#8DA0CB", "validation": "#F5B97F"}, cut=0, inner="quartile",
        linewidth=0.5, ax=axis,
    )
    for dataset, offset in [("train", -0.21), ("validation", 0.21)]:
        for dim_index, dim in enumerate((10, 20, 40)):
            values = spread[(spread["dataset"] == dataset) & (spread["dimension"] == dim)]["spread"]
            axis.text(dim_index + offset, values.median(), f"{values.median():.2f}",
                      ha="center", va="bottom", fontsize=5.2)
    axis.set_xlabel("Dimension")
    axis.set_ylabel("Per-problem best-minus-worst\nalgorithm spread (delta log10 gap)")
    axis.legend(title="Corpus", ncols=2)
    axis.set_title("Seed-mean final-gap spread across portfolio algorithms", loc="left", fontsize=7)
    panel_label(axis, "c")

    # (d) target-hit rate heatmap (rows: dataset x algorithm, cols: dimension)
    axis = fig.add_subplot(grid[1, 1])
    rate = fp.groupby(["dataset", "algorithm", "dimension"])["success"].mean().reset_index()
    rate["success"] *= 100
    matrix = rate.pivot(index=["dataset", "algorithm"], columns="dimension", values="success")
    row_order = [(dataset, algo) for dataset in ("train", "validation") for algo in ALGOS]
    matrix = matrix.reindex(row_order)[[10, 20, 40]]
    im = axis.imshow(matrix.to_numpy(), cmap="PuBu", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(3))
    axis.set_xticklabels(["10D", "20D", "40D"])
    axis.set_yticks(range(len(row_order)))
    axis.set_yticklabels(
        [f"{d.capitalize()} · {ALGO_LABEL[a]}" for d, a in row_order], fontsize=6,
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            axis.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=5.6,
                      color="white" if value > 60 else "#252525")
    axis.axhline(3.5, color="white", linewidth=1.4)
    cbar = fig.colorbar(im, ax=axis, fraction=0.035, pad=0.02)
    cbar.set_label("Target-hit rate (%)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    axis.set_xlabel("Dimension")
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    panel_label(axis, "d")

    stats["fig2"] = {
        "final_log10_gap_quantiles": {
            f"{dataset}|{algo}|{dim}": {
                "q10": round(float(sub.quantile(0.10)), 4),
                "median": round(float(sub.median()), 4),
                "q90": round(float(sub.quantile(0.90)), 4),
            }
            for (dataset, algo, dim), sub in fp.groupby(["dataset", "algorithm", "dimension"])["log10_gap"]
        },
        "spread_quantiles": {
            f"{dataset}|{dim}": {
                "median": round(float(values.median()), 4),
                "q25": round(float(values.quantile(0.25)), 4),
                "q75": round(float(values.quantile(0.75)), 4),
                "n_problems": int(len(values)),
            }
            for (dataset, dim), values in spread.groupby(["dataset", "dimension"])["spread"]
        },
        "target_hit_rate": {
            f"{dataset}|{algo}|{dim}": round(float(value), 2)
            for (dataset, algo, dim), value in
            fp.groupby(["dataset", "algorithm", "dimension"])["success"].mean().items()
        },
    }
    final_quantiles = (
        fp.groupby(["dataset", "algorithm", "dimension"])["log10_gap"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .reset_index()
    )
    final_quantiles.columns = [
        "dataset", "algorithm", "dimension", "q10", "q25", "median", "q75", "q90",
    ]
    save_source("fig2ab_final_log10_gap_quantiles", final_quantiles)
    save_source("fig2c_problem_algorithm_spread", spread)
    save_source("fig2d_target_hit_rate", rate)
    fig.tight_layout(pad=0.6)
    save_pub(fig, "fig2_portfolio_heterogeneity")


def figure3(beh: pd.DataFrame, stats: dict) -> None:
    train = beh[beh["dataset"] == "train"]
    features = train[B3].to_numpy(dtype=float)
    feature_names = [name[3:] for name in B3]

    # (a) Spearman correlation, hierarchically ordered
    corr = pd.DataFrame(features, columns=feature_names).corr(method="spearman")
    linkage_order = _leaves_order(corr.to_numpy())
    ordered = corr.iloc[linkage_order, linkage_order]

    fig = plt.figure(figsize=(190 * MM, 128 * MM))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.05, 1.5], height_ratios=[1.0, 1.25],
        hspace=0.42, wspace=0.28,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    im = ax_a.imshow(ordered.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_xlabel("31 B3 behavior features (cluster-ordered)", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.045, pad=0.02)
    cbar.set_label("Spearman correlation", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    panel_label(ax_a, "a")

    # (b) effective rank per algorithm (participation ratio)
    ax_b = fig.add_subplot(grid[0, 1])
    participation = {}
    for algo in ALGOS:
        matrix = train.loc[train["algorithm"] == algo, B3].to_numpy(dtype=float)
        z = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) > 0, matrix.std(axis=0), 1.0)
        eigenvalues = np.linalg.eigvalsh(np.corrcoef(z, rowvar=False))
        participation[algo] = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())
    bars = ax_b.bar(
        [ALGO_LABEL[a] for a in ALGOS], [participation[a] for a in ALGOS],
        color=[ALGO_COLOR[a] for a in ALGOS], width=0.6, linewidth=0,
    )
    for bar, algo in zip(bars, ALGOS):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                  f"{participation[algo]:.1f}", ha="center", va="bottom", fontsize=6)
    ax_b.set_ylim(0, 31)
    ax_b.set_ylabel("Effective rank of B3 features")
    ax_b.axhline(31, color="#969696", linewidth=0.6, linestyle=(0, (3, 2)))
    ax_b.text(3.42, 31.3, "31 (full)", fontsize=5.5, color="#525252", ha="right")
    ax_b.text(0.99, 0.90, "train corpus states", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=5.5, color="#404040")
    panel_label(ax_b, "b")

    # (c) per-algorithm median profile, z-scored over pooled train states
    ax_c = fig.add_subplot(grid[1, 1])
    pooled_mean = train[B3].mean()
    pooled_std = train[B3].std().replace(0.0, 1.0)
    profile = pd.DataFrame({
        ALGO_LABEL[algo]: (train.loc[train["algorithm"] == algo, B3].median() - pooled_mean) / pooled_std
        for algo in ALGOS
    }).T.iloc[:, linkage_order]
    im = ax_c.imshow(profile.to_numpy(), cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax_c.set_yticks(range(4))
    ax_c.set_yticklabels(profile.index)
    ax_c.set_xticks([])
    ax_c.set_xlabel("31 B3 behavior features (same order as a)", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.025, pad=0.02)
    cbar.set_label("Median z-score", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    panel_label(ax_c, "c")

    stats["fig3"] = {
        "effective_rank_participation": participation,
        "abs_spearman_mean": round(float(np.abs(ordered.to_numpy()).mean()), 4),
        "n_states_train": int(len(train)),
    }
    save_source("fig3a_b3_spearman_correlation_ordered", ordered.reset_index(drop=True))
    save_source(
        "fig3b_b3_effective_rank_by_algorithm",
        pd.DataFrame({
            "algorithm": ALGOS,
            "algorithm_label": [ALGO_LABEL[a] for a in ALGOS],
            "effective_rank": [participation[a] for a in ALGOS],
        }),
    )
    save_source("fig3c_b3_algorithm_median_z_profile", profile.reset_index(names="algorithm"))
    fig.tight_layout(pad=0.6)
    save_pub(fig, "fig3_behavior_space")


def _leaves_order(corr_matrix: np.ndarray) -> list[int]:
    """Average-linkage leaf order on |corr| distance without scipy linkage deps."""
    n = corr_matrix.shape[0]
    clusters = [[i] for i in range(n)]
    distances = 1.0 - np.abs(corr_matrix)
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                value = distances[np.ix_(clusters[i], clusters[j])].mean()
                if best is None or value < best[0]:
                    best = (value, i, j)
        _, i, j = best
        merged = clusters[i] + clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    return clusters[0]


def figure4(desc: pd.DataFrame, stats: dict) -> None:
    columns = [c for c in desc.columns if c.startswith("descriptor_")]
    z = (desc[columns] - desc[columns].mean()) / desc[columns].std()
    centered = z - z.mean()
    u, s, vt = np.linalg.svd(centered.to_numpy(), full_matrices=False)
    scores = u[:, :2] * s[:2]
    explained = (s ** 2 / (s ** 2).sum()) * 100

    fig, axis = plt.subplots(figsize=(140 * MM, 104 * MM))
    for suite in ("BBOB", "MA-BBOB"):
        suite_mask = desc["suite"] == suite
        suite_points = scores[suite_mask.to_numpy()]
        if len(suite_points) >= 3:
            hull_points = _convex_hull(suite_points)
            polygon = Polygon(
                hull_points, closed=True, fill=True, facecolor=SUITE_COLOR[suite],
                alpha=0.08, edgecolor=SUITE_COLOR[suite], linewidth=0.7, linestyle=(0, (3, 2)),
            )
            axis.add_patch(polygon)
    style_map = {("train",): ("o", 1.0), ("validation",): ("s", 0.55)}
    for dataset, (marker, alpha) in style_map.items():
        dataset = dataset[0]
        for suite in ("BBOB", "MA-BBOB"):
            mask = (desc["suite"] == suite) & (desc["dataset"] == dataset)
            count = int(mask.sum())
            axis.scatter(
                scores[mask.to_numpy(), 0], scores[mask.to_numpy(), 1],
                marker=marker, s=13 if marker == "o" else 16,
                facecolor=SUITE_COLOR[suite] if marker == "o" else "none",
                edgecolor=SUITE_COLOR[suite], linewidth=0.7, alpha=alpha,
                label=f"{suite}, {dataset} (n={count})",
            )
    axis.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    axis.legend(loc="best", handletextpad=0.3, borderpad=0.2)
    panel_label(axis, "a")

    stats["fig4"] = {
        "explained_variance_pc1_pc2_pct": [round(float(explained[0]), 2), round(float(explained[1]), 2)],
        "problems_by_suite_dataset": {
            f"{suite}|{dataset}": int(((desc["suite"] == suite) & (desc["dataset"] == dataset)).sum())
            for suite in ("BBOB", "MA-BBOB") for dataset in ("train", "validation")
        },
    }
    scores_frame = desc[["group", "dataset", "suite", "problem_id", "dimension"]].copy()
    scores_frame["PC1"] = scores[:, 0]
    scores_frame["PC2"] = scores[:, 1]
    save_source("fig4_descriptor_pca_scores", scores_frame)
    fig.tight_layout(pad=0.6)
    save_pub(fig, "fig4_descriptor_coverage")


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull."""
    pts = sorted(map(tuple, points))
    if len(pts) < 3:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def effective_rank(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float)
    z = (values - values.mean(axis=0)) / np.where(values.std(axis=0) > 0, values.std(axis=0), 1.0)
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(z, rowvar=False))
    return float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())


def build_feature_engineering_outputs(
    traj: pd.DataFrame,
    beh: pd.DataFrame,
    desc: pd.DataFrame,
    stats: dict,
) -> pd.DataFrame:
    feature_keys = ["group", "problem_id", "dimension", "algorithm", "seed", "FE"]
    budget = traj[feature_keys + ["FE_total"]].copy()
    feature_matrix = beh.merge(budget, on=feature_keys, how="inner", validate="one_to_one")
    descriptor_keys = ["group", "problem_id", "dimension"]
    feature_matrix = feature_matrix.merge(
        desc[descriptor_keys + DESCRIPTOR_COLS],
        on=descriptor_keys,
        how="left",
        validate="many_to_one",
    )
    feature_matrix["FE_prefix"] = feature_matrix["FE"]
    feature_matrix["remaining_budget_ratio"] = (
        (feature_matrix["FE_total"] - feature_matrix["FE_prefix"]) / feature_matrix["FE_total"]
    )
    feature_cols = B3 + DESCRIPTOR_COLS + ["FE_prefix", "FE_total", "remaining_budget_ratio"]
    assert len(B3) == 31, f"Expected 31 B3 behavior features, got {len(B3)}"
    assert len(DESCRIPTOR_COLS) == 14, f"Expected 14 descriptor features, got {len(DESCRIPTOR_COLS)}"
    assert len(feature_cols) >= 45, f"Expected >=45 features (B3 + descriptors), got {len(feature_cols)}"
    assert any(c.startswith("bf_") for c in feature_cols), "Missing behavior features"
    assert any(c.startswith("descriptor_") for c in feature_cols), "Missing descriptor features"

    feature_path = OUT / "engineered_b3_descriptor_budget_features.parquet"
    feature_matrix[
        ["group", "dataset", "suite", "problem_id", "dimension", "algorithm", "seed", "FE"] + feature_cols
    ].to_parquet(feature_path, index=False)

    nonfinite = {
        col: int((~np.isfinite(feature_matrix[col].to_numpy(dtype=float))).sum())
        for col in feature_cols
    }
    quality = []
    for (group, dimension, algorithm), sub in feature_matrix.groupby(["group", "dimension", "algorithm"]):
        behavior_values = sub[B3]
        descriptor_values = sub[DESCRIPTOR_COLS]
        quality.append({
            "group": group,
            "dimension": int(dimension),
            "algorithm": algorithm,
            "n_states": int(len(sub)),
            "b3_effective_rank": round(effective_rank(behavior_values), 4),
            "b3_zero_variance_features": int((behavior_values.std(axis=0) == 0).sum()),
            "descriptor_zero_variance_features": int((descriptor_values.std(axis=0) == 0).sum()),
            "feature_nonfinite_cells": int((~np.isfinite(sub[feature_cols].to_numpy(dtype=float))).sum()),
            "remaining_budget_min": round(float(sub["remaining_budget_ratio"].min()), 4),
            "remaining_budget_max": round(float(sub["remaining_budget_ratio"].max()), 4),
        })
    quality_frame = pd.DataFrame(quality).sort_values(["group", "dimension", "algorithm"])
    quality_frame.to_csv(OUT / "feature_engineering_quality_by_group_dimension_algorithm.csv", index=False)

    feature_stats = (
        feature_matrix.groupby(["group", "dimension", "algorithm"])[feature_cols]
        .agg(["mean", "std"])
    )
    feature_stats.columns = [f"{col}_{stat}" for col, stat in feature_stats.columns]
    feature_stats.reset_index().to_csv(OUT / "feature_engineering_summary_statistics.csv", index=False)

    stats["feature_engineering"] = {
        "feature_matrix_path": str(feature_path.relative_to(REPO)),
        "rows": int(len(feature_matrix)),
        "feature_columns": int(len(feature_cols)),
        "behavior_feature_columns": int(len(B3)),
        "descriptor_feature_columns": int(len(DESCRIPTOR_COLS)),
        "budget_feature_columns": 3,
        "nonfinite_feature_cells": int(sum(nonfinite.values())),
        "quality_table_path": "results/dataset_analysis/feature_engineering_quality_by_group_dimension_algorithm.csv",
        "summary_statistics_path": "results/dataset_analysis/feature_engineering_summary_statistics.csv",
        "min_group_dimension_algorithm_b3_effective_rank": round(float(quality_frame["b3_effective_rank"].min()), 4),
        "max_group_dimension_algorithm_b3_effective_rank": round(float(quality_frame["b3_effective_rank"].max()), 4),
    }
    return feature_matrix


def write_markdown_summary(stats: dict) -> None:
    spread = stats["fig2"]["spread_quantiles"]
    rank = stats["fig3"]["effective_rank_participation"]
    feature = stats["feature_engineering"]
    lines = [
        "# 数据集质量与特征工程汇总",
        "",
        "数据范围：预先指定的 BBOB / MA-BBOB train-validation 语料；所有 shard 行均纳入统计，无行排除。",
        "",
        "## 管线级检查",
        "",
        f"- runs 总数：{stats['corpora']['runs_total']:,}；decision states：{stats['corpora']['decision_states_total']:,}；失败 runs：{stats['corpora']['failed_runs']:,}。",
        f"- final `log10_gap` 非有限值：{stats['corpora']['nonfinite_log10_gap_rows']:,}；descriptor 非有限值：{stats['pipeline_checks']['descriptor_nonfinite']:,}。",
        f"- 每 run 状态数在协议范围 [12,18] 内：{stats['pipeline_checks']['states_per_run_within_protocol_bounds']}。",
        f"- behavior 行数等于 trajectory 行数：{stats['pipeline_checks']['behavior_rows_equal_trajectory_rows']}。",
        "",
        "## 特征工程产物",
        "",
        f"- 已生成 `{feature['feature_matrix_path']}`：{feature['rows']:,} 行，{feature['feature_columns']} 个输入列。",
        f"- 列组成：{feature['behavior_feature_columns']} 个 B3 行为特征、{feature['descriptor_feature_columns']} 个 landscape descriptor、3 个预算字段。",
        f"- 所有工程特征非有限 cell：{feature['nonfinite_feature_cells']:,}。",
        f"- 分组有效秩范围：{feature['min_group_dimension_algorithm_b3_effective_rank']:.2f}–{feature['max_group_dimension_algorithm_b3_effective_rank']:.2f}。",
        "",
        "## 论文必要图件",
        "",
        "- Fig. 1：数据构成、状态数和 milestone/event 采样协议一致性。",
        "- Fig. 2：算法组合终端性能分布、逐问题算法极差和 target-hit 率，说明选择问题非平凡。",
        "- Fig. 3：B3 行为特征相关结构、有效秩和分算法行为画像，说明 Decision X 非退化。",
        "- Fig. 4：14 维 descriptor 的 problem-level 覆盖，说明 MA-BBOB 是受控景观扩展。",
        "",
        "## 关键数值",
        "",
        f"- train 逐问题算法极差中位数：10D {spread['train|10']['median']:.2f}、20D {spread['train|20']['median']:.2f}、40D {spread['train|40']['median']:.2f} log10 gap。",
        f"- validation 逐问题算法极差中位数：10D {spread['validation|10']['median']:.2f}、20D {spread['validation|20']['median']:.2f}、40D {spread['validation|40']['median']:.2f} log10 gap。",
        f"- train B3 分算法有效秩：DE {rank['de']:.2f}、PSO {rank['pso']:.2f}、CMA-ES {rank['cmaes']:.2f}、SHADE {rank['shade']:.2f}。",
        f"- descriptor PCA 前两轴解释方差：{stats['fig4']['explained_variance_pc1_pc2_pct'][0]:.1f}% + {stats['fig4']['explained_variance_pc1_pc2_pct'][1]:.1f}%。",
        "",
        "## 复现",
        "",
        "```bash",
        "uv run python -m experiments.analysis.dataset_figures",
        "```",
        "",
        "注意：这里的逐问题 best-minus-worst algorithm spread 是描述性数据质量指标，不替代协议中的 fold-specific SBS/VBS 或后续 Utility 评价。",
    ]
    (OUT / "dataset_quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    style_axes()
    OUT.mkdir(parents=True, exist_ok=True)
    SRCDIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {"corpora": {}, "pipeline_checks": {}}
    fp, traj, beh = load_frames()
    desc = load_descriptors()

    stats["corpora"] = {
        "runs_total": int(len(fp)),
        "decision_states_total": int(len(traj)),
        "failed_runs": int((~fp["path_completed"].astype(bool)).sum()),
        "runs_by_group": {g: int((fp["group"] == g).sum()) for g in GROUP_ORDER},
        "nonfinite_log10_gap_rows": int((~np.isfinite(fp["log10_gap"])).sum()),
    }
    stats["pipeline_checks"] = {
        "states_per_run_within_protocol_bounds": bool(
            traj.groupby(["group", "problem_id", "algorithm", "seed"]).size().between(12, 18).all()
        ),
        "behavior_rows_equal_trajectory_rows": bool(len(beh) == len(traj)),
        "descriptor_problems": int(len(desc)),
        "descriptor_nonfinite": int((~np.isfinite(desc[[c for c in desc.columns if c.startswith('descriptor_')]].to_numpy())).sum()),
    }

    figure1(traj, fp, stats)
    figure2(fp, stats)
    figure3(beh, stats)
    figure4(desc, stats)
    build_feature_engineering_outputs(traj, beh, desc, stats)

    with (OUT / "dataset_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    write_markdown_summary(stats)
    print(json.dumps({k: stats[k] for k in ("corpora", "pipeline_checks")}, indent=2))
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
