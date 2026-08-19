#!/usr/bin/env python
"""Multi-model comparison: train each model on nested OOF utility labels and evaluate."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from decision.model_protocol import (
    DecisionModelSpec,
    extended_model_specs,
    decision_scores,
)
from decision.cluster_weighting import WeightedMedianImputer, cluster_balanced_row_weights


# ── Elsevier style ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.4,
    "grid.linewidth": 0.3,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
})

# Colour palette (colourblind-safe)
COLORS = {
    "lda_classifier": "#1b9e77",
    "logistic_regression_classifier": "#d95f02",
    "ridge_regression": "#7570b3",
    "random_forest_classifier": "#e7298a",
    "random_forest_regressor": "#66a61e",
    "mlp_classifier": "#e6ab02",
    "mlp_regressor": "#a6761d",
    "svm_rbf_classifier": "#666666",
    "svm_rbf_regressor": "#1f78b4",
    "knn_classifier": "#b2df8a",
    "knn_regressor": "#fb9a99",
    "linear_regression": "#fdbf6f",
    "kernel_ridge_regressor": "#cab2d6",
    "xgboost_classifier": "#ff7f00",
    "xgboost_regressor": "#e31a1c",
}

MARKERS = {
    "classification": "o",
    "regression": "s",
}


def load_data():
    """Load utility labels from the selection reference + query features + behavior features."""
    sr = pq.read_table(
        "results/pilot_v2/selection_reference/selection_reference.parquet"
    ).to_pandas()

    # Load query features (14 landscape descriptors)
    qf = pq.read_table(
        "results/landscape_queries/features/lhs_50d/bbob_train/features.parquet"
    ).to_pandas()

    import json
    descriptor_cols = json.loads(qf["query_feature_columns"].iloc[0])
    descriptor_cols = [c for c in descriptor_cols if c in qf.columns]

    # Join descriptor features into selection reference
    join_key_qf = ["split", "problem_id", "function_id", "family", "dimension"]
    sr = sr.merge(
        qf[join_key_qf + descriptor_cols],
        on=join_key_qf,
        how="left",
        suffixes=("", "_feat"),
    )

    # Load behavior features (34 bf_ columns) from behavior files
    bf_frames = []
    for f in [1, 3, 15, 24]:
        bf = pq.read_table(
            f"results/phase1_pilot/bbob_train/bbob_f{f:03d}/dimension_10/behavior.parquet"
        ).to_pandas()
        # Only need one row per (algorithm, seed, FE) — take the prefix algorithm's row
        # The behavior file has rows for the prefix algorithm at each sampling point
        # We need to join on (problem_id, prefix_algorithm, seed, FE)
        bf_cols = [c for c in bf.columns if c.startswith("bf_")]
        keep_cols = ["problem_id", "function_id", "family", "cv_group_id",
                      "dimension", "algorithm", "seed", "FE"] + bf_cols
        bf_frames.append(bf[keep_cols])
    bf_all = pd.concat(bf_frames, ignore_index=True)

    # Join behavior features: selection_reference.prefix_algorithm == behavior.algorithm
    # selection_reference has prefix_algorithm, behavior has algorithm
    join_key_bf = ["problem_id", "function_id", "family", "cv_group_id",
                   "dimension", "algorithm", "seed", "FE"]
    # Rename prefix_algorithm to algorithm for join
    sr_for_join = sr.copy()
    sr_for_join["_algorithm"] = sr_for_join["prefix_algorithm"].astype(str)

    # Add algorithm column to sr for join
    sr["_join_algo"] = sr["prefix_algorithm"].astype(str)
    bf_all["_join_algo"] = bf_all["algorithm"].astype(str)

    join_key_bf_actual = ["problem_id", "function_id", "family", "cv_group_id",
                          "dimension", "_join_algo", "seed", "FE"]
    bf_cols = [c for c in bf_all.columns if c.startswith("bf_")]
    sr = sr.merge(
        bf_all[join_key_bf_actual + bf_cols],
        on=join_key_bf_actual,
        how="left",
        suffixes=("", "_bf"),
    )
    sr = sr.drop(columns=["_join_algo"])
    print(f"Behavior features joined: {len([c for c in bf_cols if c in sr.columns])} columns")

    # Combine all feature columns: descriptor + behavior + budget
    feature_cols = descriptor_cols + bf_cols
    extra_features = ["FE_prefix", "FE_total", "remaining_budget_ratio"]
    for c in extra_features:
        if c in sr.columns and c not in feature_cols:
            feature_cols.append(c)

    # Mandatory feature count check (see AGENTS.md §6.1, PROJECT_HANDOFF.md §13.1)
    assert len(feature_cols) >= 45, f"Expected >=45 features (B3), got {len(feature_cols)}"
    assert any(c.startswith("bf_") for c in feature_cols), "Missing behavior features"
    assert any(c.startswith("descriptor_") for c in feature_cols), "Missing descriptor features"

    found = [c for c in feature_cols if c in sr.columns]
    print(f"Total features: {len(found)} (descriptor={len(descriptor_cols)}, behavior={len(bf_cols)}, extra={len(extra_features)})")

    # Target: G_FE (compute from p_skip and p_query)
    benchmark = sr["benchmark_reference_value"].astype(float).to_numpy()
    p_skip_raw = sr["p_skip_raw"].astype(float).to_numpy()
    p_query_raw = sr["p_query_raw"].astype(float).to_numpy()

    e_skip = np.maximum(p_skip_raw - benchmark, 0.0)
    e_query = np.maximum(p_query_raw - benchmark, 0.0)
    eps = 0.01 * np.maximum(e_skip, 1.0)
    g_fe = np.log((e_skip + eps) / (e_query + eps))

    # Binary label: need_query = g_fe > 0
    need_query = (g_fe > 0).astype(int)

    # Runtime info
    runtime_skip = sr["runtime_no_query_optimization"].astype(float).to_numpy()
    runtime_query = (
        sr["runtime_query"].astype(float).to_numpy()
        + sr["runtime_selection"].astype(float).to_numpy()
        + sr["runtime_handoff"].astype(float).to_numpy()
        + sr["runtime_selected_action_optimization"].astype(float).to_numpy()
    )

    # Terminal gaps
    gap_skip = sr["loss_skip"].astype(float).to_numpy()
    gap_query = sr["selected_action_loss"].astype(float).to_numpy()
    gap_vbs = sr["best_observed_loss"].astype(float).to_numpy()

    # CV group for OOF
    cv_groups = sr["cv_group_id"].astype(str).to_numpy()

    data = {
        "features": sr[feature_cols].to_numpy(dtype=float),
        "feature_names": feature_cols,
        "g_fe": g_fe,
        "need_query": need_query,
        "gap_skip": gap_skip,
        "gap_query": gap_query,
        "gap_vbs": gap_vbs,
        "runtime_skip": runtime_skip,
        "runtime_query": runtime_query,
        "cv_groups": cv_groups,
        "fe_prefix": sr["FE_prefix"].astype(int).to_numpy(),
        "fe_total": sr["FE_total"].astype(int).to_numpy(),
        "fe_ratio": sr["FE_prefix"].astype(float).to_numpy() / sr["FE_total"].astype(float).to_numpy(),
    }
    return data


def run_model_comparison(data, random_seed=1701):
    """Train each model with leave-one-CV-group-out OOF and evaluate."""
    specs = extended_model_specs(random_seed)
    unique_groups = sorted(set(data["cv_groups"]))
    n_groups = len(unique_groups)

    X = data["features"]
    y_reg = data["g_fe"]
    y_cls = data["need_query"]
    cv_groups = data["cv_groups"]

    results = []

    for spec in specs:
        model_name = spec.model_name
        print(f"  Training {model_name} ({spec.objective})...")

        oof_scores = np.full(len(X), np.nan)
        oof_predictions = np.full(len(X), np.nan)

        for holdout_group in unique_groups:
            mask = cv_groups == holdout_group
            train_mask = ~mask
            test_mask = mask

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            X_train = X[train_mask]
            X_test = X[test_mask]

            if spec.objective == "classification":
                y_train = y_cls[train_mask]
                if len(np.unique(y_train)) < 2:
                    # Constant prediction
                    oof_scores[test_mask] = 0.5
                    oof_predictions[test_mask] = y_train[0]
                    continue
            else:
                y_train = y_reg[train_mask]

            try:
                model = clone(spec.estimator)
                model.fit(X_train, y_train)
                scores = decision_scores(model, X_test)
                oof_scores[test_mask] = scores

                if spec.objective == "classification":
                    oof_predictions[test_mask] = (scores > 0.5).astype(int)
                else:
                    oof_predictions[test_mask] = scores
            except Exception as e:
                print(f"    {model_name} failed on fold {holdout_group}: {e}")
                oof_scores[test_mask] = 0.0
                oof_predictions[test_mask] = 0.0

        # Evaluate: for each threshold, decide query vs skip
        # gap_selected = if threshold says "query": gap_query, else: gap_skip
        # runtime_selected = if threshold says "query": runtime_query, else: runtime_skip

        # For classification: prediction > 0.5 means "need query"
        # For regression: prediction > 0 means "need query" (since g_fe > 0)
        if spec.objective == "classification":
            need_query_pred = (oof_predictions > 0.5).astype(int)
        else:
            need_query_pred = (oof_predictions > 0).astype(int)

        # Apply decision: when need_query_pred=1, use query path; else skip path
        gap_selected = np.where(need_query_pred == 1, data["gap_query"], data["gap_skip"])
        runtime_selected = np.where(
            need_query_pred == 1, data["runtime_query"], data["runtime_skip"]
        )
        fe_selected = np.where(
            need_query_pred == 1,
            data["fe_prefix"] + 500 + (data["fe_total"] - data["fe_prefix"] - 500),
            data["fe_total"],
        )

        # Compute metrics
        valid = ~np.isnan(oof_scores)
        n_valid = valid.sum()

        # AUC for classification
        if spec.objective == "classification" and len(np.unique(y_cls[valid])) == 2:
            try:
                auc = roc_auc_score(y_cls[valid], oof_scores[valid])
            except Exception:
                auc = 0.5
        else:
            # For regression, compute correlation with binary label
            try:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(y_cls[valid], oof_scores[valid])
                auc = (corr + 1) / 2  # map to [0, 1]
            except Exception:
                auc = 0.5

        # G_FE of selected path
        benchmark = np.maximum(data["gap_skip"], 1e-30)
        e_skip = data["gap_skip"]
        e_selected = gap_selected
        eps_sel = 0.01 * np.maximum(e_skip, 1.0)
        g_fe_selected = np.log((e_skip + eps_sel) / (e_selected + eps_sel))

        # Precision / recall for need_query
        tp = int(np.sum((need_query_pred == 1) & (y_cls == 1)))
        fp = int(np.sum((need_query_pred == 1) & (y_cls == 0)))
        tn = int(np.sum((need_query_pred == 0) & (y_cls == 0)))
        fn = int(np.sum((need_query_pred == 0) & (y_cls == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        result = {
            "model_name": model_name,
            "model_family": spec.model_family,
            "objective": spec.objective,
            "n_valid": n_valid,
            "auc": float(auc),
            "gap_mean": float(np.mean(gap_selected)),
            "gap_median": float(np.median(gap_selected)),
            "g_fe_mean": float(np.mean(g_fe_selected)),
            "g_fe_positive_rate": float(np.mean(g_fe_selected > 0)),
            "runtime_mean": float(np.mean(runtime_selected)),
            "runtime_median": float(np.median(runtime_selected)),
            "n_query": int(np.sum(need_query_pred == 1)),
            "n_skip": int(np.sum(need_query_pred == 0)),
            "query_rate": float(np.mean(need_query_pred == 1)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "gap_skip_mean": float(np.mean(data["gap_skip"])),
            "gap_vbs_mean": float(np.mean(data["gap_vbs"])),
            "gap_improvement_vs_sbs": float(
                (np.mean(data["gap_skip"]) - np.mean(gap_selected)) / np.mean(data["gap_skip"]) * 100
            ),
            "gap_improvement_vs_vbs_fraction": float(
                (np.mean(data["gap_skip"]) - np.mean(gap_selected))
                / max(np.mean(data["gap_skip"]) - np.mean(data["gap_vbs"]), 1e-10)
            ),
        }
        results.append(result)

    return results


def plot_model_comparison(results, data, output_dir=Path("figures")):
    """Generate Elsevier-style comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)

    # ── Figure 1: Gap vs Runtime Pareto ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # SBS and VBS reference lines
    sbs_gap = float(np.mean(data["gap_skip"]))
    vbs_gap = float(np.mean(data["gap_vbs"]))
    sbs_rt = float(np.mean(data["runtime_skip"]))

    ax.axhline(y=sbs_gap, color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.text(ax.get_xlim()[0], sbs_gap + 0.3, "SBS", fontsize=6, color="gray", va="bottom")
    ax.axhline(y=vbs_gap, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.text(ax.get_xlim()[0], vbs_gap + 0.3, "VBS (upper bound)", fontsize=6, color="gray", va="bottom")

    for _, row in df.iterrows():
        color = COLORS.get(row["model_name"], "#333333")
        marker = MARKERS.get(row["objective"], "o")
        ax.scatter(
            row["runtime_mean"], row["gap_mean"],
            color=color, marker=marker, s=25, zorder=5,
            edgecolors="white", linewidths=0.3,
        )
        ax.annotate(
            row["model_name"].replace("_classifier", "_cls").replace("_regressor", "_reg").replace("_regression", "_reg"),
            (row["runtime_mean"], row["gap_mean"]),
            fontsize=4.5, color=color,
            xytext=(3, 3), textcoords="offset points",
        )

    ax.scatter(sbs_rt, sbs_gap, color="black", marker="^", s=30, zorder=5, edgecolors="white", linewidths=0.3)
    ax.annotate("SBS", (sbs_rt, sbs_gap), fontsize=6, color="black", xytext=(3, -8), textcoords="offset points")

    ax.set_xlabel("Mean wall-clock time (s)")
    ax.set_ylabel("Mean terminal gap")
    ax.set_title("Pareto: gap vs runtime")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "model_pareto.pdf")
    fig.savefig(output_dir / "model_pareto.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'model_pareto.pdf'}")

    # ── Figure 2: AUC and F1 bar chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    df_sorted = df.sort_values("auc", ascending=True)
    colors = [COLORS.get(n, "#333") for n in df_sorted["model_name"]]

    ax1.barh(df_sorted["model_name"], df_sorted["auc"], color=colors, height=0.6, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("AUC")
    ax1.set_title("(a) Need-query classification")
    ax1.axvline(x=0.5, color="gray", linestyle=":", linewidth=0.5)
    ax1.grid(True, axis="x", alpha=0.3)

    df_sorted2 = df.sort_values("f1", ascending=True)
    colors2 = [COLORS.get(n, "#333") for n in df_sorted2["model_name"]]
    ax2.barh(df_sorted2["model_name"], df_sorted2["f1"], color=colors2, height=0.6, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("F1")
    ax2.set_title("(b) Need-query F1")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(output_dir / "model_auc_f1.pdf")
    fig.savefig(output_dir / "model_auc_f1.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'model_auc_f1.pdf'}")

    # ── Figure 3: Gap improvement bar chart ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    df_sorted3 = df.sort_values("gap_mean")
    colors3 = [COLORS.get(n, "#333") for n in df_sorted3["model_name"]]

    bars = ax.barh(df_sorted3["model_name"], df_sorted3["gap_mean"], color=colors3, height=0.6, edgecolor="white", linewidth=0.3)
    ax.axvline(x=sbs_gap, color="gray", linestyle=":", linewidth=0.5)
    ax.text(sbs_gap, -0.3, f"SBS={sbs_gap:.1f}", fontsize=5, color="gray", ha="center")
    ax.axvline(x=vbs_gap, color="gray", linestyle="--", linewidth=0.5)
    ax.text(vbs_gap, -0.3, f"VBS={vbs_gap:.1f}", fontsize=5, color="gray", ha="center")

    ax.set_xlabel("Mean terminal gap")
    ax.set_title("Gap by model")
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(output_dir / "model_gap_bar.pdf")
    fig.savefig(output_dir / "model_gap_bar.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'model_gap_bar.pdf'}")

    # ── Figure 4: Query rate vs gap improvement ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for _, row in df.iterrows():
        color = COLORS.get(row["model_name"], "#333")
        marker = MARKERS.get(row["objective"], "o")
        ax.scatter(
            row["query_rate"] * 100, row["gap_improvement_vs_sbs"],
            color=color, marker=marker, s=25, zorder=5,
            edgecolors="white", linewidths=0.3,
        )
        ax.annotate(
            row["model_name"].replace("_classifier", "_cls").replace("_regressor", "_reg").replace("_regression", "_reg"),
            (row["query_rate"] * 100, row["gap_improvement_vs_sbs"]),
            fontsize=4.5, color=color,
            xytext=(3, 3), textcoords="offset points",
        )

    ax.set_xlabel("Query rate (%)")
    ax.set_ylabel("Gap improvement vs SBS (%)")
    ax.set_title("Query rate vs improvement")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "model_query_vs_improvement.pdf")
    fig.savefig(output_dir / "model_query_vs_improvement.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'model_query_vs_improvement.pdf'}")

    # ── Figure 5: Comprehensive table-style summary ──
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # (a) AUC
    ax = axes[0, 0]
    df_s = df.sort_values("auc", ascending=True)
    ax.barh(df_s["model_name"], df_s["auc"], color=[COLORS.get(n, "#333") for n in df_s["model_name"]], height=0.6, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("AUC")
    ax.set_title("(a) Classification AUC")
    ax.grid(True, axis="x", alpha=0.3)

    # (b) Gap
    ax = axes[0, 1]
    df_s = df.sort_values("gap_mean")
    ax.barh(df_s["model_name"], df_s["gap_mean"], color=[COLORS.get(n, "#333") for n in df_s["model_name"]], height=0.6, edgecolor="white", linewidth=0.3)
    ax.axvline(x=sbs_gap, color="gray", linestyle=":", linewidth=0.5)
    ax.axvline(x=vbs_gap, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Mean terminal gap")
    ax.set_title("(b) Terminal gap")
    ax.grid(True, axis="x", alpha=0.3)

    # (c) F1
    ax = axes[1, 0]
    df_s = df.sort_values("f1", ascending=True)
    ax.barh(df_s["model_name"], df_s["f1"], color=[COLORS.get(n, "#333") for n in df_s["model_name"]], height=0.6, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("F1")
    ax.set_title("(c) Need-query F1")
    ax.grid(True, axis="x", alpha=0.3)

    # (d) Runtime vs gap scatter
    ax = axes[1, 1]
    for _, row in df.iterrows():
        color = COLORS.get(row["model_name"], "#333")
        marker = MARKERS.get(row["objective"], "o")
        ax.scatter(row["runtime_mean"], row["gap_mean"], color=color, marker=marker, s=20, zorder=5, edgecolors="white", linewidths=0.3)
    ax.axhline(y=sbs_gap, color="gray", linestyle=":", linewidth=0.5)
    ax.axhline(y=vbs_gap, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Mean runtime (s)")
    ax.set_ylabel("Mean terminal gap")
    ax.set_title("(d) Pareto: gap vs time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    fig.savefig(output_dir / "model_comparison_summary.pdf")
    fig.savefig(output_dir / "model_comparison_summary.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'model_comparison_summary.pdf'}")


def main():
    print("=" * 70)
    print("Multi-Model Comparison")
    print("=" * 70)

    data = load_data()
    print(f"Loaded {len(data['features'])} states, {len(data['feature_names'])} features")
    print(f"Need-query rate: {data['need_query'].mean()*100:.1f}%")
    print(f"SBS gap mean: {np.mean(data['gap_skip']):.4f}")
    print(f"VBS gap mean: {np.mean(data['gap_vbs']):.4f}")
    print()

    results = run_model_comparison(data, random_seed=1701)

    # Print table
    print(f"\n{'Model':<35} {'Obj':<5} {'AUC':<6} {'F1':<6} {'Gap':<8} {'Impr%':<7} {'QryR%':<6} {'Time(s)':<8} {'VBS%':<6}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["gap_mean"]):
        print(
            f"{r['model_name']:<35} {r['objective'][:4]:<5} "
            f"{r['auc']:<6.3f} {r['f1']:<6.3f} "
            f"{r['gap_mean']:<8.2f} {r['gap_improvement_vs_sbs']:<7.1f} "
            f"{r['query_rate']*100:<6.1f} {r['runtime_mean']:<8.4f} "
            f"{r['gap_improvement_vs_vbs_fraction']*100:<6.1f}"
        )

    # SBS reference
    sbs_gap = float(np.mean(data["gap_skip"]))
    vbs_gap = float(np.mean(data["gap_vbs"]))
    print(f"\nSBS gap: {sbs_gap:.4f}")
    print(f"VBS gap: {vbs_gap:.4f}")
    print(f"Max possible improvement: {(sbs_gap - vbs_gap) / sbs_gap * 100:.1f}%")

    import json
    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open("results/pilot_v2/model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print("\nSaved results/pilot_v2/model_comparison_results.json")

    # Generate figures
    plot_model_comparison(results, data)


if __name__ == "__main__":
    main()
