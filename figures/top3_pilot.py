#!/usr/bin/env python
"""Top-3 decision model pilot: RF-Reg, RF-Cls, LDA — deep OOF analysis and Elsevier figures."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision.cluster_weighting import (
    WeightedLinearDiscriminantAnalysis,
    WeightedMedianImputer,
)
from decision.model_protocol import decision_scores

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

COLORS = {
    "RF-Reg": "#1b9e77",
    "RF-Cls": "#d95f02",
    "LDA":    "#7570b3",
    "SBS":    "#666666",
    "VBS":    "#e31a1c",
}

MARKERS = {"RF-Reg": "s", "RF-Cls": "o", "LDA": "D", "SBS": "^", "VBS": "*"}


def load_data():
    sr = pq.read_table("results/pilot_v2/selection_reference/selection_reference.parquet").to_pandas()

    # Load descriptor features
    qf = pq.read_table("results/landscape_queries/features/lhs_50d/bbob_train/features.parquet").to_pandas()
    descriptor_cols = json.loads(qf["query_feature_columns"].iloc[0])
    descriptor_cols = [c for c in descriptor_cols if c in qf.columns]
    join_key_qf = ["split", "problem_id", "function_id", "family", "dimension"]
    sr = sr.merge(qf[join_key_qf + descriptor_cols], on=join_key_qf, how="left", suffixes=("", "_feat"))

    # Load behavior features
    bf_frames = []
    for f in [1, 3, 15, 24]:
        bf = pq.read_table(f"results/phase1_pilot/bbob_train/bbob_f{f:03d}/dimension_10/behavior.parquet").to_pandas()
        bf_cols = [c for c in bf.columns if c.startswith("bf_")]
        keep = ["problem_id", "function_id", "family", "cv_group_id", "dimension", "algorithm", "seed", "FE"] + bf_cols
        bf_frames.append(bf[keep])
    bf_all = pd.concat(bf_frames, ignore_index=True)
    sr["_join_algo"] = sr["prefix_algorithm"].astype(str)
    bf_all["_join_algo"] = bf_all["algorithm"].astype(str)
    join_key_bf = ["problem_id", "function_id", "family", "cv_group_id", "dimension", "_join_algo", "seed", "FE"]
    bf_cols = [c for c in bf_all.columns if c.startswith("bf_")]
    sr = sr.merge(bf_all[join_key_bf + bf_cols], on=join_key_bf, how="left", suffixes=("", "_bf"))
    sr = sr.drop(columns=["_join_algo"])

    feature_cols = descriptor_cols + bf_cols
    for c in ["FE_prefix", "FE_total", "remaining_budget_ratio"]:
        if c in sr.columns and c not in feature_cols:
            feature_cols.append(c)

    assert len(feature_cols) >= 45, f"Expected >=45 features, got {len(feature_cols)}"
    assert any(c.startswith("bf_") for c in feature_cols)
    assert any(c.startswith("descriptor_") for c in feature_cols)

    benchmark = sr["benchmark_reference_value"].astype(float).to_numpy()
    p_skip = sr["p_skip_raw"].astype(float).to_numpy()
    p_query = sr["p_query_raw"].astype(float).to_numpy()
    e_skip = np.maximum(p_skip - benchmark, 0.0)
    e_query = np.maximum(p_query - benchmark, 0.0)
    eps = 0.01 * np.maximum(e_skip, 1.0)
    g_fe = np.log((e_skip + eps) / (e_query + eps))
    need_query = (g_fe > 0).astype(int)

    runtime_skip = sr["runtime_no_query_optimization"].astype(float).to_numpy()
    runtime_query = (
        sr["runtime_query"].astype(float).to_numpy()
        + sr["runtime_selection"].astype(float).to_numpy()
        + sr["runtime_handoff"].astype(float).to_numpy()
        + sr["runtime_selected_action_optimization"].astype(float).to_numpy()
    )

    return {
        "sr": sr,
        "features": sr[feature_cols].to_numpy(dtype=float),
        "feature_names": feature_cols,
        "g_fe": g_fe,
        "need_query": need_query,
        "gap_skip": sr["loss_skip"].astype(float).to_numpy(),
        "gap_query": sr["selected_action_loss"].astype(float).to_numpy(),
        "gap_vbs": sr["best_observed_loss"].astype(float).to_numpy(),
        "runtime_skip": runtime_skip,
        "runtime_query": runtime_query,
        "cv_groups": sr["cv_group_id"].astype(str).to_numpy(),
        "function_ids": sr["function_id"].astype(str).to_numpy(),
        "prefix_algos": sr["prefix_algorithm"].astype(str).to_numpy(),
        "first_hit_skip": sr["skip_first_hit_FE"].to_numpy(dtype=float),
        "first_hit_query": sr["selected_action_first_hit_FE"].to_numpy(dtype=float),
        "skip_success": sr["skip_success"].astype(bool).to_numpy(),
        "query_success": sr["selected_action_success"].astype(bool).to_numpy(),
    }


def make_models(seed=1701):
    xgb_seed = int(np.random.SeedSequence([seed, 20260811, 99]).generate_state(1, dtype=np.uint32)[0])
    return {
        "RF-Reg": Pipeline([
            ("imputer", WeightedMedianImputer()),
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=200, max_depth=8, max_features="sqrt", random_state=xgb_seed, n_jobs=1)),
        ]),
        "RF-Cls": Pipeline([
            ("imputer", WeightedMedianImputer()),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=200, max_depth=8, max_features="sqrt", random_state=xgb_seed, n_jobs=1)),
        ]),
        "LDA": Pipeline([
            ("imputer", WeightedMedianImputer()),
            ("scaler", StandardScaler()),
            ("classifier", WeightedLinearDiscriminantAnalysis()),
        ]),
    }


def run_oof(data, models):
    X = data["features"]
    y_reg = data["g_fe"]
    y_cls = data["need_query"]
    groups = data["cv_groups"]
    unique_groups = sorted(set(groups))

    results = {}
    for name, base_model in models.items():
        print(f"  OOF training: {name}")
        oof_pred = np.full(len(X), np.nan)
        oof_score = np.full(len(X), np.nan)
        is_cls = name in ("RF-Cls", "LDA")

        for holdout in unique_groups:
            mask = groups == holdout
            train_mask = ~mask
            test_mask = mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            X_train, X_test = X[train_mask], X[test_mask]
            y_train = y_cls[train_mask] if is_cls else y_reg[train_mask]
            if is_cls and len(np.unique(y_train)) < 2:
                oof_pred[test_mask] = 0
                oof_score[test_mask] = 0.5
                continue
            try:
                model = clone(base_model)
                model.fit(X_train, y_train)
                scores = decision_scores(model, X_test)
                oof_score[test_mask] = scores
                if is_cls:
                    oof_pred[test_mask] = (scores > 0.5).astype(int)
                else:
                    oof_pred[test_mask] = scores
            except Exception as e:
                print(f"    {name} failed on {holdout}: {e}")
                oof_pred[test_mask] = 0
                oof_score[test_mask] = 0.0

        if is_cls:
            trigger = (oof_pred > 0.5).astype(int)
        else:
            trigger = (oof_pred > 0).astype(int)

        gap_sel = np.where(trigger == 1, data["gap_query"], data["gap_skip"])
        rt_sel = np.where(trigger == 1, data["runtime_query"], data["runtime_skip"])
        fh_sel = np.where(trigger == 1, data["first_hit_query"], data["first_hit_skip"])
        succ_sel = np.where(trigger == 1, data["query_success"], data["skip_success"])

        results[name] = {
            "trigger": trigger,
            "oof_score": oof_score,
            "gap_sel": gap_sel,
            "rt_sel": rt_sel,
            "fh_sel": fh_sel,
            "succ_sel": succ_sel,
        }
    return results


def plot_figures(data, results, outdir=Path("figures")):
    outdir.mkdir(parents=True, exist_ok=True)
    sr = data["sr"]
    sbs_gap = float(np.mean(data["gap_skip"]))
    vbs_gap = float(np.mean(data["gap_vbs"]))
    sbs_rt = float(np.mean(data["runtime_skip"]))

    model_names = ["RF-Reg", "RF-Cls", "LDA"]

    # ── Figure 1: 4-panel summary ──
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # (a) Mean gap bar
    ax = axes[0, 0]
    names = model_names + ["SBS", "VBS"]
    gaps = [float(np.mean(results[n]["gap_sel"])) for n in model_names] + [sbs_gap, vbs_gap]
    colors = [COLORS[n] for n in names]
    ax.barh(names, gaps, color=colors, height=0.6, edgecolor="white", linewidth=0.3)
    ax.axvline(x=sbs_gap, color="gray", linestyle=":", linewidth=0.5)
    ax.axvline(x=vbs_gap, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Mean terminal gap")
    ax.set_title("(a) Terminal gap")
    ax.grid(True, axis="x", alpha=0.3)

    # (b) Query rate vs improvement scatter
    ax = axes[0, 1]
    for n in model_names:
        qr = float(np.mean(results[n]["trigger"]))
        impr = (sbs_gap - float(np.mean(results[n]["gap_sel"]))) / sbs_gap * 100
        ax.scatter(qr * 100, impr, color=COLORS[n], marker=MARKERS[n], s=30, zorder=5, edgecolors="white", linewidths=0.3)
        ax.annotate(n, (qr * 100, impr), fontsize=6, color=COLORS[n], xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Query trigger rate (%)")
    ax.set_ylabel("Gap improvement vs SBS (%)")
    ax.set_title("(b) Query rate vs improvement")
    ax.grid(True, alpha=0.3)

    # (c) Gap CDF
    ax = axes[1, 0]
    for n in model_names:
        gaps_sorted = np.sort(results[n]["gap_sel"])
        cdf = np.arange(1, len(gaps_sorted) + 1) / len(gaps_sorted)
        ax.plot(gaps_sorted, cdf, color=COLORS[n], linewidth=0.8, label=n)
    ax.plot(np.sort(data["gap_skip"]), np.arange(1, len(data["gap_skip"]) + 1) / len(data["gap_skip"]),
            color=COLORS["SBS"], linestyle=":", linewidth=0.6, label="SBS")
    ax.plot(np.sort(data["gap_vbs"]), np.arange(1, len(data["gap_vbs"]) + 1) / len(data["gap_vbs"]),
            color=COLORS["VBS"], linestyle="--", linewidth=0.6, label="VBS (upper bound)")
    ax.set_xlabel("Terminal gap")
    ax.set_ylabel("CDF")
    ax.set_title("(c) Gap distribution")
    ax.legend(loc="lower right", framealpha=0.8, edgecolor="none")
    ax.grid(True, alpha=0.3)

    # (d) Runtime vs gap Pareto
    ax = axes[1, 1]
    for n in model_names:
        ax.scatter(float(np.mean(results[n]["rt_sel"])), float(np.mean(results[n]["gap_sel"])),
                    color=COLORS[n], marker=MARKERS[n], s=30, zorder=5, edgecolors="white", linewidths=0.3)
        ax.annotate(n, (float(np.mean(results[n]["rt_sel"])), float(np.mean(results[n]["gap_sel"]))),
                    fontsize=6, color=COLORS[n], xytext=(4, 4), textcoords="offset points")
    ax.scatter(sbs_rt, sbs_gap, color=COLORS["SBS"], marker=MARKERS["SBS"], s=30, zorder=5, edgecolors="white", linewidths=0.3)
    ax.annotate("SBS", (sbs_rt, sbs_gap), fontsize=6, color=COLORS["SBS"], xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("Mean wall-clock time (s)")
    ax.set_ylabel("Mean terminal gap")
    ax.set_title("(d) Pareto: gap vs time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    fig.savefig(outdir / "top3_summary.pdf")
    fig.savefig(outdir / "top3_summary.png", dpi=300)
    plt.close(fig)
    print(f"Saved {outdir / 'top3_summary.pdf'}")

    # ── Figure 2: By function ──
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    funcs = sorted(set(data["function_ids"]))
    x = np.arange(len(funcs))
    width = 0.15

    ax = axes[0]
    for i, n in enumerate(model_names + ["SBS", "VBS"]):
        vals = []
        for f in funcs:
            mask = data["function_ids"] == f
            if n in results:
                vals.append(float(np.mean(results[n]["gap_sel"][mask])))
            elif n == "SBS":
                vals.append(float(np.mean(data["gap_skip"][mask])))
            elif n == "VBS":
                vals.append(float(np.mean(data["gap_vbs"][mask])))
        ax.bar(x + i * width, vals, width, color=COLORS[n], label=n, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f.replace("bbob_", "") for f in funcs], fontsize=7)
    ax.set_ylabel("Mean terminal gap")
    ax.set_title("(a) Gap by function")
    ax.legend(fontsize=5, ncol=3, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    # (b) By prefix algorithm
    ax = axes[1]
    prefixes = sorted(set(data["prefix_algos"]))
    x = np.arange(len(prefixes))
    for i, n in enumerate(model_names + ["SBS", "VBS"]):
        vals = []
        for p in prefixes:
            mask = data["prefix_algos"] == p
            if n in results:
                vals.append(float(np.mean(results[n]["gap_sel"][mask])))
            elif n == "SBS":
                vals.append(float(np.mean(data["gap_skip"][mask])))
            elif n == "VBS":
                vals.append(float(np.mean(data["gap_vbs"][mask])))
        ax.bar(x + i * width, vals, width, color=COLORS[n], label=n, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(prefixes, fontsize=7)
    ax.set_ylabel("Mean terminal gap")
    ax.set_title("(b) Gap by prefix algorithm")
    ax.legend(fontsize=5, ncol=3, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(outdir / "top3_by_group.pdf")
    fig.savefig(outdir / "top3_by_group.png", dpi=300)
    plt.close(fig)
    print(f"Saved {outdir / 'top3_by_group.pdf'}")

    # ── Figure 3: Per-fold OOF gap ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    groups = sorted(set(data["cv_groups"]))
    x = np.arange(len(groups))
    width = 0.2

    for i, n in enumerate(model_names):
        vals = []
        for g in groups:
            mask = data["cv_groups"] == g
            vals.append(float(np.mean(results[n]["gap_sel"][mask])))
        ax.bar(x + i * width, vals, width, color=COLORS[n], label=n, edgecolor="white", linewidth=0.3)
    # SBS reference
    sbs_per = [float(np.mean(data["gap_skip"][data["cv_groups"] == g])) for g in groups]
    ax.plot(x + width, sbs_per, color=COLORS["SBS"], linestyle=":", linewidth=0.8, label="SBS")
    vbs_per = [float(np.mean(data["gap_vbs"][data["cv_groups"] == g])) for g in groups]
    ax.plot(x + width, vbs_per, color=COLORS["VBS"], linestyle="--", linewidth=0.8, label="VBS")

    ax.set_xticks(x + width)
    ax.set_xticklabels([g.replace("bbob_", "") for g in groups], fontsize=7)
    ax.set_ylabel("Mean terminal gap")
    ax.set_title("OOF gap by CV group (holdout)")
    ax.legend(fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(outdir / "top3_per_fold.pdf")
    fig.savefig(outdir / "top3_per_fold.png", dpi=300)
    plt.close(fig)
    print(f"Saved {outdir / 'top3_per_fold.pdf'}")

    # ── Figure 4: Convergence — first-hit FE & time ──
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    for n in model_names + ["SBS", "VBS"]:
        if n in results:
            fh = results[n]["fh_sel"]
            succ = results[n]["succ_sel"]
        elif n == "SBS":
            fh = data["first_hit_skip"]
            succ = data["skip_success"]
        elif n == "VBS":
            # VBS: best first_hit among 4 algorithms per state
            fh_cols = [f"first_hit_FE_{a}" for a in ["de", "pso", "cmaes", "shade"]]
            # Not available per-algo in SR; use query_path first_hit as proxy
            fh = data["first_hit_query"]
            succ = data["query_success"]
        valid = succ & ~np.isnan(fh) & (fh > 0)
        if valid.sum() > 0:
            fh_sorted = np.sort(fh[valid])
            cdf = np.arange(1, len(fh_sorted) + 1) / len(fh_sorted)
            style = "-" if n in model_names else (":" if n == "SBS" else "--")
            lw = 0.8 if n in model_names else 0.6
            ax.plot(fh_sorted, cdf, color=COLORS[n], linestyle=style, linewidth=lw, label=n)
    ax.set_xlabel("First-hit FE")
    ax.set_ylabel("CDF (successful states only)")
    ax.set_title("(a) Convergence speed (FE)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for n in model_names + ["SBS", "VBS"]:
        if n in results:
            gaps_arr = results[n]["gap_sel"]
            rts_arr = results[n]["rt_sel"]
        elif n == "SBS":
            gaps_arr = data["gap_skip"]
            rts_arr = data["runtime_skip"]
        elif n == "VBS":
            # VBS gap per state = min of 4 algo gaps; runtime = skip runtime (oracle has no overhead)
            gaps_arr = data["gap_vbs"]
            rts_arr = data["runtime_skip"]
        # Scatter gap vs time
        ax.scatter(float(np.mean(rts_arr)), float(np.mean(gaps_arr)),
                    color=COLORS[n], marker=MARKERS.get(n, "o"), s=30, zorder=5,
                    edgecolors="white", linewidths=0.3)
        ax.annotate(n, (float(np.mean(rts_arr)), float(np.mean(gaps_arr))),
                    fontsize=6, color=COLORS[n], xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean wall-clock time (s)")
    ax.set_ylabel("Mean terminal gap")
    ax.set_title("(b) Pareto: gap vs time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(outdir / "top3_convergence.pdf")
    fig.savefig(outdir / "top3_convergence.png", dpi=300)
    plt.close(fig)
    print(f"Saved {outdir / 'top3_convergence.pdf'}")


def main():
    print("=" * 70)
    print("Top-3 Decision Model Pilot (RF-Reg, RF-Cls, LDA)")
    print("=" * 70)

    data = load_data()
    print(f"States: {len(data['features'])}, Features: {len(data['feature_names'])}")
    print(f"Need-query rate: {data['need_query'].mean()*100:.1f}%")
    print(f"SBS gap: {np.mean(data['gap_skip']):.4f}")
    print(f"VBS gap: {np.mean(data['gap_vbs']):.4f}")
    print()

    models = make_models()
    results = run_oof(data, models)

    # Print summary table
    print(f"\n{'Model':<10} {'Gap':<8} {'Impr%':<7} {'QryR%':<7} {'Time(s)':<9} {'VBS%':<6} {'F1-prec':<8} {'F1-rec':<8} {'F1':<6}")
    print("-" * 80)
    sbs_gap = float(np.mean(data["gap_skip"]))
    vbs_gap = float(np.mean(data["gap_vbs"]))
    for n in ["RF-Reg", "RF-Cls", "LDA"]:
        r = results[n]
        gap = float(np.mean(r["gap_sel"]))
        impr = (sbs_gap - gap) / sbs_gap * 100
        qr = float(np.mean(r["trigger"]))
        rt = float(np.mean(r["rt_sel"]))
        vbs_pct = (sbs_gap - gap) / max(sbs_gap - vbs_gap, 1e-10) * 100
        tp = int(np.sum((r["trigger"] == 1) & (data["need_query"] == 1)))
        fp = int(np.sum((r["trigger"] == 1) & (data["need_query"] == 0)))
        tn = int(np.sum((r["trigger"] == 0) & (data["need_query"] == 0)))
        fn = int(np.sum((r["trigger"] == 0) & (data["need_query"] == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        print(f"{n:<10} {gap:<8.2f} {impr:<7.1f} {qr*100:<7.1f} {rt:<9.4f} {vbs_pct:<6.1f} {prec:<8.3f} {rec:<8.3f} {f1:<6.3f}")
    print(f"{'SBS':<10} {sbs_gap:<8.2f} {'0.0':<7} {'0.0':<7} {float(np.mean(data['runtime_skip'])):<9.4f} {'0.0':<6}")
    print(f"{'VBS':<10} {vbs_gap:<8.2f} {44.6:<7.1f} {'—':<7} {'—':<9} {100.0:<6.1f}")

    # By function
    print(f"\nBy function:")
    print(f"{'Function':<12}", end="")
    for n in ["RF-Reg", "RF-Cls", "LDA", "SBS", "VBS"]:
        print(f" {n:<10}", end="")
    print()
    for f in sorted(set(data["function_ids"])):
        mask = data["function_ids"] == f
        print(f"{f.replace('bbob_', ''):<12}", end="")
        for n in ["RF-Reg", "RF-Cls", "LDA"]:
            print(f" {np.mean(results[n]['gap_sel'][mask]):<10.2f}", end="")
        print(f" {np.mean(data['gap_skip'][mask]):<10.2f}", end="")
        print(f" {np.mean(data['gap_vbs'][mask]):<10.2f}")

    # By prefix
    print(f"\nBy prefix algorithm:")
    print(f"{'Prefix':<10}", end="")
    for n in ["RF-Reg", "RF-Cls", "LDA", "SBS", "VBS"]:
        print(f" {n:<10}", end="")
    print()
    for p in sorted(set(data["prefix_algos"])):
        mask = data["prefix_algos"] == p
        print(f"{p:<10}", end="")
        for n in ["RF-Reg", "RF-Cls", "LDA"]:
            print(f" {np.mean(results[n]['gap_sel'][mask]):<10.2f}", end="")
        print(f" {np.mean(data['gap_skip'][mask]):<10.2f}", end="")
        print(f" {np.mean(data['gap_vbs'][mask]):<10.2f}")

    # Per fold
    print(f"\nPer-fold OOF gap (holdout group):")
    print(f"{'Group':<12}", end="")
    for n in ["RF-Reg", "RF-Cls", "LDA", "SBS", "VBS"]:
        print(f" {n:<10}", end="")
    print()
    for g in sorted(set(data["cv_groups"])):
        mask = data["cv_groups"] == g
        print(f"{g.replace('bbob_', ''):<12}", end="")
        for n in ["RF-Reg", "RF-Cls", "LDA"]:
            print(f" {np.mean(results[n]['gap_sel'][mask]):<10.2f}", end="")
        print(f" {np.mean(data['gap_skip'][mask]):<10.2f}", end="")
        print(f" {np.mean(data['gap_vbs'][mask]):<10.2f}")

    plot_figures(data, results)
    print("\nDone.")


if __name__ == "__main__":
    main()
