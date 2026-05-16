#!/usr/bin/env python3
"""Aggregate experiment JSON artifacts and generate comparison plots.

Examples:
  python3 scripts/plot_experiments.py outputs --outdir outputs/reports/all_runs
  python3 scripts/plot_experiments.py outputs/sweeps/_smoke_resnet18_k outputs/btad_vitb16_coreset0.0005.json --outdir outputs/reports/mixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.experiments import choose_primary_metric, load_experiment_rows


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _save_leaderboard(df: pd.DataFrame, metric: str, out_path: Path, top_n: int) -> None:
    cols = [c for c in ["dataset", "category", "backbone", "image_size", "coreset_ratio", metric, "total_s"] if c in df.columns]
    plot_df = df.sort_values(metric, ascending=False).head(top_n).iloc[::-1]
    labels = plot_df.apply(
        lambda row: " | ".join(str(row[c]) for c in ["dataset", "category", "backbone", "image_size"] if c in plot_df.columns and pd.notna(row[c])),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(plot_df))))
    ax.barh(labels, plot_df[metric], color="#247ba0")
    ax.set_xlabel(metric)
    ax.set_title(f"Top runs by {metric}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_frontier(df: pd.DataFrame, metric: str, time_key: str, out_path: Path) -> None:
    plot_df = df.dropna(subset=[metric, time_key]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for backbone, group in plot_df.groupby(plot_df.get("backbone", pd.Series(["unknown"] * len(plot_df)))):
        ax.scatter(group[time_key], group[metric], label=str(backbone), s=70, alpha=0.85)

    best = plot_df.sort_values(metric, ascending=False).head(min(8, len(plot_df)))
    for _, row in best.iterrows():
        label = f"{row.get('backbone', 'run')} @ {row.get('image_size', '?')}"
        ax.annotate(label, (row[time_key], row[metric]), fontsize=8, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(time_key)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {time_key}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_coreset_plot(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if "coreset_ratio" not in df.columns:
        return
    plot_df = df.dropna(subset=[metric, "coreset_ratio"]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for backbone, group in plot_df.groupby(plot_df.get("backbone", pd.Series(["unknown"] * len(plot_df)))):
        group = group.sort_values("coreset_ratio")
        ax.plot(group["coreset_ratio"], group[metric], marker="o", label=str(backbone))
    ax.set_xscale("log")
    ax.set_xlabel("coreset_ratio")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs coreset ratio")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON file(s) or directories containing experiment artifacts")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", default="auto", help="Primary metric to rank/plot; use 'auto' to infer")
    ap.add_argument("--time-key", default="total_s")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    rows = load_experiment_rows(list(args.inputs))
    if not rows:
        raise SystemExit("No experiment rows found in the provided inputs")

    df = pd.DataFrame(rows)
    df = _coerce_numeric(
        df,
        [
            "image_size",
            "coreset_ratio",
            "num_neighbors",
            "image_auroc",
            "pixel_auroc",
            "pro_auc",
            "total_s",
            "threshold_recall",
            "threshold_precision",
            "threshold_fpr",
            "threshold_alert_rate",
        ],
    )
    metric = choose_primary_metric(rows) if args.metric == "auto" else args.metric
    if metric not in df.columns:
        raise SystemExit(f"Metric column '{metric}' not found")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(metric, ascending=False)
    df.to_csv(outdir / "summary.csv", index=False)
    df.to_json(outdir / "summary.json", orient="records", indent=2)

    _save_leaderboard(df, metric, outdir / "leaderboard.png", int(args.top_n))
    _save_frontier(df, metric, args.time_key, outdir / "frontier.png")
    _save_coreset_plot(df, metric, outdir / "metric_vs_coreset.png")

    print(f"Wrote plots and summaries to {outdir}")


if __name__ == "__main__":
    main()
