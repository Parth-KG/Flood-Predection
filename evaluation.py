"""
evaluation.py — metrics summary, predicted-vs-observed, residual,
                and metric comparison bar charts.
"""

import pandas as pd
import matplotlib.pyplot as plt

from config import MODEL_COLORS


def build_summary(results: dict) -> pd.DataFrame:
    """Return a DataFrame with RMSE / MAE / R2 for each model, sorted by RMSE."""
    summary = pd.DataFrame({
        "Model": list(results.keys()),
        "RMSE":  [results[m]["rmse"] for m in results],
        "MAE":   [results[m]["mae"]  for m in results],
        "R2":    [results[m]["r2"]   for m in results],
    }).sort_values("RMSE").reset_index(drop=True)

    print("\nResults:")
    print(summary.to_string(index=False))
    return summary


def save_summary_csv(summary: pd.DataFrame, path: str = "model_comparison_summary.csv"):
    summary.to_csv(path, index=False)
    print(f"Summary saved to {path}")


def plot_predicted_vs_observed(
    results: dict,
    y_test,
    save_path: str = "predicted_vs_observed.png",
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        ax.scatter(y_test, res["preds"], alpha=0.45, s=18,
                   color=MODEL_COLORS.get(name, "purple"), edgecolors="none")
        lims = [
            min(y_test.min(), res["preds"].min()),
            max(y_test.max(), res["preds"].max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="1:1 line")
        ax.set_title(f"{name}\nRMSE={res['rmse']:.3f}  R2={res['r2']:.3f}")
        ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    axes[-1].set_visible(False)
    plt.suptitle("Predicted vs Observed", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_residuals(
    results: dict,
    save_path: str = "residual_plots.png",
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (name, res) in enumerate(results.items()):
        ax    = axes[i]
        resid = res["preds"] - res["preds"]   # placeholder — filled below
        # use actual residuals: observed - predicted (passed via results)
        # residuals are computed here so evaluation.py stays self-contained
        resid = res.get("residuals", res["preds"])   # fallback if key missing
        ax.scatter(res["preds"], resid, alpha=0.45, s=18,
                   color=MODEL_COLORS.get(name, "purple"), edgecolors="none")
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"{name} - Residuals")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residuals")
        ax.grid(alpha=0.3)

    axes[-1].set_visible(False)
    plt.suptitle("Residual Plots", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_metric_comparison(
    summary: pd.DataFrame,
    save_path: str = "model_comparison_metrics.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, color in zip(axes, ["RMSE", "MAE", "R2"],
                                 ["#2196F3", "#FF9800", "#4CAF50"]):
        bars = ax.bar(summary["Model"], summary[metric], color=color, alpha=0.85)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, summary[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=9,
            )

    plt.suptitle("Model Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def run_evaluation(results: dict, y_test):
    """Run the full evaluation pipeline and save all outputs."""
    # attach residuals to results so plot_residuals can use them
    for res in results.values():
        res["residuals"] = y_test - res["preds"]

    summary = build_summary(results)
    save_summary_csv(summary)
    plot_predicted_vs_observed(results, y_test)
    plot_residuals(results)
    plot_metric_comparison(summary)
