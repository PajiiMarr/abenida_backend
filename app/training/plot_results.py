"""Grouped horizontal bar chart of suite model performance.

Usage (from backend/):
    python -m app.training.plot_results                  # AUC -> outputs/phase3/model_comparison_auc.png
    python -m app.training.plot_results --metric f1
    python -m app.training.plot_results --metric all     # one chart per metric
    python -m app.training.plot_results --input outputs/phase3/suite_model_comparison.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHASE3 = Path(__file__).resolve().parents[2] / "outputs" / "phase3"
METRICS = {"accuracy": "Accuracy", "f1": "F1-Score", "auc": "AUC", "kappa": "Kappa"}
MODEL_LABELS = {
    "m1_gwpca_svm": "m1 GWPCA-SVM",
    "m2_gwr_mgwr": "m2 GWR/MGWR",
    "m3_gwpca_mars": "m3 GWPCA-MARS",
    "m4_stacking_all": "m4 Stacking",
    "m5_gwr_mars_svm_pipeline": "m5 GWR-MARS-SVM",
}
COLORS = ["#8c8c8c", "#2a9d8f", "#e9c46a", "#264653", "#e76f51"]


def load(path: Path) -> pd.DataFrame:
    """Read the summary table (.txt) or the comparison CSV into event/model/metric columns."""
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        ev = next(c for c in ("event", "category") if c in df.columns)
        df = df.rename(columns={ev: "event", "f1-score": "f1", "f1_score": "f1"})
        return df
    rows = []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 6 and cells[0] != "Event":
            rows.append([cells[0], cells[1], *map(float, cells[2:])])
    return pd.DataFrame(rows, columns=["event", "model", "accuracy", "f1", "auc", "kappa"])


def plot(df: pd.DataFrame, metric: str, out: Path) -> Path:
    pivot = df.pivot(index="event", columns="model", values=metric)
    models = [m for m in MODEL_LABELS if m in pivot.columns]
    # chronological events, then overall, then holdouts; first row is drawn on top
    order = ["oct_2022", "jan_2023", "jul_2024", "overall"]
    events = [e for e in order if e in pivot.index]
    events += sorted(e for e in pivot.index if e not in order and not e.startswith("holdout"))
    events += sorted(e for e in pivot.index if e.startswith("holdout"))
    pivot = pivot.loc[events, models]

    fig, ax = plt.subplots(figsize=(10, 1.3 * len(events) + 1))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        ys = [k + (i - (len(models) - 1) / 2) * h for k in range(len(events))]
        bars = ax.barh(ys, pivot[m], height=h * 0.92, color=COLORS[i], label=MODEL_LABELS[m])
        for b, v in zip(bars, pivot[m]):
            ax.text(v + (0.004 if v >= 0 else -0.004), b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7)

    ax.invert_yaxis()
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(events)
    ax.set_xlabel(METRICS[metric])
    ax.set_title(f"Model performance by event ({METRICS[metric]})", loc="left", fontweight="bold")
    if metric == "auc":
        ax.axvline(0.5, color="#999", lw=0.8, ls="--")
    ax.margins(x=0.08)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(models), frameon=False, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=PHASE3 / "model_performance_summary.txt")
    p.add_argument("--metric", choices=[*METRICS, "all"], default="auc")
    p.add_argument("--outdir", type=Path, default=PHASE3)
    a = p.parse_args()

    df = load(a.input)
    for m in (METRICS if a.metric == "all" else [a.metric]):
        print(plot(df, m, a.outdir / f"model_comparison_{m}.png"))


if __name__ == "__main__":
    main()