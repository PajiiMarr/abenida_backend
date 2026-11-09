"""
plot_feature_importance.py
--------------------------
Visualise which features drive the flood model.

Usage:
    python plot_feature_importance.py
    python plot_feature_importance.py --model models/flood_model_rf_weighted.pkl
    python plot_feature_importance.py --csv outputs/phase2/feature_importance/feature_importance.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. Feature name lists (match your preprocessing/config.py)
# ----------------------------------------------------------------------
RAW_FEATURES = [
    "dem", "slope", "aspect", "twi", "hand",
    "flow_accumulation", "distance_to_river", "chirps", "gpm",
]

ENGINEERED_FEATURES = [
    "wetness_index", "flood_concentration", "twi_hand_ratio",
    "northness", "eastness", "relative_elevation", "slope_aspect",
    "river_proximity", "floodplain_index",
    "rainfall_intensity", "rainfall_ratio", "rainfall_wetness", "rainfall_flow",
    "twi_hand", "dem_river_ratio", "hand_dem_ratio", "flow_dist_ratio",
    "hand_squared",
]


# ----------------------------------------------------------------------
# 2. Load importance from a model or from an existing CSV
# ----------------------------------------------------------------------
def load_from_model(model_path: Path) -> pd.DataFrame:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Handle sklearn Pipeline / stacking / raw estimator
    est = model
    if hasattr(est, "named_steps"):
        # take the last step that has feature_importances_
        for step in reversed(list(est.named_steps.values())):
            if hasattr(step, "feature_importances_"):
                est = step
                break
    if hasattr(est, "estimators_"):          # e.g. RandomForest already fitted
        pass
    if not hasattr(est, "feature_importances_"):
        raise AttributeError(
            f"{model_path.name} has no feature_importances_. "
            "Use --csv instead, or load the RF sub-model directly."
        )

    imp = np.asarray(est.feature_importances_, dtype=float)

    # Try to recover feature names from the model itself, else fall back
    names = getattr(est, "feature_names_in_", None)
    if names is None:
        names_path = Path("outputs/phase2/preprocessed/feature_names_engineered.json")
        if names_path.exists():
            names = json.loads(names_path.read_text())
        else:
            all_feats = RAW_FEATURES + ENGINEERED_FEATURES
            if len(all_feats) == len(imp):
                names = all_feats
            else:
                names = [f"f{i}" for i in range(len(imp))]

    return pd.DataFrame({"feature": list(names), "importance": imp})


def load_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    fcol = "feature" if "feature" in df.columns else df.columns[0]
    icol = (
        "importance" if "importance" in df.columns
        else "mean_importance" if "mean_importance" in df.columns
        else df.columns[1]
    )
    df = df.rename(columns={fcol: "feature", icol: "importance"})
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    return df[["feature", "importance"]]


# ----------------------------------------------------------------------
# 3. Plot helpers
# ----------------------------------------------------------------------
def _barh(ax, df, title, color):
    df = df.sort_values("importance", ascending=True)
    bars = ax.barh(df["feature"], df["importance"], color=color, edgecolor="black", linewidth=0.4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # value labels
    max_imp = df["importance"].max() if len(df) else 1.0
    for b, v in zip(bars, df["importance"]):
        ax.text(
            b.get_width() + max_imp * 0.01,
            b.get_y() + b.get_height() / 2,
            f"{v:.4f}",
            va="center", fontsize=7,
        )


def make_plots(df: pd.DataFrame, out_dir: Path, top_n: int = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["importance"] = df["importance"].fillna(0.0)
    df = df[df["importance"] > 0].reset_index(drop=True)

    if top_n:
        df = df.nlargest(top_n, "importance")

    raw = df[df["feature"].isin(RAW_FEATURES)]
    eng = df[df["feature"].isin(ENGINEERED_FEATURES)]
    other = df[~df["feature"].isin(RAW_FEATURES + ENGINEERED_FEATURES)]

    # -------- Figure 1: combined --------
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(df))))
    _barh(ax, df, "All Feature Importances (RandomForest)", "#2e7d32")
    fig.tight_layout()
    p1 = out_dir / "feature_importance_all.png"
    fig.savefig(p1, dpi=200); fig.savefig(p1.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[saved] {p1}")

    # -------- Figure 2: split raw vs engineered --------
    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(5, 0.35 * max(len(raw), len(eng)))),
        gridspec_kw={"width_ratios": [len(raw) or 1, len(eng) or 1]},
    )
    if len(raw):
        _barh(axes[0], raw, "Raw (physical / climatic) features", "#1565c0")
    else:
        axes[0].set_visible(False)
    if len(eng):
        _barh(axes[1], eng, "Engineered features", "#ef6c00")
    else:
        axes[1].set_visible(False)
    fig.suptitle("Feature Importance — Raw vs Engineered", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = out_dir / "feature_importance_split.png"
    fig.savefig(p2, dpi=200); fig.savefig(p2.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[saved] {p2}")

    # -------- Save clean CSV --------
    df.sort_values("importance", ascending=False).to_csv(
        out_dir / "feature_importance_sorted.csv", index=False
    )
    print(f"[saved] {out_dir / 'feature_importance_sorted.csv'}")

    n_show = top_n if top_n else len(df)
    print(df.sort_values("importance", ascending=False).head(n_show).to_string(index=False))


# ----------------------------------------------------------------------
# 4. Entry point
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flood_model_rf_weighted.pkl",
                    help="Pickled model with .feature_importances_")
    ap.add_argument("--csv",   default="outputs/phase2/feature_importance/feature_importance.csv",
                    help="Pre-computed feature_importance.csv")
    ap.add_argument("--out",   default="outputs/feature_importance_plots")
    ap.add_argument("--top",   type=int, default=None, help="Only plot top N features")
    args = ap.parse_args()

    csv_p = Path(args.csv)
    model_p = Path(args.model)

    if csv_p.exists():
        print(f"Loading importance CSV: {csv_p}")
        df = load_from_csv(csv_p)
    elif model_p.exists():
        print(f"Loading model: {model_p}")
        df = load_from_model(model_p)
    else:
        raise FileNotFoundError(
            f"Neither {csv_p} nor {model_p} exist. "
            "Run your training pipeline first, or point --csv / --model to a valid file."
        )

    make_plots(df, Path(args.out), top_n=args.top)


if __name__ == "__main__":
    main()