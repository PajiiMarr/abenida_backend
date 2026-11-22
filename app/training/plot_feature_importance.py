"""Ranked horizontal bar chart of phase-3 feature importance.

Importance is read from a saved m3/m5 model (default m3 GWPCA-MARS) and
projected back onto the original input features, or from a pre-computed CSV.

Usage (from backend/):
    python -m app.training.plot_feature_importance
    python -m app.training.plot_feature_importance --category jul_2024
    python -m app.training.plot_feature_importance --model m5_gwr_mars_svm_pipeline
    python -m app.training.plot_feature_importance --categories overall,holdout_jan_2023
    python -m app.training.plot_feature_importance --input outputs/phase3/feature_importance.csv
    python -m app.training.plot_feature_importance --top 15 --export
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE3 = Path(__file__).resolve().parents[2] / "outputs" / "phase3"

MODEL_LABELS = {
    "m1_gwpca_svm": "m1 GWPCA-SVM",
    "m2_gwr_mgwr": "m2 GWR/MGWR",
    "m3_gwpca_mars": "m3 GWPCA-MARS",
    "m4_stacking_all": "m4 Stacking",
    "m5_gwr_mars_svm_pipeline": "m5 GWR-MARS-SVM",
    "m6_mgwr": "m6 MGWR (benchmark)",
}
BAR_COLOR = "#2a9d8f"
REST_COLOR = "#c9d6d3"


# ---------------------------------------------------------------------------
# Importance extraction
# ---------------------------------------------------------------------------
def _normalise(values, names) -> pd.Series:
    """Clip to non-negative, normalise to sum 1, sort descending."""
    v = np.asarray(values, float).ravel()
    names = list(names)
    if len(names) != v.size:
        names = [f"Feature_{i}" for i in range(v.size)]
    s = pd.Series(v, index=names)
    s = s.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
    total = s.sum()
    if total > 0:
        s = s / total
    return s.sort_values(ascending=False)


def _coef_scores(est):
    """Importance of an estimator in its own input space (None if unavailable)."""
    imp = getattr(est, "feature_importances_", None)
    if imp is not None and np.ndim(imp) == 1:
        return np.asarray(imp, float)

    # R earth backend: distribute each term's |coefficient| over its variables
    coefs = getattr(est, "coefs_", None)
    dirs = getattr(est, "dirs_", None)
    if coefs is not None and dirs is not None:
        b = np.abs(np.ravel(np.asarray(coefs, float)))[1:]      # drop intercept
        D = np.abs(np.asarray(dirs, float))                     # (n_terms, p)
        n = min(b.size, D.shape[0])
        if n:
            return (b[:n, None] * D[:n]).sum(0)

    # linear models (e.g. calibrated SVM coefficients, if exposed)
    coef = getattr(est, "coef_", None)
    if coef is not None:
        c = np.asarray(coef, float)
        return np.abs(c).mean(0) if c.ndim > 1 else np.abs(c)

    return None



def _squared_loadings(gwpca):
    """Extract squared PCA loadings from new and legacy GWPCA objects."""

    # Newer GWPCA implementations
    for attr in ("ref_", "components_", "loadings_"):
        value = getattr(gwpca, attr, None)

        if value is None:
            continue

        L = np.asarray(value, dtype=float)

        if L.ndim == 3:
            L = np.nanmean(L ** 2, axis=0)
        elif L.ndim == 2:
            L = L ** 2
        else:
            continue

        if L.ndim == 2 and np.isfinite(L).any():
            return np.nan_to_num(L, nan=0.0)

    # Reconstruct global PCA if the original transformed data is stored
    Z = getattr(gwpca, "Z_", None)

    if Z is not None:
        from sklearn.decomposition import PCA

        Z = np.asarray(Z, dtype=float)

        if Z.ndim == 2 and Z.shape[0] > 1:
            comp = PCA().fit(Z).components_

            n_comp = getattr(
                gwpca,
                "n_components_",
                comp.shape[0]
            )

            return comp[:n_comp] ** 2

    return None



def _project(scores, gwpca, names):
    """Project component importance back to original input features."""

    W = _squared_loadings(gwpca)

    if W is None:
        return None

    scores = np.asarray(scores, dtype=float).ravel()

    # Ensure dimensions match
    n_comp = min(len(scores), W.shape[0])
    W = W[:n_comp]
    scores = scores[:n_comp]

    if W.shape[1] != len(names):
        raise ValueError(
            f"GWPCA loading dimension mismatch: "
            f"{W.shape[1]} loadings vs {len(names)} features"
        )

    # Normalize each component's squared loadings
    W = W / np.maximum(
        W.sum(axis=1, keepdims=True),
        1e-12
    )

    # Project component importance to original features
    projected = W.T @ scores

    return _normalise(projected, names)


def importance_from_model(model, names) -> pd.Series | None:
    """Extract feature importance from a saved suite model."""
    gwpca = getattr(model, "gwpca", None)

    # m3 / m5: MARS sub-model (m3 on GWPCA scores, m5 on raw features)
    mars = getattr(model, "mars", None)
    if mars is not None:
        scores = _coef_scores(getattr(mars, "model", mars))
        if scores is not None:
            if len(scores) == len(names):
                return _normalise(scores, names)
            if gwpca is not None:
                return _project(scores, gwpca, names)

    scores = _coef_scores(model)
    if scores is not None:
        if len(scores) == len(names):
            return _normalise(scores, names)
        if gwpca is not None:
            return _project(scores, gwpca, names)

    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_features(path: Path = PHASE3 / "suite_features.json") -> list[str]:
    if path.exists():
        return list(json.loads(path.read_text())["features"])
    return []


def load_from_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    fcol = next((c for c in ("feature", "features", "variable", "name") if c in df.columns), df.columns[0])
    icol = next((c for c in ("importance", "mean_importance", "score", "value") if c in df.columns), df.columns[1])
    df = df.rename(columns={fcol: "feature", icol: "importance"})
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    return df.set_index("feature")["importance"].sort_values(ascending=False)


def importance_for(category: str, model_name: str) -> pd.Series:
    names = load_features()
    model_path = PHASE3 / category / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    model = joblib.load(model_path)
    s = importance_from_model(model, names)
    if s is None or s.empty or s.sum() <= 0:
        raise ValueError(
            f"No usable feature importance in {model_path.name} "
            f"(opaque backend or a degenerate MARS fit). "
            f"Try --model m3_gwpca_mars or pass --input with a pre-computed CSV."
        )
    return s


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(series: pd.Series, title: str, out: Path, top: int | None = None) -> Path:
    d = series[series > 0].sort_values(ascending=False)
    if top:
        d = d.head(top)
    d = d.iloc[::-1]                                    # largest bar ends on top

    pct = d * 100
    n_top = min(5, len(d))                              # emphasise the top 5
    colors = [REST_COLOR] * (len(d) - n_top) + [BAR_COLOR] * n_top

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.42 * len(d) + 1.4)))
    bars = ax.barh(d.index, pct.values, color=colors, edgecolor="white", linewidth=0.6,
                   height=0.72)

    xmax = float(pct.max()) if len(pct) else 1.0
    for b, v in zip(bars, pct.values):
        ax.text(b.get_width() + xmax * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=8, color="#333")

    ax.set_xlim(0, xmax * 1.14)
    ax.set_xlabel("Relative importance (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="overall", help="phase-3 category (default: overall)")
    p.add_argument("--categories", help="comma-separated list; overrides --category")
    p.add_argument("--model", default="m3_gwpca_mars", choices=list(MODEL_LABELS))
    p.add_argument("--input", type=Path, help="pre-computed importance CSV (single chart)")
    p.add_argument("--top", type=int, default=None, help="plot only the top N features")
    p.add_argument("--outdir", type=Path, default=PHASE3)
    p.add_argument("--export", action="store_true", help="also write the ranked table as CSV")
    a = p.parse_args()

    cats = [c.strip() for c in a.categories.split(",")] if a.categories else [a.category]

    if a.input:
        s = load_from_csv(a.input)
        label = MODEL_LABELS.get(a.model, a.model)
        out = plot(s, f"Feature importance — {label}", a.outdir / "feature_importance.png", a.top)
        print(f"saved -> {out}")
        return

    for cat in cats:
        try:
            s = importance_for(cat, a.model)
        except (FileNotFoundError, ValueError) as e:
            print(f"skipped {cat}: {e}")
            continue
        label = MODEL_LABELS.get(a.model, a.model)
        out = plot(s, f"Feature importance — {label} ({cat})",
                   a.outdir / f"feature_importance_{cat}.png", a.top)
        print(f"saved -> {out}")
        if a.export:
            csv_out = a.outdir / f"feature_importance_{cat}.csv"
            s.rename("importance").reset_index().rename(columns={"index": "feature"}) \
                .to_csv(csv_out, index=False)
            print(f"saved -> {csv_out}")
        print(s.head(a.top or 10).mul(100).round(2).to_string(header=False))


if __name__ == "__main__":
    main()
