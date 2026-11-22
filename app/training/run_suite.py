"""
Train the full model suite (m1–m5) per category.

Categories: oct_2022, jan_2023, jul_2024, overall
            + holdout_<event> (train on the other events, test on <event>)
Split     : spatial-block hold-out (whole blocks to train OR test)
Models   : m1_gwpca_svm, m2_gwr_mgwr, m3_gwpca_mars,
           m4_stacking_all, m5_gwr_mars_svm_pipeline
Outputs  : outputs/phase3/<category>/{<model>.pkl, <model>_summary.json}
           outputs/phase3/suite_model_comparison.csv
           outputs/phase3/model_performance_summary.txt
           outputs/phase3/suite_features.json (feature order the API must use)

Retraining for a newly uploaded event (set by the API):
SUITE_NEW_EVENT=<event>   -> trains <event>, overall and every holdout_<event>
  SUITE_CATEGORIES=a,b,c    -> trains exactly these categories
Results for untouched categories are kept in the comparison CSV.
"""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import functools
import json
import logging
import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from app.training.models import train_model_suite, MODEL_CLASSES, EVENT_KEYS

logger = logging.getLogger(__name__)

# ---- validation settings ----------------------------------------------------
SPLIT = "spatial"             # "spatial" (block hold-out) or "random"
TEMPORAL_HOLDOUT = True       # add leave-one-event-out categories (holdout_<event>)
# Categories to train; None = everything. Options: oct_2022, jan_2023, jul_2024,
# overall, holdout_oct_2022, holdout_jan_2023, holdout_jul_2024
RUN_CATEGORIES = None
# rainfall_mm is a per-event constant (158.3 / 182.7 / 170.2), i.e. an event ID:
# it leaks identity in 'overall' and breaks leave-one-event-out. Excluded here.
EXCLUDE_FEATURES = {"rainfall_mm"}

OUT_DIR = Path("outputs/phase3")
COMPARISON_CSV = OUT_DIR / "suite_model_comparison.csv"

PLOT_METRICS = {"accuracy": "Accuracy", "f1": "F1-Score",
                "roc_auc": "AUC", "kappa": "Kappa"}
MODEL_LABELS = {
    "m1_gwpca_svm": "m1 GWPCA-SVM",
    "m2_gwr_mgwr": "m2 GWR/MGWR",
    "m3_gwpca_mars": "m3 GWPCA-MARS",
    "m4_stacking_all": "m4 Stacking",
    "m5_gwr_mars_svm_pipeline": "m5 GWR-MARS-SVM",
    "m6_mgwr": "m6 MGWR (benchmark)",
}
MODEL_COLORS = ["#8c8c8c", "#2a9d8f", "#e9c46a", "#264653", "#e76f51", "#7b5ea7"]



def _env_categories():
    """Categories requested by the API (retraining), or None for a full run."""
    new_event = os.environ.get("SUITE_NEW_EVENT")
    if new_event:
        cats = [new_event, "overall"]
        if TEMPORAL_HOLDOUT:
            cats += [f"holdout_{k}" for k in EVENT_KEYS]
        return list(dict.fromkeys(cats))
    raw = os.environ.get("SUITE_CATEGORIES")
    if raw:
        return [c.strip() for c in raw.split(",") if c.strip()]
    return None


def _fmt(seconds: float) -> str:
    """Format seconds as 1h 02m 03s / 4m 05s / 6.7s."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _instrument_models(total_fits: int):
    """Wrap each model's fit() to log start, finish, elapsed, and overall progress."""
    state = {"done": 0, "t_start": time.perf_counter()}

    for cls in MODEL_CLASSES:
        if getattr(cls.fit, "_timed", False):
            continue
        orig = cls.fit

        @functools.wraps(orig)
        def timed_fit(self, *args, _orig=orig, **kwargs):
            name = getattr(self, "name", type(self).__name__)
            n = state["done"] + 1
            logger.info(f">>> [{n}/{total_fits}] {name}: fit started")
            t0 = time.perf_counter()
            try:
                return _orig(self, *args, **kwargs)
            finally:
                state["done"] += 1
                dt = time.perf_counter() - t0
                elapsed = time.perf_counter() - state["t_start"]
                avg = elapsed / state["done"]
                eta = avg * (total_fits - state["done"])
                logger.info(
                    f"<<< [{state['done']}/{total_fits}] {name}: fit finished in {_fmt(dt)} "
                    f"| elapsed {_fmt(elapsed)} | rough ETA {_fmt(eta)}"
                )

        timed_fit._timed = True
        cls.fit = timed_fit


def load_phase2(phase2_dir: Path = Path("outputs/phase2")):
    """
    Load the phase-2 feature table with the metadata phase-3 needs.
    Features are derived from the CSV itself (the config.py-driven set),
    because outputs/phase2/feature_names.json is stale from an older config.
    """
    csv_candidates = [
        phase2_dir / "full_dataset_engineered.csv",
        phase2_dir / "full_dataset_engineered_standardized.csv",
    ]
    dataset_path = next((p for p in csv_candidates if p.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError(f"No engineered dataset under {phase2_dir}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} rows from {dataset_path.name}")

    # ---- attach `event` ---------------------------------------------------
    if "event" not in df.columns:
        shp = phase2_dir / "sample_points" / "sample_points.shp"
        if not shp.exists():
            raise FileNotFoundError(f"Missing {shp}")
        samples_df = pd.DataFrame(gpd.read_file(shp).drop(columns="geometry"))

        if len(df) == len(samples_df):
            df["event"] = samples_df["event"].values
            logger.info(f"Attached 'event' by row-index alignment "
                        f"({df['event'].notna().sum()}/{len(df)} matched)")
        elif "sample_id" in df.columns and "sample_id" in samples_df.columns:
            df = df.merge(samples_df[["sample_id", "event"]],
                          on="sample_id", how="left")
            logger.info("Attached 'event' by sample_id")
        else:
            dedup = samples_df.drop_duplicates(subset=["row", "col"], keep="first")
            df = df.merge(dedup[["row", "col", "event"]],
                          on=["row", "col"], how="left")
            logger.warning(
                f"Lossy (row,col) merge: {df['event'].notna().sum()}/{len(df)} matched. "
                f"Pixels spanning multiple events keep only the first."
            )
        if df["event"].notna().sum() == 0:
            raise ValueError("Zero rows matched to an event — check alignment/merge key.")

    # ---- coordinate columns ----------------------------------------------
    if {"x", "y"}.issubset(df.columns):
        coord_cols = ("x", "y")
    elif {"row", "col"}.issubset(df.columns):
        df = df.rename(columns={"row": "x", "col": "y"})
        coord_cols = ("x", "y")
        logger.info("Renamed (row, col) -> (x, y)")
    else:
        raise ValueError(f"No coordinate columns in CSV: {df.columns.tolist()[:10]}")

    # ---- feature list: derive from CSV, not the stale JSON ----------------
    meta = {"flood_label", "event", "sample_id", "row", "col", "x", "y"}
    feature_cols = [c for c in df.columns
                    if c not in meta and c not in EXCLUDE_FEATURES]
    dropped = sorted(EXCLUDE_FEATURES & set(df.columns))
    if dropped:
        logger.info(f"Excluded from features: {dropped}")
    logger.info(f"Using {len(feature_cols)} feature columns from {dataset_path.name}:")
    logger.info(f"  {feature_cols}")

    # Cross-check against config.ALL_FEATURES, if importable
    try:
        from app.preprocessing.config import ALL_FEATURES
        cfg_missing = set(ALL_FEATURES) - set(feature_cols)
        if cfg_missing:
            logger.warning(
                f"{len(cfg_missing)} config features absent from CSV "
                f"(older config, or dropped in preprocessing): {sorted(cfg_missing)}"
            )
    except Exception:
        pass

    return df, feature_cols, coord_cols


# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------
_SUMMARY_COLUMNS = [
    # (results column, header label, is_numeric)
    ("category", "Event",    False),
    ("model",    "Model",    False),
    ("accuracy", "Accuracy", True),
    ("f1",       "F1-Score", True),
    ("roc_auc",  "AUC",      True),
    ("kappa",    "Kappa",    True),
]


def _build_performance_summary(results: pd.DataFrame) -> str:
    """
    Build a boxed ASCII table: one row per (event, model), grouped by event
    (alphabetical) and ranked by AUC (best first) within each event.
    Failed fits are listed underneath the table.
    """
    banner = "=" * 70
    title = f"{banner}\nMODEL PERFORMANCE SUMMARY\n{banner}"

    if results is None or results.empty:
        return f"{title}\n(no results)\n"

    df = results.copy()

    failed = pd.DataFrame()
    if "error" in df.columns:
        failed = df[df["error"].notna()]
        df = df[df["error"].isna()]

    needed = [c for c, _, _ in _SUMMARY_COLUMNS]
    if df.empty or not set(needed).issubset(df.columns):
        body = "(no successful model fits to summarise)"
    else:
        df = df.sort_values(["category", "roc_auc"], ascending=[True, False])

        headers = [h for _, h, _ in _SUMMARY_COLUMNS]
        rows = []
        for _, r in df.iterrows():
            rows.append([
                f"{r[c]:.4f}" if numeric else str(r[c])
                for c, _, numeric in _SUMMARY_COLUMNS
            ])

        widths = [
            max(len(h), *(len(row[i]) for row in rows)) + 2
            for i, h in enumerate(headers)
        ]
        sep = "+" + "+".join("-" * w for w in widths) + "+"

        def line(cells):
            return "|" + "|".join(c.center(w) for c, w in zip(cells, widths)) + "|"

        body = "\n".join([sep, line(headers), sep, *(line(r) for r in rows), sep])

    out = f"{title}\n{body}\n"

    if not failed.empty:
        out += "\nFAILED FITS:\n"
        for _, r in failed.iterrows():
            err = str(r["error"]).replace("\n", " ")
            out += f"  - {r.get('category', '?')} / {r.get('model', '?')}: {err[:120]}\n"

    return out


def _print_performance_summary(results: pd.DataFrame,
                               out_path: Path = OUT_DIR / "model_performance_summary.txt"):
    """Print the summary table and save a copy next to the suite CSV."""
    text = _build_performance_summary(results)
    print("\n" + text)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        logger.info(f"Performance summary saved -> {out_path}")
    except Exception as e:
        logger.warning(f"Could not save performance summary: {e}")

def _plot_metric(results: pd.DataFrame, metric: str, out_path: Path):
    """Grouped horizontal bar chart: one group per category, one bar per model."""
    df = results.copy()
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty or metric not in df.columns:
        logger.warning(f"No data for metric '{metric}'; plot skipped")
        return None

    pivot = df.pivot_table(index="category", columns="model", values=metric)
    models = [m for m in MODEL_LABELS if m in pivot.columns]
    order = [*EVENT_KEYS, "overall"]
    cats = [c for c in order if c in pivot.index]
    cats += sorted(c for c in pivot.index
                   if c not in order and not c.startswith("holdout"))
    cats += sorted(c for c in pivot.index if c.startswith("holdout"))
    pivot = pivot.loc[cats, models]

    fig, ax = plt.subplots(figsize=(10, 1.4 * len(cats) + 1.5))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        ys = [k + (i - (len(models) - 1) / 2) * h for k in range(len(cats))]
        vals = pivot[m].fillna(0)
        bars = ax.barh(ys, vals, height=h * 0.92,
                       color=MODEL_COLORS[i % len(MODEL_COLORS)],
                       label=MODEL_LABELS[m])
        for b, v in zip(bars, vals):
            ax.text(v + (0.004 if v >= 0 else -0.004),
                    b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7)

    ax.invert_yaxis()
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel(PLOT_METRICS[metric])
    ax.set_title(f"Model performance by category ({PLOT_METRICS[metric]})",
                 loc="left", fontweight="bold")
    if metric == "roc_auc":
        ax.axvline(0.5, color="#999", lw=0.8, ls="--")
    ax.margins(x=0.08)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=min(len(models), 3), frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _save_plots(results: pd.DataFrame, out_dir: Path = OUT_DIR):
    """One PNG per metric next to the suite CSV. Never fails the run."""
    for metric in PLOT_METRICS:
        try:
            p = _plot_metric(results, metric,
                             out_dir / f"model_comparison_{metric}.png")
            if p:
                logger.info(f"Plot saved -> {p}")
        except Exception as e:
            logger.warning(f"Plot for '{metric}' failed: {e}")


def main():
    t_total = time.perf_counter()
    logger.info("=" * 60)
    logger.info("STARTING PHASE 3: MODEL SUITE TRAINING")
    logger.info("=" * 60)

    df, feature_cols, coord_cols = load_phase2()

    # Feature order the API needs to rebuild inputs for the saved models
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "suite_features.json").write_text(
        json.dumps({"features": feature_cols, "coords": list(coord_cols)}, indent=2)
    )

    run_categories = list(RUN_CATEGORIES) if RUN_CATEGORIES else _env_categories()
    if run_categories:
        categories = run_categories
    else:
        categories = list(EVENT_KEYS) + ["overall"]
        if TEMPORAL_HOLDOUT:
            categories += [f"holdout_{k}" for k in EVENT_KEYS]
    total_fits = len(categories) * len(MODEL_CLASSES)
    logger.info(f"Categories ({len(categories)}): {categories}")
    logger.info(f"Models per category ({len(MODEL_CLASSES)}): "
                f"{[c.name for c in MODEL_CLASSES]}")
    logger.info(f"Total fits: {total_fits}")

    _instrument_models(total_fits)

    # train_model_suite overwrites the CSV, so keep the previous rows to merge
    previous = pd.read_csv(COMPARISON_CSV) if COMPARISON_CSV.exists() else None

    results = train_model_suite(
        df,
        feature_cols=feature_cols,
        coord_cols=coord_cols,
        label_col="flood_label",
        event_col="event",
        out_dir=str(OUT_DIR),
        max_spatial_samples=2000,
        split=SPLIT,
        temporal_holdout=TEMPORAL_HOLDOUT,
        run_categories=run_categories,
    )

    if run_categories and previous is not None and not results.empty:
        kept = previous[~previous["category"].isin(results["category"].unique())]
        results = pd.concat([kept, results], ignore_index=True)
        results.to_csv(COMPARISON_CSV, index=False)

    _print_performance_summary(results)
    _save_plots(results)
    print(f"\n{len(results)} model/category rows -> {COMPARISON_CSV}")

    n_failed = int(results["error"].notna().sum()) if "error" in results.columns else 0
    logger.info("=" * 60)
    logger.info(f"PHASE 3 COMPLETED in {_fmt(time.perf_counter() - t_total)} "
                f"({len(results) - n_failed} ok, {n_failed} failed)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()