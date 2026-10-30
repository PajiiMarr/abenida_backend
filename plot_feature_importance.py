#!/usr/bin/env python3
"""plot_feature_importance.py — bar chart of feature importances, raw vs engineered."""

import argparse, importlib.util, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_cfg(p):
    s = importlib.util.spec_from_file_location("cfg", p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def dedup(x):
    seen, out = set(), []
    for i in x:
        if i not in seen: seen.add(i); out.append(i)
    return out

def extract(est, names):
    while hasattr(est, "steps") and est.steps: est = est.steps[-1][1]
    subs = list(getattr(est, "named_estimators_", {}).values()) or \
           [e[1] if isinstance(e, tuple) else e for e in getattr(est, "estimators_", [])]
    if subs:
        vs = [extract(s, names) for s in subs]
        vs = [v.values for v in vs if v is not None]
        if vs:
            a = np.mean(np.vstack(vs), axis=0); s = pd.Series(a, index=names)
            return s / s.sum() if s.sum() > 0 else s
    if hasattr(est, "feature_importances_"):
        fi = np.asarray(est.feature_importances_, float)
        if len(fi) == len(names): return pd.Series(fi, index=names)
    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_, float)
        c = np.abs(c).mean(0) if c.ndim > 1 else np.abs(c)
        if len(c) == len(names):
            s = pd.Series(c, index=names); return s / s.sum() if s.sum() > 0 else s
    return None

def _pick_cols(df):
    """Return (feature_col, importance_col) trying hard; None if hopeless."""
    cols = list(df.columns)
    lc = [(c, str(c).strip().lower()) for c in cols]

    def first(match):
        for orig, low in lc:
            if match(low): return orig
        return None

    fcol = (
        first(lambda s: s == "feature") or
        first(lambda s: s in ("features", "name", "variable", "feat")) or
        first(lambda s: "feature" in s or "variable" in s) or
        first(lambda s: "name" in s)
    )
    icol = (
        first(lambda s: s == "importance") or
        first(lambda s: s in ("importances", "value", "score", "weight",
                              "mean_abs_shap", "shap", "coef")) or
        first(lambda s: "import" in s) or
        first(lambda s: "shap" in s) or
        first(lambda s: "score" in s) or
        first(lambda s: "weight" in s)
    )

    # Fall back: exactly 2 columns → assume first is name, second is value
    if (fcol is None or icol is None) and len(cols) >= 2:
        fcol = fcol or cols[0]
        icol = icol or cols[1]

    return fcol, icol

def load_importance(out_dir, names, verbose=True):
    csvs = sorted(out_dir.rglob("*importance*.csv"))
    # also grab any CSV whose *parent directory* is named something-importance
    for sub in out_dir.rglob("*importance*"):
        if sub.is_dir():
            csvs += sorted(sub.rglob("*.csv"))
    csvs = sorted(set(csvs))

    if verbose and csvs:
        print(f"  scanning {len(csvs)} CSV candidate(s):")
        for c in csvs[:8]:
            print(f"    · {c}")

    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            if verbose: print(f"    (skip {p.name}: {e})")
            continue

        fcol, icol = _pick_cols(df)
        if verbose:
            print(f"    {p.name}: columns={list(df.columns)}  "
                  f"→ feature={fcol!r} importance={icol!r}")

        if fcol and icol and df[[fcol, icol]].dropna().shape[0] > 0:
            return dict(zip(df[fcol], df[icol])), str(p)

    # Last resort: try loading a pickle/joblib model
    try:
        import pickle, joblib
    except ImportError:
        return None, None
    for p in list(out_dir.rglob("*.pkl")) + list(out_dir.rglob("*.joblib")):
        try:
            m = joblib.load(p) if p.suffix == ".joblib" else pickle.load(open(p, "rb"))
        except Exception:
            continue
        s = extract(m, names)
        if s is not None:
            return s.to_dict(), str(p)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("app/preprocessing/config.py"))
    ap.add_argument("--save", type=Path, default=Path("feature_importance.png"))
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    raw  = list(cfg.RAW_FEATURES_TO_EXTRACT)
    eng  = dedup(cfg.ENGINEERED_FEATURES)
    allf = list(getattr(cfg, "ALL_FEATURES", raw + eng))

    out = Path(cfg.OUTPUT_DIR)
    if not out.is_absolute(): out = Path.cwd() / out
    print(f"output dir: {out}")
    imp, src = load_importance(out, allf)
    if imp is None: sys.exit(f"No usable importances found under {out}")
    print(f"source    : {src}")

    def top_n(names):
        rows = sorted([(n, imp.get(n, 0.0)) for n in names],
                      key=lambda r: r[1], reverse=True)
        return rows[:a.top]

    raw_rows, eng_rows = top_n(raw), top_n(eng)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, rows, title, color in [
        (axes[0], raw_rows, "Raw features", "#2563eb"),
        (axes[1], eng_rows, "Engineered features", "#f59e0b"),
    ]:
        names_ = [r[0] for r in rows][::-1]
        vals   = [r[1] for r in rows][::-1]
        ax.barh(names_, vals, color=color, edgecolor="black", linewidth=0.4)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    fig.suptitle("Feature Importance — Raw vs Engineered",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.save, dpi=150, bbox_inches="tight")
    print(f"saved     : {a.save}")

if __name__ == "__main__":
    main()