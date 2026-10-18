#!/usr/bin/env python3
"""Find out why a flood model under-predicts near rivers.

Input: the table of training samples (CSV/parquet) with one row per sampled pixel:
  label column (1=flood, 0=non-flood), the model features, and x/y in metres (UTM).
Optional: the trained model (.pkl) for partial-dependence curves.

  python diagnose_river_underprediction.py outputs/phase2/full_dataset_engineered.csv \
      --label flood_label --x col --y row --cell 160 --dist distance_to_river --hand hand
  (row/col are 12.5 m pixel indices, so --cell 160 = 2 km spatial blocks)
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BINS = [0, 50, 100, 250, 500, 1000, 2500, np.inf]


def oof_probs(df, feats, label, groups):
    p = np.zeros(len(df))
    cv = GroupKFold(5) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in cv.split(df[feats], df[label], groups):
        m = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced",
                                   n_jobs=-1, random_state=0)
        m.fit(df.iloc[tr][feats], df.iloc[tr][label])
        p[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    return p


def by_distance(df, label, dist, p, thr=0.5, cols=None):
    d = df.assign(bin=pd.cut(df[dist], BINS, right=False), p=p)
    npos, nneg = (d[label] == 1).sum(), (d[label] == 0).sum()
    rows = []
    for b, g in d.groupby("bin", observed=True):
        pos, neg = g[g[label] == 1], g[g[label] == 0]
        rows.append(dict(bin=str(b), n=len(g),
                         pos_share=len(pos) / npos, neg_share=len(neg) / nneg,
                         mean_prob=g.p.mean(),
                         recall=(pos.p >= thr).mean() if len(pos) else np.nan,
                         false_pos_rate=(neg.p >= thr).mean() if len(neg) else np.nan))
    out = pd.DataFrame(rows).round(3)
    return (out[cols] if cols else out).to_string(index=False)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("data"); a.add_argument("--label", default="flood")
    a.add_argument("--dist", default="distance_to_river"); a.add_argument("--hand", default="hand")
    a.add_argument("--x"); a.add_argument("--y"); a.add_argument("--model")
    a.add_argument("--features", help="comma list; default = all numeric columns")
    a.add_argument("--cell", type=float, default=2000, help="spatial CV block size (m)")
    o = a.parse_args()

    df = pd.read_parquet(o.data) if o.data.endswith(".parquet") else pd.read_csv(o.data)
    skip = {o.label, o.x, o.y, "sample_id", "row", "col"}
    feats = o.features.split(",") if o.features else [
        c for c in df.select_dtypes("number").columns if c not in skip]
    df = df.dropna(subset=feats + [o.label])
    groups = None
    if o.x and o.y:
        groups = (df[o.x] // o.cell).astype(int).astype(str) + "_" + (df[o.y] // o.cell).astype(int).astype(str)
    print(f"{len(df)} rows, {int(df[o.label].sum())} flood, {len(feats)} features, "
          f"CV = {'spatial blocks' if groups is not None else 'RANDOM (give --x/--y!)'}")

    print("\n== 1. LABEL AUDIT: where are flood vs non-flood samples relative to rivers? ==")
    print(df.groupby(o.label)[o.dist].describe(percentiles=[.1, .5, .9]).round(1).to_string())
    print("(constant / absurd values or degrees instead of metres => distance raster bug)")
    print(by_distance(df, o.label, o.dist, df[o.label].astype(float),
                      cols=["bin", "n", "pos_share", "neg_share"]))
    print("pos_share >> neg_share in the near-river bins is what a healthy inventory looks like.")
    print("\nSamples that look like water / nodata (use the UNSCALED csv):")
    for lab, g in df.groupby(o.label):
        msg = f"  label={int(lab)}: dist<25 = {(g[o.dist] < 25).mean():.1%}"
        if "dem" in g: msg += f", dem<=1 = {(g['dem'] <= 1).mean():.1%}"
        print(msg)
    print("(a big share of NEGATIVES here => river channel / sea pixels are being labelled non-flood)")

    print("\n== 2. FEATURE QUALITY ==")
    print(pd.DataFrame({"nunique": df[feats].nunique(), "std": df[feats].std().round(4),
                        "pct_zero": (df[feats] == 0).mean().round(3),
                        "pct_eq_median": df[feats].eq(df[feats].median()).mean().round(3)}).sort_values("nunique").head(12).to_string())

    print("\n== 3. ABLATION (out-of-fold; near = recall of flood pixels < 250 m from river) ==")
    rain = [c for c in feats if any(k in c.lower() for k in ("rain", "chirps", "gpm"))]
    rivd = [c for c in feats if any(k in c.lower() for k in ("river", "floodplain", "dist"))]
    core = [c for c in feats if c in {o.dist, o.hand, "dem", "slope", "twi", "flow_accumulation", "relative_elevation"}]
    sets = {"all": feats, "no_rainfall": [c for c in feats if c not in rain],
            "no_river_derived": [c for c in feats if c not in rivd], "core_terrain+river": core}
    for name, fs in sets.items():
        if not fs: continue
        p = oof_probs(df, fs, o.label, groups)
        near = (df[o.dist] < 250) & (df[o.label] == 1)
        print(f"{name:22s} AUC={roc_auc_score(df[o.label], p):.3f}  near-river recall={(p[near] >= .5).mean():.3f}  "
              f"mean P near river={p[df[o.dist] < 250].mean():.3f}")
        if name == "all":
            p_all = p
    print("If AUC drops a lot with spatial vs random CV, the model is memorising location.")

    print("\n== 4. PER-DISTANCE PERFORMANCE of the 'all' model (out-of-fold) ==")
    print(by_distance(df, o.label, o.dist, p_all))
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(df[o.label], p_all)
    print(f"Youden-optimal threshold = {thr[np.argmax(tpr - fpr)]:.3f}  (0.5 is rarely right after class weighting)")

    if o.model:
        import joblib
        from sklearn.inspection import partial_dependence
        m = joblib.load(o.model)
        names = list(getattr(m, "feature_names_in_", feats))
        X = df[names].sample(min(len(df), 3000), random_state=0)
        print("\n== 5. PARTIAL DEPENDENCE of the saved model (should fall with distance & HAND) ==")
        for f in (o.dist, o.hand):
            if f in names:
                pd_ = partial_dependence(m, X, [f], grid_resolution=10)
                print(f, "\n  x:", np.round(pd_["grid_values"][0], 1), "\n  P:", np.round(pd_["average"][0], 3))


if __name__ == "__main__":
    main()