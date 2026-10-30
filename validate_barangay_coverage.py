"""
Validate model risk predictions against ZCDRRMO flooded-barangay records
(Paeng 2022, LPA/Shearline 2023, Habagat 2024).

Label mode (preprocessing check):
  python validate_barangay_coverage.py \
    --points outputs/phase2/sample_points/sample_points.shp \
    --barangays zamboanga_city_barangays.geojson --label-col flood_labe --min-points 3

Prediction mode (after training). Points file needs coordinates + a risk score:
  python validate_barangay_coverage.py \
    --points predictions.csv --x lon --y lat --crs EPSG:4326 \
    --score-col flood_prob --barangays zamboanga_city_barangays.geojson
  (.shp/.gpkg/.geojson/.parquet prediction files also work; --x/--y/--crs only needed for CSV)

Optional: --threshold 0.5 flags barangays whose share of high-risk points is >= --share (default 0.25).
"""
import argparse, re, unicodedata
import geopandas as gpd
import pandas as pd

PAENG = """Ayala,Cawit,La Paz,Labuan,Limpapa,Maasin,Pamucutan,Patalon,Sinunuc,Talisayan,Tulungatung,Malagutay,Recodo,
Baliwasan,Cabatangan,Calarian,Campo Islam,San Roque,San Jose Gusu,Sta. Maria,Pasonanca,Canelar,Camino Nuevo,Tumaga,Tetuan,
Rio Hondo,Kasanyangan,Guiwan,Tugbungan,Talon-Talon,Mampang,Divisoria,Boalan,Lunzuran,Zambowood,
Cabaluay,Curuan,Lumbangan,Manicahan,Mercedes,Putik,Sangali,Tigtabon"""
LPA = """Ayala,Cawit,La Paz,Labuan,Limpapa,Maasin,Pamucutan,Patalon,Sinubong,Sinunuc,Talisayan,Tulungatung,Malagutay,Recodo,
Baliwasan,Cabatangan,Calarian,San Roque,San Jose Gusu,Sta. Maria,Pasonanca,Canelar,Camino Nuevo,Zone III,Tumaga,Tetuan,
Sta. Catalina,Kasanyangan,Guiwan,Tugbungan,Talon-Talon,Mampang,Divisoria,Lunzuran,
Bolong,Bunguiao,Cabaluay,Culianan,Curuan,Guisao,Licomo,Mangusu,Manicahan,Mercedes,Putik,Salaan,Sangali,Tictapul,
Victoria,Vitali,Cacao,Boalan,Zambowood"""
HABAGAT = """Ayala,Cawit,Labuan,Maasin,Pamucutan,Patalon,Sinubong,Sinunuc,Talisayan,Malagutay,Recodo,
Baliwasan,Cabatangan,Calarian,San Roque,San Jose Gusu,Sta. Maria,Pasonanca,Canelar,Tumaga,Tetuan,Kasanyangan,Guiwan,
Tugbungan,Talon-Talon,Divisoria,Boalan,Mangusu,Putik,Sangali,Tictapul,Vitali"""

ALIAS = {"sinubong": "sinubung", "sinunuc": "sinunoc", "zone iii": "barangay zone iii",
         "sta maria": "santa maria", "sta catalina": "santa catalina"}
LABEL_CANDIDATES = ("flood_labe", "flood_label", "label", "flood", "is_flood", "target", "y")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", "", s).replace("-", " ").replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    return ALIAS.get(s, s)


def names(block):
    return {norm(n) for n in block.replace("\n", "").split(",") if n.strip()}


def load_points(path, x, y, crs):
    p = path.lower()
    if p.endswith(".csv"):
        df = pd.read_csv(path)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)
    if p.endswith(".parquet"):
        df = pd.read_parquet(path)
        if "geometry" in df.columns:
            return gpd.read_parquet(path)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)
    return gpd.read_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--barangays", required=True)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--x"); ap.add_argument("--y"); ap.add_argument("--crs", default=None)
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5, help="score >= this counts as high-risk point")
    ap.add_argument("--share", type=float, default=0.25, help="share of high-risk points to flag a barangay")
    ap.add_argument("--out", default="barangay_validation.csv")
    a = ap.parse_args()

    brgy = gpd.read_file(a.barangays)[["adm4_name", "geometry"]].rename(columns={"adm4_name": "barangay"})
    brgy["key"] = brgy["barangay"].map(norm)

    pts = load_points(a.points, a.x, a.y, a.crs).to_crs(brgy.crs)

    events = {"paeng": names(PAENG), "lpa": names(LPA), "habagat": names(HABAGAT)}
    missing = set().union(*events.values()) - set(brgy["key"])
    if missing:
        print("WARNING: unmatched PDF names (add to ALIAS):", sorted(missing))
    brgy["n_events"] = brgy["key"].map(lambda k: sum(k in v for v in events.values()))

    j = gpd.sjoin(pts, brgy[["key", "geometry"]], how="left", predicate="within")
    j = j.dropna(subset=["key"])

    if a.score_col:
        j["_hi"] = (j[a.score_col] >= a.threshold).astype(int)
        g = j.groupby("key")
        agg = pd.DataFrame({
            "n_points": g.size(),
            "mean_score": g[a.score_col].mean(),
            "p90_score": g[a.score_col].quantile(0.9),
            "high_risk_share": g["_hi"].mean(),
        }).reset_index()
        metric = "high_risk_share"
    else:
        lab = a.label_col or next((c for c in LABEL_CANDIDATES if c in j.columns), None)
        if lab is None:
            raise SystemExit(f"No label column found. Columns: {list(j.columns)}")
        print("Using label column:", lab)
        agg = pd.DataFrame({
            "flood_points": j[j[lab] == 1].groupby("key").size(),
            "total_points": j.groupby("key").size(),
        }).reset_index()
        metric = "flood_points"

    res = brgy.drop(columns="geometry").merge(agg, on="key", how="left")
    res[[c for c in agg.columns if c != "key"]] = res[[c for c in agg.columns if c != "key"]].fillna(0)
    res = res.sort_values(metric, ascending=False).reset_index(drop=True)

    truth = res["n_events"] >= 1
    core = res["n_events"] == 3
    if a.score_col:
        pred = res["high_risk_share"] >= a.share
        print(f"High-risk barangay = >= {a.share:.0%} of its points with {a.score_col} >= {a.threshold}")
        # threshold-free ranking check
        k = int(truth.sum())
        topk = res["p90_score"].rank(ascending=False, method="first") <= k
        print(f"Top-{k} by p90 score: recall = {(topk & truth).sum()}/{k} = {(topk & truth).sum()/k:.0%}")
        print(f"Top-{k} by p90 score: recall (all 3 events) = {(topk & core).sum()}/{core.sum()}")
    else:
        pred = res[metric] >= a.min_points

    tp = int((pred & truth).sum())
    print(f"\nPDF-flooded barangays: {truth.sum()} | flooded in all 3 events: {core.sum()}")
    print(f"Recall (any event):   {tp}/{truth.sum()} = {tp/truth.sum():.0%}")
    print(f"Recall (all 3 events): {(pred & core).sum()}/{core.sum()} = {(pred & core).sum()/core.sum():.0%}")
    print(f"Precision vs PDF:     {tp}/{pred.sum()} = {tp/max(int(pred.sum()),1):.0%} (PDF not exhaustive)")
    print("\nMissed PDF barangays:")
    cols = ["barangay", "n_events", metric] + (["mean_score", "p90_score"] if a.score_col else [])
    print(res[truth & ~pred][cols].sort_values("n_events", ascending=False).to_string(index=False))
    res["flagged"] = pred
    res.to_csv(a.out, index=False)
    print("\nSaved", a.out)


if __name__ == "__main__":
    main()