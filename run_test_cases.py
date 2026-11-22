import json, joblib, numpy as np, pandas as pd

df = pd.read_csv("outputs/phase2/full_dataset_engineered.csv")
F = json.load(open("outputs/phase3/suite_features.json"))["features"]
model = joblib.load("outputs/phase3/overall/m5_gwr_mars_svm_pipeline.pkl")

CLASSES = ["Low", "Medium", "High", "Very High"]

# keep only pixels whose label is identical across events
g = df.groupby(["row", "col"]).flood_label.nunique().reset_index()
ok = g[g.flood_label == 1][["row", "col"]]
u = df.merge(ok, on=["row", "col"]).drop_duplicates(["row", "col"])

# ---- class breaks -------------------------------------------------------
# Replace BREAKS with the m5 risk_breaks used by your API if you have them,
# e.g. BREAKS = [0.45, 0.55, 0.65]. Otherwise they are the quartiles of m5
# probabilities on a random sample of pixels.
BREAKS = None
if BREAKS is None:
    ref = u.sample(min(2000, len(u)), random_state=0)
    p_ref = model.predict_proba(ref[["row", "col"]].to_numpy(float),
                                ref[F].to_numpy(float))
    BREAKS = [float(q) for q in np.quantile(p_ref, [0.25, 0.50, 0.75])]
print("Class breaks (Low | Medium | High | Very High):",
      [round(b, 4) for b in BREAKS])

def to_class(p):
    return CLASSES[int(np.digitize(p, BREAKS))]

flood = u[u.flood_label == 1].sample(16, random_state=42)
non = u[u.flood_label == 0].sample(16, random_state=42)

def predict(xy, X):
    """Validate inputs first, then predict."""
    xy, X = np.asarray(xy, float), np.asarray(X, float)
    if X.shape[0] != len(F):
        return None, f"Expected {len(F)} features, got {X.shape[0]}"
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(xy))):
        return None, "Missing or non-finite input values"
    try:
        return float(model.predict_proba(xy.reshape(1, -1),
                                         X.reshape(1, -1))[0]), None
    except Exception as e:
        return None, str(e).split("\n")[0][:60]

rows, n = [], 1
for label, part in ((1, flood), (0, non)):
    allowed = ["High", "Very High"] if label else ["Low", "Medium"]
    for _, r in part.iterrows():
        p, err = predict([r.row, r.col], r[F].to_numpy(float))
        cls = to_class(p) if p is not None else None
        status = "PASS" if cls in allowed else "FAIL"
        actual = f"p={p:.3f}, {cls}" if p is not None else f"error: {err}"
        rows.append((f"TC{n}", f"row={int(r.row)}, col={int(r.col)}, "
                     f"dem={r.dem:.0f}, dist_river={r.distance_to_river:.0f}m, "
                     f"label={label}", " or ".join(allowed), actual, status))
        n += 1

base = flood.iloc[0]
xy, X0 = [base.row, base.col], base[F].to_numpy(float)

def mod(**kw):
    X = X0.copy()
    for k, v in kw.items():
        X[F.index(k)] = v
    return X

def scaled(**kw):
    X = X0.copy()
    for k, v in kw.items():
        X[F.index(k)] = X[F.index(k)] * v
    return X

edge = [
    # expected to be rejected
    ("NaN in hand", xy, mod(hand=np.nan), "error"),
    ("NaN in dem and chirps", xy, mod(dem=np.nan, chirps=np.nan), "error"),
    ("NaN in twi", xy, mod(twi=np.nan), "error"),
    ("All features NaN", xy, np.full(len(F), np.nan), "error"),
    ("Infinite slope value", xy, mod(slope=np.inf), "error"),
    ("Negative infinite dem", xy, mod(dem=-np.inf), "error"),
    ("Too few features (27 of 28)", xy, X0[:-1], "error"),
    ("Too many features (29 of 28)", xy, np.append(X0, 0.0), "error"),
    ("NaN coordinates", [np.nan, np.nan], X0, "error"),
    # expected to return a valid class
    ("Extreme dem = 10000 m", xy, mod(dem=10000), "valid"),
    ("Negative slope = -10", xy, mod(slope=-10), "valid"),
    ("All features = 0", xy, np.zeros(len(F)), "valid"),
    ("Coordinates outside city (99999, 99999)", [99999, 99999], X0, "valid"),
    ("Negative coordinates (-1, -1)", [-1, -1], X0, "valid"),
    ("Rainfall (chirps) x5", xy, scaled(chirps=5), "valid"),
    ("Distance to river = 0 m", xy, mod(distance_to_river=0), "valid"),
    ("Very large flow accumulation (1e7)", xy, mod(flow_accumulation=1e7), "valid"),
    ("Population density = 0", xy, mod(population_density=0), "valid"),
]
for desc, c, X, exp in edge:
    p, err = predict(c, X)
    if exp == "error":
        status = "PASS" if err else "FAIL"
        actual = f"error: {err}" if err else f"p={p:.3f}, {to_class(p)}"
        e = "Graceful error / imputation"
    else:
        ok_ = p is not None and 0 <= p <= 1
        status = "PASS" if ok_ else "FAIL"
        actual = f"p={p:.3f}, {to_class(p)}" if p is not None else f"error: {err}"
        e = "Valid class (Low/Medium/High/Very High), no crash"
    rows.append((f"TC{n}", desc, e, actual, status))
    n += 1

out = pd.DataFrame(rows, columns=["ID", "Input", "Expected", "Actual", "Status"])
print(out.to_string(index=False))
out.to_csv("outputs/phase3/test_cases_results_m5_50.csv", index=False)

print("\nPassed:", (out.Status == "PASS").sum(), "/", len(out))
print("Flood pixels passed:", (out.Status[:16] == "PASS").sum(), "/ 16")
print("Non-flood pixels passed:", (out.Status[16:32] == "PASS").sum(), "/ 16")
print("Robustness passed:", (out.Status[32:] == "PASS").sum(), "/ 18")

# class distribution of the 32 pixel cases
for name, sl in (("Flood", slice(0, 16)), ("Non-flood", slice(16, 32))):
    cl = out.Actual[sl].str.extract(r",\s*(Very High|High|Medium|Low)")[0]
    print(f"{name} pixels by class:", cl.value_counts().reindex(CLASSES, fill_value=0).to_dict())