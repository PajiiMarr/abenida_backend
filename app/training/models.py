"""
Machine Learning Models for Flood Susceptibility
Phase 3: GWR, MGWR, MARS, SVM, and Stacking Ensemble
(+ 20-model suite: 5 models x oct_2022 / jan_2023 / jul_2024 / overall)
(+ m5: true GWR -> MARS -> SVM pipeline)
(+ GWPCA applied before GWR/MGWR to fix multicollinearity-induced singular matrices)
"""

import json
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Tuple, Optional, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
    cross_val_score, train_test_split,
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, r2_score,
    mean_squared_error, mean_absolute_error, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import joblib
from pathlib import Path

from app.preprocessing.config import GWPCA_PARAMS, AVAILABLE_EVENTS

logger = logging.getLogger(__name__)

# Single source of truth: config.py
VARIANCE_THRESHOLD = GWPCA_PARAMS["variance_threshold"]

# SVM Bayesian tuning settings. Optionally define SVM_TUNING in config.py to
# override these defaults (see keys below).
from app.preprocessing import config as _cfg
SVM_TUNING = {
    "n_trials": 50,          # Bayesian (TPE) iterations
    "cv_folds": 5,           # CV folds per trial
    "max_tune_samples": 5000,  # tune on a stratified subsample, refit on all
    **getattr(_cfg, "SVM_TUNING", {}),
}

# Spatial-block settings (override via SPATIAL_CV in config.py). Coordinates
# are cut into a blocks_per_axis x blocks_per_axis grid; whole blocks go to
# either train or test (and to a single CV fold), never both.
SPATIAL_CV = {
    "blocks_per_axis": 12,
    **getattr(_cfg, "SPATIAL_CV", {}),
}

from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=UserWarning, module="spglm")
warnings.filterwarnings("ignore", category=LinAlgWarning, module="spglm")


# ===========================================================================
# Shared spatial helpers
# ===========================================================================

def _sanitize_coords(coords: np.ndarray) -> np.ndarray:
    """Jitter duplicate / degenerate coordinates so kernels stay invertible."""
    coords_clean = np.asarray(coords, dtype=float).copy()

    unique = np.unique(coords_clean, axis=0)
    if len(unique) < len(coords_clean):
        n_dupes = len(coords_clean) - len(unique)
        logger.warning(f"Found {n_dupes} duplicate coordinates; adding jitter.")
        coords_clean = coords_clean + np.random.normal(0, 1e-6, coords_clean.shape)

    if np.std(coords_clean, axis=0).sum() < 1e-10:
        logger.warning("All coordinates identical; adding larger jitter.")
        coords_clean = coords_clean + np.random.normal(0, 1e-3, coords_clean.shape)

    return coords_clean


def _knn_bandwidth(coords: np.ndarray, k_frac: float = 0.5) -> float:
    """Median k-th NN distance as a fallback bandwidth."""
    from scipy.spatial import distance_matrix
    d = distance_matrix(coords, coords)
    np.fill_diagonal(d, np.inf)
    k = max(2, int(k_frac * len(coords)))
    kth = np.partition(d, k, axis=1)[:, k]
    return float(np.median(kth))


def _ridge_fallback(X, y, penalty=1.0):
    """Ridge regression fallback for GWR/MGWR when the spatial fit fails."""
    ridge = Ridge(alpha=penalty)
    ridge.fit(X, y)
    preds = ridge.predict(X)
    r2 = ridge.score(X, y)
    n, k = X.shape
    denom = n - k - 1
    adj = r2 - (1 - r2) * k / denom if denom > 0 else r2
    return preds, ridge.coef_, r2, adj, ridge


def _gwr_local_fit(coords_train, X_train_scaled, y_train,
                   coords_test, X_test_scaled,
                   bandwidth, kernel='bisquare', fixed=False,
                   ridge_eps=1e-8):
    """
    Local weighted least squares at every test point.

    Returns
    -------
    preds : (n_test,)          y_hat(s0)
    betas : (n_test, k+1)      local coefficients incl. intercept
    """
    from scipy.spatial.distance import cdist

    coords_train = np.asarray(coords_train, dtype=float)
    coords_test  = np.asarray(coords_test,  dtype=float)
    X_train_scaled = np.asarray(X_train_scaled, dtype=float)
    X_test_scaled  = np.asarray(X_test_scaled,  dtype=float)
    y_train = np.asarray(y_train, dtype=float).flatten()

    n_train, n_test = len(coords_train), len(coords_test)
    Xt = np.column_stack([np.ones(n_train), X_train_scaled])
    Xs = np.column_stack([np.ones(n_test),  X_test_scaled])
    D  = cdist(coords_test, coords_train)

    if fixed:
        t = D / max(float(bandwidth), 1e-9)
        if kernel == 'gaussian':      W = np.exp(-0.5 * t ** 2)
        elif kernel == 'exponential': W = np.exp(-t)
        else:                         W = np.where(t < 1, (1 - t ** 2) ** 2, 0.0)
    else:
        k_nb = max(2, int(round(float(bandwidth))))
        k_nb = min(k_nb, n_train - 1)
        kth  = np.partition(D, k_nb, axis=1)[:, k_nb]
        kth  = np.where(kth > 0, kth, 1.0)
        t = D / kth[:, None]
        if kernel == 'bisquare':      W = np.where(t < 1, (1 - t ** 2) ** 2, 0.0)
        elif kernel == 'gaussian':    W = np.exp(-0.5 * t ** 2)
        else:                         W = np.exp(-t)

    k1   = Xt.shape[1]
    eye  = ridge_eps * np.eye(k1)
    preds = np.empty(n_test)
    betas = np.empty((n_test, k1))
    for i in range(n_test):
        wi   = W[i]
        XtW  = Xt.T * wi
        XtWX = XtW @ Xt + eye
        XtWy = XtW @ y_train
        try:
            b = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        betas[i] = b
        preds[i] = Xs[i] @ b
    return preds, betas


def _gwr_local_predict(coords_train, X_train_scaled, y_train,
                       coords_test, X_test_scaled,
                       bandwidth, kernel='bisquare', fixed=False,
                       ridge_eps=1e-8):
    """Backwards-compatible wrapper — predictions only."""
    preds, _ = _gwr_local_fit(
        coords_train, X_train_scaled, y_train,
        coords_test,  X_test_scaled,
        bandwidth, kernel=kernel, fixed=fixed, ridge_eps=ridge_eps)
    return preds


# ===========================================================================
# Picklable snapshots
# ===========================================================================

class GWRFallbackResults:
    def __init__(self, predy, params, R2, R2_adj, aicc, n, k):
        self.predy = predy
        self.params = params
        self.R2 = R2
        self.R2_adj = R2_adj
        self.aicc = aicc
        self.n = n
        self.k = k


class GWRLiteResults:
    def __init__(self, predy, R2, R2_adj, aicc, n, k):
        self.predy = predy
        self.R2 = R2
        self.R2_adj = R2_adj
        self.aicc = aicc
        self.n = n
        self.k = k


# ===========================================================================
# MARS backend: R's earth::earth
# ===========================================================================

_R_EARTH = None


def _get_earth():
    global _R_EARTH
    if _R_EARTH is not None:
        return _R_EARTH
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    ro.r("suppressPackageStartupMessages(library(earth))")
    _R_EARTH = (ro, pandas2ri, localconverter)
    return _R_EARTH


class EarthBackend:
    """MARS via R's earth::earth, with numpy-based pickle-safe predict()."""

    def __init__(self, max_degree: int = 2, penalty: float = 3.0,
                 max_terms: int = 21, tol: float = 1e-3):
        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.tol = tol
        self.coefs_ = None
        self.dirs_ = None
        self.cuts_ = None
        self.nterms_ = None
        self._x_min = None
        self._requires_r = False

    def fit(self, X, y):
        ro, pandas2ri, localconverter = _get_earth()
        import pandas as pd

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])])
        df["y"] = y

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["df_"] = pandas2ri.py2rpy(df)
        ro.globalenv["deg_"] = self.max_degree
        ro.globalenv["pen_"] = self.penalty
        ro.globalenv["nprune_"] = self.max_terms

        r_fit = ro.r(
            "earth::earth(y ~ ., data = df_, degree = deg_, penalty = pen_, "
            "nprune = nprune_, pmethod = 'backward')"
        )

        self.coefs_ = np.asarray(
            ro.r("function(m) as.numeric(m$coefficients)")(r_fit), dtype=float
        )
        dirs = np.asarray(ro.r("function(m) m$dirs")(r_fit), dtype=float)
        cuts = np.asarray(ro.r("function(m) m$cuts")(r_fit), dtype=float)
        selected = np.asarray(
            ro.r("function(m) as.integer(m$selected.terms)")(r_fit), dtype=int
        )

        if selected.size == len(self.coefs_) - 1:
            selected_rows = selected - 1
        elif selected.size == len(self.coefs_):
            selected_rows = selected[1:] - 1
        else:
            raise RuntimeError(
                f"Cannot map earth coefficients: len(coefs)={len(self.coefs_)}, "
                f"len(selected)={selected.size}"
            )

        self.dirs_ = dirs[selected_rows]
        self.cuts_ = cuts[selected_rows]
        self.nterms_ = int(self.coefs_.shape[0])
        self._x_min = np.nanmin(X, axis=0)

        ro.globalenv["fit_"] = r_fit
        r_train_preds = np.asarray(
            ro.r("function() as.numeric(predict(fit_, newdata = df_))")()
        )

        try:
            our_preds = self.predict(X)
            finite = np.isfinite(our_preds) & np.isfinite(r_train_preds)
            max_diff = float(
                np.max(np.abs(our_preds[finite] - r_train_preds[finite]))
            ) if finite.sum() else float("inf")
            self._requires_r = max_diff > self.tol
        except Exception:
            self._requires_r = True

        if not self._requires_r:
            ro.r("rm(list = intersect(ls(), "
                 "c('df_','deg_','pen_','nprune_','fit_')))")
        return self

    def predict(self, X):
        if self.coefs_ is None:
            raise ValueError("EarthBackend is not fitted")
        if self._requires_r:
            return self._predict_via_r(X)

        X = np.asarray(X, dtype=float)
        n, p = X.shape
        ncoef = self.coefs_.shape[0]
        B = np.ones((n, ncoef))
        for i in range(1, ncoef):
            term = np.ones(n)
            for j in range(p):
                d = self.dirs_[i - 1, j]
                if not np.isfinite(d) or d == 0:
                    continue
                c = self.cuts_[i - 1, j]
                if not np.isfinite(c):
                    c = self._x_min[j]
                if d > 0:
                    term = term * np.maximum(X[:, j] - c, 0.0)
                else:
                    term = term * np.maximum(c - X[:, j], 0.0)
            B[:, i] = term
        return B @ self.coefs_

    def transform(self, X):
        """
        Return MARS hinge basis matrix B[:, 1:] (no intercept column).
        Returns None when the backend requires live R (no numeric decomposition).
        """
        if self.coefs_ is None:
            raise ValueError("EarthBackend is not fitted")
        if self._requires_r:
            return None
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        ncoef = self.coefs_.shape[0]
        B = np.zeros((n, ncoef - 1))
        for i in range(1, ncoef):
            term = np.ones(n)
            for j in range(p):
                d = self.dirs_[i - 1, j]
                if not np.isfinite(d) or d == 0:
                    continue
                c = self.cuts_[i - 1, j]
                if not np.isfinite(c):
                    c = self._x_min[j]
                if d > 0:
                    term = term * np.maximum(X[:, j] - c, 0.0)
                else:
                    term = term * np.maximum(c - X[:, j], 0.0)
            B[:, i - 1] = term
        return B

    def _predict_via_r(self, X):
        ro, pandas2ri, localconverter = _get_earth()
        import pandas as pd
        X = np.asarray(X, dtype=float)
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])])
        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["newdf_"] = pandas2ri.py2rpy(df)
        return np.asarray(
            ro.r("function() as.numeric(predict(fit_, newdata = newdf_))")()
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_requires_r"):
            raise TypeError("EarthBackend requires live R for predict().")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ===========================================================================
# GWR
# ===========================================================================

class GWRModel:
    """
    Geographically Weighted Regression.

    Out-of-sample prediction is computed manually via local weighted least
    squares (see _gwr_local_fit) rather than through mgwr's own
    GWR.predict(), which raises shape errors when predicting outside the
    training set. Training arrays (coords, scaled X, y) are retained on
    the instance so predictions can be made after unpickling.
    """

    def __init__(self, bandwidth: Optional[float] = None,
                 kernel: str = 'bisquare', fixed: bool = False):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.fixed = fixed
        self.model = None
        self._ridge_fallback = None
        self.results = None
        self.is_fitted = False
        self._x_scaler: Optional[StandardScaler] = None
        self._coords_train: Optional[np.ndarray] = None
        self._X_train_scaled: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, coords: np.ndarray, X: np.ndarray, y: np.ndarray):
        try:
            from mgwr.gwr import GWR
            from mgwr.sel_bw import Sel_BW

            logger.info(f"Fitting GWR with {X.shape[0]} samples, {X.shape[1]} features")

            coords_clean = _sanitize_coords(coords)
            self._coords_train = coords_clean

            self._x_scaler = StandardScaler()
            X_scaled = self._x_scaler.fit_transform(X)
            self._X_train_scaled = X_scaled
            self._y_train = np.asarray(y, dtype=float).flatten()

            y_col = self._y_train.reshape(-1, 1)

            if self.bandwidth is None:
                strategies = [
                    ("adaptive bisquare AICc",
                     dict(fixed=False, kernel='bisquare'), 'AICc'),
                    ("adaptive bisquare CV",
                     dict(fixed=False, kernel='bisquare'), 'CV'),
                    ("fixed gaussian AICc",
                     dict(fixed=True, kernel='gaussian'), 'AICc'),
                ]
                for name, kwargs, criterion in strategies:
                    try:
                        selector = Sel_BW(coords_clean, y_col, X_scaled, **kwargs)
                        self.bandwidth = float(selector.search(criterion=criterion))
                        logger.info(
                            f"Bandwidth selected via {name}: {self.bandwidth:.4f}"
                        )
                        self.fixed = kwargs.get('fixed', self.fixed)
                        self.kernel = kwargs.get('kernel', self.kernel)
                        break
                    except Exception as e:
                        logger.warning(f"Bandwidth strategy '{name}' failed: {e}")
                else:
                    self.bandwidth = _knn_bandwidth(coords_clean, k_frac=0.5)
                    logger.warning(
                        f"All bandwidth strategies failed; using k-NN heuristic: "
                        f"{self.bandwidth:.4f}"
                    )

            if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
                self.bandwidth = _knn_bandwidth(coords_clean)
                logger.warning(f"Invalid bandwidth; using k-NN: {self.bandwidth:.4f}")

            self.model = GWR(
                coords_clean, y_col, X_scaled,
                self.bandwidth, kernel=self.kernel, fixed=self.fixed,
            )
            self.results = self.model.fit()
            self.is_fitted = True

            logger.info(
                f"GWR fitted. R²: {self.results.R2:.4f}, "
                f"bandwidth: {self.bandwidth:.4f}, "
                f"kernel: {self.kernel}, fixed: {self.fixed}"
            )

        except Exception as e:
            logger.error(f"GWR fitting failed: {e}")
            preds, coef, r2, adj, ridge = _ridge_fallback(X, y)
            n, k = X.shape
            self.results = GWRFallbackResults(
                predy=preds, params=coef, R2=r2, R2_adj=adj,
                aicc=None, n=n, k=k,
            )
            self._ridge_fallback = ridge
            self.is_fitted = True
            logger.info("Using Ridge regression fallback for GWR")

    def predict_training(self) -> np.ndarray:
        if not self.is_fitted or self.results is None:
            raise ValueError("Model not fitted")
        return np.asarray(self.results.predy).flatten()

    def predict_at(self, coords: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = np.asarray(X, dtype=float)

        if self._ridge_fallback is not None:
            return self._ridge_fallback.predict(X)

        if (self._X_train_scaled is None or self._y_train is None
                or self._coords_train is None or self._x_scaler is None):
            raise RuntimeError(
                "GWRModel.predict_at requires training arrays, which are "
                "missing. Retrain the model with the current code."
            )

        X_scaled = self._x_scaler.transform(X)
        return _gwr_local_predict(
            coords_train=self._coords_train,
            X_train_scaled=self._X_train_scaled,
            y_train=self._y_train,
            coords_test=np.asarray(coords, dtype=float),
            X_test_scaled=X_scaled,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            fixed=self.fixed,
        )

    def local_coefficients_at(self, coords: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self._ridge_fallback is not None:
            return np.tile(self._ridge_fallback.coef_, (len(coords), 1))

        if (self._X_train_scaled is None or self._y_train is None
                or self._coords_train is None or self._x_scaler is None):
            raise RuntimeError(
                "GWRModel.local_coefficients_at requires training arrays, which "
                "are missing. Retrain the model with the current code."
            )

        X = np.asarray(X, dtype=float)
        _, betas = _gwr_local_fit(
            coords_train=self._coords_train,
            X_train_scaled=self._X_train_scaled,
            y_train=self._y_train,
            coords_test=np.asarray(coords, dtype=float),
            X_test_scaled=self._x_scaler.transform(X),
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            fixed=self.fixed,
        )
        return betas[:, 1:]

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        return self.predict_training()

    def get_summary(self) -> Dict:
        if not self.is_fitted or self.results is None:
            return {"status": "not_fitted"}
        return {
            "bandwidth": self.bandwidth,
            "kernel": self.kernel,
            "fixed": self.fixed,
            "r2": getattr(self.results, "R2", None),
            "r2_adj": getattr(self.results, "R2_adj", None),
            "aicc": getattr(self.results, "aicc", None),
            "n": getattr(self.results, "n", None),
            "k": getattr(self.results, "k", None),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        state["_ridge_fallback"] = None
        res = state.get("results")
        if res is not None and not isinstance(res, (GWRFallbackResults, GWRLiteResults)):
            predy = getattr(res, "predy", None)
            if predy is not None:
                predy = np.asarray(predy).flatten()
            state["results"] = GWRLiteResults(
                predy=predy,
                R2=getattr(res, "R2", None),
                R2_adj=getattr(res, "R2_adj", None),
                aicc=getattr(res, "aicc", None),
                n=getattr(res, "n", None),
                k=getattr(res, "k", None),
            )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ===========================================================================
# MGWR
# ===========================================================================

class MGWRModel:
    """
    Multiscale GWR via backfitting, implemented directly in numpy.

    Each coefficient surface (intercept + each covariate) has its own
    adaptive bisquare bandwidth, chosen from a grid by leave-one-out error.
    A small local ridge keeps the local solves stable. Out-of-sample
    prediction applies the same smoothers at the new locations.

    This replaces mgwr.MGWR, whose local solves diverged on this data.
    """

    def __init__(self, multi: bool = True, fracs=None, ridge: float = 1e-2,
                 max_iter: int = 40, tol: float = 5e-4):
        self.multi = multi
        self.fracs = fracs or [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.99]
        self.ridge = ridge
        self.max_iter = max_iter
        self.tol = tol

        self.results = None
        self.is_fitted = False
        self._x_scaler = None
        self._coords_train = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._partial = None
        self._train_pred = None
        self._Xd_train = None
        self._grid = None
        self._fallback_used = None
        self._gwr_fallback = None
        self._ridge_fallback = None
        self.bw = None
        self._r2 = None
        self._n_iter = None

    @staticmethod
    def _weights(Dm, Dsorted, kn):
        h = Dsorted[:, kn - 1] * 1.0000001
        h = np.where(h > 0, h, 1e-9)
        t = Dm / h[:, None]
        return np.where(t < 1, (1 - t ** 2) ** 2, 0.0)

    def fit(self, coords: np.ndarray, X: np.ndarray, y: np.ndarray):
        from scipy.spatial.distance import cdist
        try:
            coords_clean = _sanitize_coords(coords)
            self._coords_train = coords_clean
            n = len(coords_clean)

            self._x_scaler = StandardScaler()
            Xs_ = self._x_scaler.fit_transform(np.asarray(X, dtype=float))
            y_raw = np.asarray(y, dtype=float).flatten()
            self._y_mean = float(y_raw.mean())
            self._y_std = float(y_raw.std()) or 1.0
            yz = (y_raw - self._y_mean) / self._y_std

            Xd = np.column_stack([np.ones(n), Xs_])
            self._Xd_train = Xd
            k = Xd.shape[1]

            logger.info(f"Fitting MGWR (numpy backfit) with {n} samples, "
                        f"{X.shape[1]} features")

            D = cdist(coords_clean, coords_clean)
            Dsorted = np.sort(D, axis=1)
            grid = sorted({int(max(20, min(n - 1, round(f * n)))) for f in self.fracs})
            self._grid = grid
            Wc = {kn: self._weights(D, Dsorted, kn) for kn in grid}
            Ws = {kn: Wc[kn].sum(1) for kn in grid}

            def pick(j, e):
                x = Xd[:, j]
                best = None
                for kn in grid:
                    den = Wc[kn] @ (x * x) + self.ridge * Ws[kn]
                    b = (Wc[kn] @ (x * e)) / den
                    hii = np.clip(x * x / den, 0, 0.99)
                    score = np.mean(((e - x * b) / (1 - hii)) ** 2)
                    if best is None or score < best[0]:
                        best = (score, kn, b)
                return best[1], best[2]

            ols = LinearRegression().fit(Xs_, yz)
            B = np.tile(np.r_[ols.intercept_, ols.coef_], (n, 1))
            bw = [grid[-1]] * k
            fitted = (Xd * B).sum(1)
            rss_old = np.sum((yz - fitted) ** 2)

            for it in range(self.max_iter):
                for j in range(k):
                    e = yz - fitted + Xd[:, j] * B[:, j]
                    bw[j], bj = pick(j, e)
                    fitted += Xd[:, j] * (bj - B[:, j])
                    B[:, j] = bj
                rss = np.sum((yz - fitted) ** 2)
                rel = abs(rss_old - rss) / rss_old
                self._n_iter = it + 1
                logger.info(f"MGWR backfit iter {it + 1}: RSS={rss:.2f} "
                            f"rel.change={rel:.2e}")
                if rel < self.tol:
                    break
                rss_old = rss

            r2 = 1 - np.sum((yz - fitted) ** 2) / np.sum((yz - yz.mean()) ** 2)
            if (not np.isfinite(r2) or r2 < 0 or np.abs(B).max() > 1e3):
                raise RuntimeError(
                    f"MGWR diverged (R2={r2}, max|beta|={np.abs(B).max():.3g})")

            self.bw = np.asarray(bw, dtype=float)
            self._partial = Xd * B + (yz - fitted)[:, None]
            self._train_pred = fitted * self._y_std + self._y_mean
            self._r2 = float(r2)
            self.results = GWRLiteResults(
                predy=self._train_pred, R2=self._r2, R2_adj=None,
                aicc=None, n=n, k=k,
            )
            self.is_fitted = True
            del Wc, Ws, D, Dsorted

            logger.info(f"MGWR fitted. R²: {self._r2:.4f}")
            return

        except Exception as e:
            logger.exception(f"MGWR fit failed: {type(e).__name__}: {e}")
            logger.info("MGWR falling back to standardized GWR")
            self.is_fitted = False
            self._partial = None
            self._Xd_train = None

        try:
            gwr = GWRModel()
            gwr.fit(coords, X, y)
            self._gwr_fallback = gwr
            self.results = gwr.results
            self._train_pred = gwr.predict_training()
            self._fallback_used = "gwr"
            self.is_fitted = True
            return
        except Exception as e:
            logger.warning(f"GWR fallback also failed: {e}")

        preds, coef, r2, adj, ridge = _ridge_fallback(X, y)
        n_, k_ = X.shape
        self.results = GWRFallbackResults(
            predy=preds, params=coef, R2=r2, R2_adj=adj, aicc=None, n=n_, k=k_)
        self._train_pred = np.asarray(preds).flatten()
        self._ridge_fallback = ridge
        self._fallback_used = "ridge"
        self.is_fitted = True

    def predict_training(self) -> np.ndarray:
        if not self.is_fitted or self._train_pred is None:
            raise ValueError("Model not fitted")
        return np.asarray(self._train_pred).flatten()

    def predict_at(self, coords: np.ndarray, X: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import cdist
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = np.asarray(X, dtype=float)

        if self._partial is not None and self._fallback_used is None:
            coords = np.asarray(coords, dtype=float)
            Xd = np.column_stack([np.ones(len(coords)),
                                  self._x_scaler.transform(X)])
            Dt = cdist(coords, self._coords_train)
            Dts = np.sort(Dt, axis=1)
            pred = np.zeros(len(coords))
            for j in range(Xd.shape[1]):
                W = self._weights(Dt, Dts, int(self.bw[j]))
                xtr = self._Xd_train[:, j]
                den = W @ (xtr ** 2) + self.ridge * W.sum(1)
                pred += Xd[:, j] * (W @ (xtr * self._partial[:, j])) / den
            return pred * self._y_std + self._y_mean

        if self._gwr_fallback is not None:
            return self._gwr_fallback.predict_at(coords, X)
        if self._ridge_fallback is not None:
            return self._ridge_fallback.predict(X)
        raise RuntimeError("MGWRModel.predict_at has no available prediction path.")

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        return self.predict_training()

    def get_summary(self) -> Dict:
        if not self.is_fitted or self.results is None:
            return {"status": "not_fitted"}
        summary = {
            "r2": getattr(self.results, "R2", None),
            "r2_adj": getattr(self.results, "R2_adj", None),
            "aicc": getattr(self.results, "aicc", None),
            "n": getattr(self.results, "n", None),
            "k": getattr(self.results, "k", None),
            "fallback_used": self._fallback_used,
            "prediction_method": (self._fallback_used or "numpy_multiscale_backfit"),
        }
        if self.bw is not None:
            summary["bandwidths"] = [float(b) for b in self.bw]
            summary["bandwidths_min"] = float(np.min(self.bw))
            summary["bandwidths_max"] = float(np.max(self.bw))
            summary["n_bandwidths"] = int(len(self.bw))
            summary["n_iter"] = self._n_iter
        return summary


# ===========================================================================
# MARS
# ===========================================================================

class MARSModeL(BaseEstimator, RegressorMixin):
    """MARS with R earth backend priority, pymars/tree/forest fallbacks."""

    def __init__(self, max_degree: int = 2, penalty: float = 3.0,
                 random_state: int = 42, max_terms: int = 21,
                 max_samples: int = 1000):
        self.max_degree = max_degree
        self.penalty = penalty
        self.random_state = random_state
        self.max_terms = max_terms
        self.max_samples = max_samples
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self._estimator_type = "regressor"
        self._backend = None

    def _subsample(self, X, y):
        n = X.shape[0]
        if not self.max_samples or n <= self.max_samples:
            return X, y
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(n, size=self.max_samples, replace=False)
        logger.info(f"MARS: subsampling {n} -> {self.max_samples} samples")
        return X[idx], y[idx]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        try:
            backend = EarthBackend(
                max_degree=self.max_degree,
                penalty=self.penalty,
                max_terms=self.max_terms,
            )
            logger.info(
                f"Fitting MARS (R earth) with {X.shape[0]} samples, "
                f"{X.shape[1]} features, max_degree={self.max_degree}"
            )
            backend.fit(X, y)
            n_check = min(200, X.shape[0])
            train_preds = np.asarray(backend.predict(X[:n_check])).flatten()
            if np.std(train_preds) < 1e-6:
                raise ValueError("R earth predictions are near-constant.")
            if getattr(backend, "_requires_r", False):
                raise RuntimeError("R earth reconstruction unreliable.")
            self.model = backend
            self.is_fitted = True
            self._backend = "r_earth"
            self.degree_used_ = self.max_degree
            logger.info(f"MARS (R earth) fitted: {backend.nterms_} terms")
            return self
        except ImportError as e:
            logger.info(f"rpy2 unavailable ({e}) — trying pymars")
        except Exception as e:
            logger.warning(f"R earth unusable ({type(e).__name__}: {e})")

        try:
            from pymars import Earth
            X_fit, y_fit = self._subsample(X, y)
            # Manuscript: second-order interaction terms. Keep max_degree (2) at any
            # feature count; only drop to 1 if the degree-2 fit itself fails.
            for deg in dict.fromkeys([self.max_degree, 1]):
                try:
                    self.model = Earth(
                        max_degree=deg, penalty=self.penalty, max_terms=self.max_terms,
                    )
                    self.model.fit(X_fit, y_fit)
                    break
                except Exception as e:
                    if deg == 1:
                        raise
                    logger.warning(f"pymars degree={deg} failed ({type(e).__name__}: {e}); retrying degree=1")
            self.degree_used_ = deg
            if deg < self.max_degree:
                logger.warning(f"MARS (pymars) fitted with degree={deg} < required {self.max_degree}: "
                               f"interaction terms NOT included")
            self.is_fitted = True
            self._backend = "pymars"
            nterms = (getattr(self.model, "nterms_", None)
                      or getattr(self.model, "n_terms", None)
                      or len(getattr(self.model, "coef_", [])))
            logger.info(f"MARS (pymars) fitted: {nterms} terms, degree={deg}")
            return self
        except ImportError:
            logger.info("pymars unavailable — using DecisionTree")
        except Exception as e:
            logger.warning(f"pymars failed ({type(e).__name__}: {e})")

        try:
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(
                max_depth=8, min_samples_split=20,
                random_state=self.random_state,
            )
            self.model.fit(X, y)
            self.is_fitted = True
            self._backend = "tree"
            self.degree_used_ = 0
            logger.warning("DecisionTreeRegressor used as MARS fallback: NOT a MARS model "
                           "(no hinge functions / interactions); install rpy2+earth or pymars")
            return self
        except Exception as e:
            logger.error(f"DecisionTree fallback failed: {e}")

        self.model = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=self.random_state,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self._backend = "forest"
        self.degree_used_ = 0
        logger.warning("RandomForestRegressor used as MARS fallback: NOT a MARS model "
                       "(no hinge functions / interactions); install rpy2+earth or pymars")
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        preds = np.asarray(self.model.predict(X)).flatten()
        return (np.clip(preds, 0, 1) >= 0.5).astype(int)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        preds = np.clip(np.asarray(self.model.predict(X)).flatten(), 0, 1)
        return np.column_stack([1 - preds, preds])

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        if self._backend == "r_earth" and hasattr(self.model, "transform"):
            return self.model.transform(X)
        return None

    def get_summary(self):
        if not self.is_fitted:
            return {"status": "not_fitted"}
        return {
            "model_type": type(self.model).__name__,
            "backend": self._backend,
            "max_degree": self.max_degree,
            "degree_used": getattr(self, "degree_used_", None),
            "is_true_mars": self._backend in ("r_earth", "pymars"),
            "max_terms": self.max_terms,
            "penalty": self.penalty,
        }

    def get_feature_importance(self):
        if not self.is_fitted:
            return pd.DataFrame()
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return pd.DataFrame()
        imp = np.asarray(imp)
        if imp.ndim != 1:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": [f"Feature_{i}" for i in range(len(imp))],
            "importance": imp,
        }).sort_values("importance", ascending=False)

    def get_hinge_functions(self):
        if not self.is_fitted:
            return []
        if (self._backend == "r_earth"
                and getattr(self.model, "coefs_", None) is not None
                and not getattr(self.model, "_requires_r", False)):
            rows = []
            ncoef = self.model.coefs_.shape[0]
            for i in range(ncoef):
                if i == 0:
                    rows.append(f"(Intercept)  coef={self.model.coefs_[i]:+.6f}")
                    continue
                terms = []
                for j in range(self.model.dirs_.shape[1]):
                    d = self.model.dirs_[i - 1, j]
                    if not np.isfinite(d) or d == 0:
                        continue
                    c = self.model.cuts_[i - 1, j]
                    if not np.isfinite(c):
                        c = self.model._x_min[j]
                    terms.append(f"h(V{j} {'-' if d > 0 else '+'} {c:.4f})")
                rows.append(
                    f"term{i}: {' * '.join(terms)}  coef={self.model.coefs_[i]:+.6f}"
                )
            return rows
        if hasattr(self.model, "summary"):
            try:
                return [str(self.model.summary())]
            except Exception:
                return []
        return []


# ===========================================================================
# SVM
# ===========================================================================

class SVMModel(BaseEstimator, ClassifierMixin):
    """
    SVM with RBF kernel and calibrated probabilities.

    The `probability` constructor argument is retained for backward
    compatibility but is never forwarded to SVC. In sklearn 1.9 the
    parameter is deprecated at the SVC level regardless of value, so
    passing it at all triggers the FutureWarning.
    """

    def __init__(self, C: float = 1.0, gamma='scale',
                 kernel: str = 'rbf', probability: bool = True):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.probability = probability
        self.model = None
        self.best_params = None
        self.tuning = None
        self.is_fitted = False
        self.scaler = None

    def _tune_grid(self, X_scaled, y, cv):
        """Fallback when optuna is unavailable: small grid search."""
        base_svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        grid = GridSearchCV(
            base_svm,
            {'C': [0.1, 1, 10], 'gamma': [0.1, 1]},
            cv=min(cv, 3), scoring='roc_auc', n_jobs=-1, verbose=0,
        )
        grid.fit(X_scaled, y)
        self.tuning = {"method": "grid_search", "cv_auc": float(grid.best_score_)}
        return dict(grid.best_params_)

    def _tune_bayes(self, X_scaled, y, groups=None):
        """Bayesian optimisation (Optuna TPE) of C and gamma, maximising CV AUC."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n_trials = int(SVM_TUNING["n_trials"])
        folds = int(SVM_TUNING["cv_folds"])
        cap = SVM_TUNING["max_tune_samples"]

        y = np.asarray(y)
        g = None if groups is None else np.asarray(groups)
        if cap and len(y) > cap:
            idx = _stratified_subsample(y, cap, 42)
            Xt, yt = X_scaled[idx], y[idx]
            gt = None if g is None else g[idx]
        else:
            Xt, yt, gt = X_scaled, y, g

        if gt is not None and len(np.unique(gt)) >= folds:
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
            cv_scheme = "spatial_block"
        else:
            gt = None
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            cv_scheme = "stratified_random"
        logger.info(f"SVM Bayesian tuning: {n_trials} trials x {folds}-fold "
                    f"{cv_scheme} CV on {len(yt)} / {len(y)} samples")

        def objective(trial):
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
            svc = SVC(C=C, gamma=gamma, kernel='rbf',
                      class_weight='balanced', random_state=42)
            return float(cross_val_score(
                svc, Xt, yt, cv=cv, groups=gt,
                scoring='roc_auc', n_jobs=-1).mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials)
        self.tuning = {"method": "bayesian_tpe_optuna", "n_trials": n_trials,
                       "cv_folds": folds, "cv_scheme": cv_scheme,
                       "tune_samples": int(len(yt)),
                       "cv_auc": float(study.best_value)}
        logger.info(f"Best CV AUC {study.best_value:.4f} with {study.best_params}")
        return {"C": float(study.best_params["C"]),
                "gamma": float(study.best_params["gamma"])}

    def fit(self, X, y, tune_hyperparams: bool = True, cv: int = 5,
            groups=None):
        logger.info(f"Fitting SVM with {X.shape[0]} samples, {X.shape[1]} features")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if tune_hyperparams:
            try:
                self.best_params = self._tune_bayes(X_scaled, y, groups)
            except ImportError:
                logger.warning("optuna not installed; falling back to grid search")
                self.best_params = self._tune_grid(X_scaled, y, cv)
            logger.info(f"Best parameters: {self.best_params}")
            inner_svm = SVC(
                C=self.best_params['C'],
                gamma=self.best_params['gamma'],
                kernel='rbf',
                class_weight='balanced',
                random_state=42,
            )
        else:
            inner_svm = SVC(
                C=self.C, gamma=self.gamma, kernel=self.kernel,
                class_weight='balanced', random_state=42,
            )

        cal_cv = 3
        if groups is not None:
            g = np.asarray(groups)
            if len(np.unique(g)) >= 3:
                cal_cv = list(StratifiedGroupKFold(
                    n_splits=3, shuffle=True, random_state=42
                ).split(X_scaled, y, g))

        self.model = CalibratedClassifierCV(
            inner_svm, method='sigmoid', cv=cal_cv, ensemble=False,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info("SVM model fitted successfully")
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(self.scaler.transform(X))

    def get_summary(self):
        if not self.is_fitted:
            return {"status": "not_fitted"}
        return {
            'best_params': self.best_params or {'C': self.C, 'gamma': self.gamma},
            'kernel': self.kernel,
            'calibrated': True,
            'calibration_method': 'sigmoid',
            'tuning': self.tuning,
        }


# ===========================================================================
# Stacking
# ===========================================================================

class StackingEnsemble:
    """Stacked ensemble with logistic-regression meta-learner."""

    def __init__(self, base_models: List = None):
        self.base_models = base_models
        self.stacked_model = None
        self.is_fitted = False
        self.scaler = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Fitting Stacking Ensemble with {X_train.shape[0]} samples")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        if self.base_models is None:
            inner_svm = SVC(
                kernel='rbf', C=10, gamma=0.1,
                class_weight='balanced', random_state=42,
            )
            calibrated_svm = CalibratedClassifierCV(
                inner_svm, method='sigmoid', cv=3, ensemble=False,
            )
            base_models = [
                ('lr', LogisticRegression(
                    max_iter=1000, random_state=42, class_weight='balanced',
                )),
                ('svm', calibrated_svm),
                ('rf', RandomForestClassifier(
                    n_estimators=50, random_state=42, class_weight='balanced',
                )),
            ]
        else:
            if isinstance(self.base_models, dict):
                base_models = list(self.base_models.items())
            elif isinstance(self.base_models, list):
                base_models = self.base_models
            else:
                base_models = [('model', self.base_models)]

        meta = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.stacked_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta,
            cv=3,
            stack_method='predict_proba',
        )
        self.stacked_model.fit(X_scaled, y_train)
        self.is_fitted = True
        logger.info("Stacking ensemble fitted successfully")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.stacked_model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.stacked_model.predict_proba(self.scaler.transform(X))

    def get_summary(self):
        if not self.is_fitted:
            return {"status": "not_fitted"}
        return {
            'n_estimators': (len(self.stacked_model.estimators_)
                             if hasattr(self.stacked_model, 'estimators_') else 0),
            'meta_estimator': (str(self.stacked_model.final_estimator_)
                               if hasattr(self.stacked_model, 'final_estimator_') else None),
        }


# ===========================================================================
# Evaluator
# ===========================================================================

class ModelEvaluator:

    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'kappa': cohen_kappa_score(y_true, y_pred),
        }
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.5
            mse = mean_squared_error(y_true, y_proba)
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
            metrics['mae'] = mean_absolute_error(y_true, y_proba)
            metrics['r2'] = r2_score(y_true, y_proba)
        return metrics

    @staticmethod
    def cross_validate(model, X, y, cv=5):
        scores = {
            'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
            'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc'),
            'f1': cross_val_score(model, X, y, cv=cv, scoring='f1_weighted'),
        }
        return {
            k: {
                'mean': v.mean() if len(v) else 0,
                'std': v.std() if len(v) else 0,
                'scores': v.tolist() if len(v) else [],
            }
            for k, v in scores.items()
        }

# ===========================================================================
# Suite: 5 models x (every event in config.AVAILABLE_EVENTS + overall)
# ===========================================================================

EVENT_KEYS = tuple(AVAILABLE_EVENTS)


def _stratified_subsample(y, m, seed):
    """Indices of a stratified subsample (GWR/MGWR are O(n^2) in memory)."""
    n = len(y)
    if m is None or n <= m:
        return np.arange(n)
    idx, _ = train_test_split(np.arange(n), train_size=m, stratify=y, random_state=seed)
    return idx


def _block_groups(coords, blocks_per_axis=None):
    """Spatial block id per point (regular grid over the coordinate extent)."""
    b = int(blocks_per_axis or SPATIAL_CV["blocks_per_axis"])
    c = np.asarray(coords, dtype=float)
    lo, hi = c.min(axis=0), c.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    ij = np.minimum(((c - lo) / span * b).astype(int), b - 1)
    return ij[:, 0] * b + ij[:, 1]


def _spatial_block_split(coords, y, test_size, seed, n_candidates=50):
    """
    Train/test indices where whole spatial blocks are held out. Several random
    block assignments are tried; the one whose test share and flood rate are
    closest to the overall values (and has both classes on both sides) wins.
    """
    y = np.asarray(y)
    groups = _block_groups(coords)
    overall = y.mean()
    best = None
    for k in range(n_candidates):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=seed + k)
        tr, te = next(gss.split(coords, y, groups))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        score = abs(y[te].mean() - overall) + abs(len(te) / len(y) - test_size)
        if best is None or score < best[0]:
            best = (score, tr, te)
    if best is None:
        raise ValueError("No spatial block split gave both classes in train and "
                         "test; lower SPATIAL_CV['blocks_per_axis'].")
    return best[1], best[2]


# ---------------------------------------------------------------------------
# GWPCA transformer (local weighted PCA on anchor points, bisquare adaptive)
# ---------------------------------------------------------------------------
# Drop-in replacement for the GWPCATransformer class in backend/app/training/models.py
# (delete the old class; the imports below already exist at the top of models.py).
#
# Imports needed (already present in models.py):
#   import numpy as np, logging
#   from scipy.spatial import cKDTree
#   from sklearn.decomposition import PCA
#   from sklearn.preprocessing import StandardScaler
#   VARIANCE_THRESHOLD, logger

class GWPCATransformer:
    """
    Geographically weighted PCA (manuscript-aligned).

    - Kernel: adaptive GAUSSIAN, w_ij = exp(-0.5 * (d_ij / h_i)^2), where h_i is
      the distance from i to its k-th nearest training neighbour.
    - Bandwidth (k neighbours): golden-section search minimising leave-one-out
      CV reconstruction error (the point and any pixel sharing its coordinates
      are excluded from its own local PCA).
    - Components: smallest number whose GLOBAL cumulative variance reaches
      `variance_threshold` (GWPCA_PARAMS['variance_threshold'], default 85%).
    - Scores: each observation is projected with the local loadings computed AT
      ITS OWN LOCATION (exact_scores=True). With exact_scores=False, the loadings
      of the nearest of `n_anchors` anchor points are used instead (much faster
      for whole-raster prediction, at a small cost in fidelity).
    - Signs of local components are aligned to the global PCA.
    """

    def __init__(self, variance_threshold=VARIANCE_THRESHOLD, n_anchors=300,
                 k_frac=None, min_k=None, random_state=42,
                 exact_scores=True, n_cv=400, max_k=3000, k_tol=5,
                 chunk=2000):
        self.variance_threshold = variance_threshold
        self.n_anchors = n_anchors
        self.k_frac = k_frac          # if set, skips the search: k = k_frac * n
        self.min_k = min_k            # lower bound for the search (default p + 5)
        self.random_state = random_state
        self.exact_scores = exact_scores
        self.n_cv = n_cv
        self.max_k = max_k
        self.k_tol = k_tol
        self.chunk = chunk

    # ------------------------------------------------------------------
    # Core: one local weighted PCA from a neighbourhood
    # ------------------------------------------------------------------
    def _local_pca(self, Zn, d):
        """Zn: (k, p) neighbour data, d: (k,) distances (ascending). -> mu, V, ratio."""
        h = max(float(d[-1]) * 1.0000001, 1e-9)
        w = np.exp(-0.5 * (d / h) ** 2)
        sw = w.sum()
        mu = (w[:, None] * Zn).sum(0) / sw
        Zc = Zn - mu
        C = (Zc * w[:, None]).T @ Zc / sw
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1][:self.n_components_]
        V = vecs[:, order].T
        s = np.sign((V * self.ref_).sum(1))
        s[s == 0] = 1
        V = V * s[:, None]
        ratio = vals[order].sum() / max(vals.sum(), 1e-12)
        return mu, V, ratio

    def _neighbours(self, q, k, exclude_same_coords=False):
        """k nearest training points to coordinate q."""
        if not exclude_same_coords:
            d, nb = self.tree_.query(q, k=k)
            return np.atleast_1d(d), np.atleast_1d(nb)
        extra = 30  # slack: pixels re-sampled across events share coordinates
        d, nb = self.tree_.query(q, k=min(k + extra, len(self.Z_)))
        keep = d > 0
        return d[keep][:k], nb[keep][:k]

    # ------------------------------------------------------------------
    # Bandwidth: golden-section search on leave-one-out CV
    # ------------------------------------------------------------------
    def _loo_cv(self, k, idx):
        k = int(round(k))
        err = 0.0
        for i in idx:
            d, nb = self._neighbours(self.coords_[i], k, exclude_same_coords=True)
            if len(d) < self.Z_.shape[1] + 2:
                continue
            mu, V, _ = self._local_pca(self.Z_[nb], d)
            r = self.Z_[i] - mu
            err += float(np.sum((r - V.T @ (V @ r)) ** 2))
        return err / len(idx)

    def _select_k(self):
        n, p = self.Z_.shape
        lo = int(self.min_k or (p + 5))
        hi = int(min(n - 1, self.max_k))
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(n, size=min(self.n_cv, n), replace=False)

        phi = (np.sqrt(5) - 1) / 2
        a, b = float(lo), float(hi)
        c, d_ = b - phi * (b - a), a + phi * (b - a)
        fc, fd = self._loo_cv(c, idx), self._loo_cv(d_, idx)
        while abs(b - a) > self.k_tol:
            if fc < fd:
                b, d_, fd = d_, c, fc
                c = b - phi * (b - a)
                fc = self._loo_cv(c, idx)
            else:
                a, c, fc = c, d_, fd
                d_ = a + phi * (b - a)
                fd = self._loo_cv(d_, idx)
        k = int(round((a + b) / 2))
        logger.info(f"GWPCA golden-section LOO-CV bandwidth: k={k} neighbours "
                    f"(searched {lo}-{hi}, {len(idx)} CV points)")
        return k

    # ------------------------------------------------------------------
    def fit(self, coords, X):
        coords = np.asarray(coords, float)
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(np.asarray(X, float))
        n, p = Z.shape

        gp = PCA().fit(Z)
        cum = np.cumsum(gp.explained_variance_ratio_)
        self.n_components_ = int(min(np.searchsorted(cum, self.variance_threshold) + 1, p))
        self.ref_ = gp.components_[:self.n_components_]

        self.coords_, self.Z_ = coords, Z
        self.tree_ = cKDTree(coords)

        if self.k_frac is not None:
            self.k_ = int(np.clip(self.k_frac * n, p + 5, n - 1))
        else:
            self.k_ = self._select_k()

        # local variance explained, averaged over a sample of locations
        rng = np.random.RandomState(self.random_state)
        samp = rng.choice(n, size=min(300, n), replace=False)
        ratios = []
        for i in samp:
            d, nb = self._neighbours(coords[i], self.k_)
            ratios.append(self._local_pca(Z[nb], d)[2])
        self.local_variance_explained_ = float(np.mean(ratios))

        # anchors (only used when exact_scores=False)
        self.anchor_loadings_, self.anchor_mu_, self.anchor_tree_ = None, None, None
        if not self.exact_scores:
            a_idx = rng.choice(n, size=min(self.n_anchors, n), replace=False)
            L, M = [], []
            for i in a_idx:
                d, nb = self._neighbours(coords[i], self.k_)
                mu, V, _ = self._local_pca(Z[nb], d)
                L.append(V)
                M.append(mu)
            self.anchor_loadings_ = np.asarray(L)
            self.anchor_mu_ = np.asarray(M)
            self.anchor_tree_ = cKDTree(coords[a_idx])

        logger.info(f"GWPCA fit: {self.n_components_} components, k={self.k_}, "
                    f"local var explained={self.local_variance_explained_:.2%}, "
                    f"scores={'exact per-point' if self.exact_scores else 'nearest-anchor'}")
        return self

    # ------------------------------------------------------------------
    def transform(self, coords, X):
        coords = np.asarray(coords, float)
        Z = self.scaler_.transform(np.asarray(X, float))

        # fallback path: older pickles / exact_scores=False
        if not getattr(self, "exact_scores", False):
            _, a = self.anchor_tree_.query(coords)
            return np.einsum("nmp,np->nm", self.anchor_loadings_[a], Z)

        out = np.empty((len(Z), self.n_components_))
        for s in range(0, len(Z), self.chunk):
            for j in range(s, min(s + self.chunk, len(Z))):
                d, nb = self._neighbours(coords[j], self.k_)
                _, V, _ = self._local_pca(self.Z_[nb], d)
                out[j] = V @ Z[j]
        return out

    def fit_transform(self, coords, X):
        return self.fit(coords, X).transform(coords, X)

# ---------------------------------------------------------------------------
# Common interface: fit(coords, X, y) / predict_proba(coords, X) -> 1-D proba
# ---------------------------------------------------------------------------
class GWPCASVMModel:
    """Model 1: GWPCA scores -> calibrated SVM."""
    name = "m1_gwpca_svm"

    def __init__(self, random_state=42, variance_threshold=VARIANCE_THRESHOLD, **_):
        self.gwpca = GWPCATransformer(variance_threshold, random_state=random_state)
        self.svm = SVMModel()

    def fit(self, coords, X, y):
        S = self.gwpca.fit_transform(coords, X)
        self.svm.fit(S, y, tune_hyperparams=True, groups=_block_groups(coords))
        return self

    def predict_proba(self, coords, X):
        return self.svm.predict_proba(self.gwpca.transform(coords, X))[:, 1]

    def get_summary(self):
        return {"n_components": self.gwpca.n_components_,
                "local_variance_explained": self.gwpca.local_variance_explained_,
                "svm": self.svm.get_summary()}


class GWRMGWRModel:
    """
    Model 2: GWPCA -> mean of GWR and MGWR probabilities.

    GWPCA is applied first to eliminate multicollinearity between the 28
    features; without it, local WLS at n=2000 is rank-deficient and mgwr
    raises 'A singular matrix detected'. Matches the manuscript's stated
    Objective 2 ('dimensionality reduction using GWPCA ... reduces
    multicollinearity among factors conditioning the flood events').
    """
    name = "m2_gwr_mgwr"

    def __init__(self, random_state=42, max_spatial_samples=2000,
                 variance_threshold=VARIANCE_THRESHOLD, **_):
        self.random_state = random_state
        self.max_spatial_samples = max_spatial_samples
        self.gwpca = GWPCATransformer(variance_threshold, random_state=random_state)
        self.gwr, self.mgwr = GWRModel(), MGWRModel()

    def fit(self, coords, X, y):
        X_pca = self.gwpca.fit_transform(coords, X)
        logger.info(f"[m2] GWPCA -> {self.gwpca.n_components_} components "
                    f"(local var: {self.gwpca.local_variance_explained_:.2%})")
        idx = _stratified_subsample(y, self.max_spatial_samples, self.random_state)
        self.gwr.fit(coords[idx], X_pca[idx], y[idx])
        self.mgwr.fit(coords[idx], X_pca[idx], y[idx])
        return self

    def predict_proba(self, coords, X):
        X_pca = self.gwpca.transform(coords, X)
        p = (np.clip(self.gwr.predict_at(coords, X_pca), 0, 1)
             + np.clip(self.mgwr.predict_at(coords, X_pca), 0, 1)) / 2
        return p

    def get_summary(self):
        return {"gwpca_components": self.gwpca.n_components_,
                "gwr": self.gwr.get_summary(),
                "mgwr": self.mgwr.get_summary()}


class GWPCAMARSModel:
    """Model 3: GWPCA scores -> MARS."""
    name = "m3_gwpca_mars"

    def __init__(self, random_state=42, variance_threshold=VARIANCE_THRESHOLD, **_):
        self.gwpca = GWPCATransformer(variance_threshold, random_state=random_state)
        self.mars = MARSModeL(random_state=random_state)

    def fit(self, coords, X, y):
        self.mars.fit(self.gwpca.fit_transform(coords, X), y)
        return self

    def predict_proba(self, coords, X):
        return self.mars.predict_proba(self.gwpca.transform(coords, X))[:, 1]

    def get_summary(self):
        return {"n_components": self.gwpca.n_components_,
                "mars": self.mars.get_summary()}


class SpatialStackingModel:
    """
    Model 4: stacking ensemble over SVM, MARS, GWR, MGWR and a GWPCA-score
    logistic model.

    SVM and MARS use raw features (they handle collinearity fine).
    GWR, MGWR, and the GWPCA-LR base learner all use GWPCA scores, so the
    spatial base learners are not rank-deficient.
    """
    name = "m4_stacking_all"

    def __init__(self, random_state=42, meta_frac=0.25,
                 max_spatial_samples=2000, variance_threshold=VARIANCE_THRESHOLD, **_):
        self.random_state = random_state
        self.meta_frac = meta_frac
        self.max_spatial_samples = max_spatial_samples
        self.svm = SVMModel(C=10, gamma=0.1)
        self.mars = MARSModeL(random_state=random_state)
        self.gwr, self.mgwr = GWRModel(), MGWRModel()
        self.gwpca = GWPCATransformer(variance_threshold, random_state=random_state)
        self.gwpca_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.base_names = ["svm", "mars", "gwr", "mgwr", "gwpca_lr"]

    def _base_matrix(self, coords, X):
        X_pca = self.gwpca.transform(coords, X)
        return np.column_stack([
            self.svm.predict_proba(X)[:, 1],
            self.mars.predict_proba(X)[:, 1],
            np.clip(self.gwr.predict_at(coords, X_pca), 0, 1),
            np.clip(self.mgwr.predict_at(coords, X_pca), 0, 1),
            self.gwpca_lr.predict_proba(X_pca)[:, 1],
        ])

    def fit(self, coords, X, y):
        ib, im = _spatial_block_split(coords, y, self.meta_frac, self.random_state)
        cb, Xb, yb = coords[ib], X[ib], y[ib]

        self.svm.fit(Xb, yb, tune_hyperparams=False, groups=_block_groups(cb))
        self.mars.fit(Xb, yb)

        # GWPCA is fit once here; reused by GWR, MGWR, and gwpca_lr.
        Xb_pca = self.gwpca.fit_transform(cb, Xb)
        logger.info(f"[m4] GWPCA -> {self.gwpca.n_components_} components")
        sub = _stratified_subsample(yb, self.max_spatial_samples, self.random_state)
        self.gwr.fit(cb[sub], Xb_pca[sub], yb[sub])
        self.mgwr.fit(cb[sub], Xb_pca[sub], yb[sub])
        self.gwpca_lr.fit(Xb_pca, yb)

        self.meta.fit(self._base_matrix(coords[im], X[im]), y[im])
        return self

    def predict_proba(self, coords, X):
        return self.meta.predict_proba(self._base_matrix(coords, X))[:, 1]

    def get_summary(self):
        return {"meta_weights": dict(zip(self.base_names, self.meta.coef_[0].round(4).tolist())),
                "gwpca_components": self.gwpca.n_components_}


# ===========================================================================
# M5 — True GWR -> MARS -> SVM pipeline (with GWPCA pre-stage)
# ===========================================================================
class GWRMARSSVMPipeline:
    """
    Sequential pipeline.

    Stage 0 — GWPCA (collinearity reduction)
        Fit GWPCA on (X, y). Produces n_components_ scores per sample.

    Stage 1 — GWR (spatial regression)
        Fit GWR on GWPCA scores. Derive:
          β(x)  : local coefficients (n_components features)  -> spatial context
          p_gwr : GWR-fitted probability (1 feature)

    Stage 2 — MARS (non-linear feature extraction)
        Fit MARS on the RAW features (MARS handles collinearity). Derive:
          h_j(x) : hinge basis functions (m features)
          p_mars : MARS-fitted probability (1 feature)
        If the backend cannot expose hinge terms, only p_mars is used.

    Stage 3 — SVM (final calibrated classifier)
        Train on the stacked design matrix
          Z = [ X_raw | h_j(x) | β(x) | p_gwr | p_mars ]
        p_svm(x) is the final flood probability.

        Leakage control: at fit time β, p_gwr and p_mars are OUT-OF-FOLD
        (spatial-block K-fold): each training point's values come from GWR /
        MARS models that never saw its block. Predicting new points uses the
        final GWR / MARS fitted on all training data. Without this the SVM
        learns from in-sample stage outputs that already encode the labels.
    """
    name = "m5_gwr_mars_svm_pipeline"

    def __init__(self, random_state=42, max_spatial_samples=2000,
                 n_risk_classes=6, variance_threshold=VARIANCE_THRESHOLD,
                 oof_folds=5, **_):
        self.random_state = random_state
        self.max_spatial_samples = max_spatial_samples
        self.n_risk_classes = n_risk_classes
        self.oof_folds = oof_folds

        self.gwpca = GWPCATransformer(variance_threshold, random_state=random_state)
        self.gwr  = GWRModel()
        self.mars = MARSModeL(random_state=random_state)
        self.svm  = SVMModel(C=10, gamma=0.1)
        self._svm_input_dim = None
        self._has_hinge = None

    # ---- stages ----------------------------------------------------------
    def _stage1(self, coords, X):
        # GWR operates in GWPCA space to avoid singular local systems.
        X_pca = self.gwpca.transform(coords, X)
        beta = self.gwr.local_coefficients_at(coords, X_pca)
        p    = np.clip(self.gwr.predict_at(coords, X_pca), 0.0, 1.0)
        return beta, p

    def _stage2(self, X):
        p     = np.clip(self.mars.predict_proba(X)[:, 1], 0.0, 1.0)
        hinge = self.mars.transform(X)
        return hinge, p

    @staticmethod
    def _assemble(X, hinge, beta, p_gwr, p_mars):
        parts = [X, beta, p_gwr[:, None], p_mars[:, None]]
        if hinge is not None:
            parts.insert(1, hinge)    # hinge features sit right after raw X
        return np.hstack(parts)

    def _compose(self, coords, X):
        beta, p_gwr   = self._stage1(coords, X)
        hinge, p_mars = self._stage2(X)
        return self._assemble(X, hinge, beta, p_gwr, p_mars)

    def _oof_stage_features(self, coords, X, X_pca, y):
        """Spatial-block out-of-fold beta / p_gwr / p_mars for the training set."""
        n = len(y)
        groups = _block_groups(coords)
        k = int(self.oof_folds)
        beta = np.zeros((n, X_pca.shape[1]))
        p_gwr = np.zeros(n)
        p_mars = np.zeros(n)
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                  random_state=self.random_state)
        for f, (tr, te) in enumerate(cv.split(X, y, groups), 1):
            logger.info(f"[m5] OOF fold {f}/{k}: fit on {len(tr)}, "
                        f"features for {len(te)} held-out-block samples")
            sub = _stratified_subsample(y[tr], self.max_spatial_samples,
                                        self.random_state)
            gwr = GWRModel()
            gwr.fit(coords[tr][sub], X_pca[tr][sub], y[tr][sub].astype(float))
            mars = MARSModeL(random_state=self.random_state)
            mars.fit(X[tr], y[tr].astype(float))
            beta[te] = gwr.local_coefficients_at(coords[te], X_pca[te])
            p_gwr[te] = np.clip(gwr.predict_at(coords[te], X_pca[te]), 0.0, 1.0)
            p_mars[te] = np.clip(mars.predict_proba(X[te])[:, 1], 0.0, 1.0)
        return beta, p_gwr, p_mars

    # ---- fit / predict ---------------------------------------------------
    def fit(self, coords, X, y):
        coords = np.asarray(coords, dtype=float)
        X      = np.asarray(X, dtype=float)
        y      = np.asarray(y, dtype=int)

        # Stage 0: GWPCA on the full training set.
        X_pca = self.gwpca.fit_transform(coords, X)
        logger.info(f"[m5] Stage 0 GWPCA -> {self.gwpca.n_components_} components "
                    f"(local var: {self.gwpca.local_variance_explained_:.2%})")

        sub = _stratified_subsample(y, self.max_spatial_samples, self.random_state)
        logger.info(f"[m5] Stage 1 GWR fit on {len(sub)} / {len(y)} samples "
                    f"in GWPCA space")
        self.gwr.fit(coords[sub], X_pca[sub], y[sub].astype(float))

        logger.info(f"[m5] Stage 2 MARS fit on {len(y)} raw samples")
        self.mars.fit(X, y.astype(float))

        logger.info(f"[m5] Stage 3 assembling SVM design matrix from "
                    f"{self.oof_folds}-fold spatial out-of-fold stage features")
        beta, p_gwr, p_mars = self._oof_stage_features(coords, X, X_pca, y)
        Z = self._assemble(X, self.mars.transform(X), beta, p_gwr, p_mars)
        self._svm_input_dim = Z.shape[1]
        self._has_hinge = (self.mars.transform(X[:1]) is not None)
        logger.info(
            f"[m5] SVM input dim: {Z.shape[1]} "
            f"(raw X={X.shape[1]}, "
            f"hinge={'yes' if self._has_hinge else 'no (MARS backend opaque)'}, "
            f"beta={X_pca.shape[1]}, + p_gwr + p_mars)"
        )
        self.svm.fit(Z, y, tune_hyperparams=True, groups=_block_groups(coords))
        return self

    def predict_proba(self, coords, X):
        Z = self._compose(np.asarray(coords, dtype=float),
                          np.asarray(X, dtype=float))
        if self._svm_input_dim is not None and Z.shape[1] != self._svm_input_dim:
            raise RuntimeError(
                f"SVM input dim mismatch: fit={self._svm_input_dim}, "
                f"predict={Z.shape[1]} — did a stage change after fit()?")
        return self.svm.predict_proba(Z)[:, 1]

    def predict(self, coords, X, threshold=0.5):
        return (self.predict_proba(coords, X) >= threshold).astype(int)

    def classify_risk(self, proba):
        proba = np.clip(np.asarray(proba, dtype=float), 0, 1)
        edges = np.linspace(0, 1, self.n_risk_classes + 1)
        idx   = np.clip(np.digitize(proba, edges[1:-1]),
                        0, self.n_risk_classes - 1)
        if self.n_risk_classes == 6:
            names = ["Very Low", "Low", "Moderate",
                     "High", "Very High", "Extreme"]
        elif self.n_risk_classes == 4:
            names = ["Low", "Medium", "High", "Very High"]
        else:
            names = [f"Class {i+1}" for i in range(self.n_risk_classes)]
        return np.asarray(names)[idx]

    def get_summary(self):
        return {
            "stages": ["gwpca", "gwr", "mars", "svm"],
            "n_risk_classes": self.n_risk_classes,
            "gwpca_components": self.gwpca.n_components_,
            "svm_input_dim":  self._svm_input_dim,
            "has_hinge_features": self._has_hinge,
            "stage_features": "spatial_block_out_of_fold",
            "oof_folds": self.oof_folds,
            "gwr":  self.gwr.get_summary(),
            "mars": self.mars.get_summary(),
            "svm":  self.svm.get_summary(),
        }


class MGWRBenchmarkModel:
    """
    Model 6: GWPCA scores -> MGWR (standalone benchmark).

    Same pre-processing as m2 (GWPCA, then a stratified subsample because
    MGWR is O(n^2) in memory), but MGWR is the only predictor, so its
    metrics can be reported on their own.

    If MGWRModel falls back to GWR or ridge, the result is no longer a true
    MGWR benchmark. By default this is logged as a warning and recorded in
    the summary; with strict_fallback=True the fit raises, and the runner
    records the category as a failed fit.
    """
    name = "m6_mgwr"

    def __init__(self, random_state=42, max_spatial_samples=2000,
                 variance_threshold=VARIANCE_THRESHOLD,
                 strict_fallback=False, **_):
        self.random_state = random_state
        self.max_spatial_samples = max_spatial_samples
        self.strict_fallback = strict_fallback
        self.gwpca = GWPCATransformer(variance_threshold, random_state=random_state)
        self.mgwr = MGWRModel()

    def fit(self, coords, X, y):
        coords = np.asarray(coords, dtype=float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        X_pca = self.gwpca.fit_transform(coords, X)
        logger.info(f"[m6] GWPCA -> {self.gwpca.n_components_} components "
                    f"(local var: {self.gwpca.local_variance_explained_:.2%})")

        idx = _stratified_subsample(y, self.max_spatial_samples, self.random_state)
        self.mgwr.fit(coords[idx], X_pca[idx], y[idx].astype(float))

        fb = self.mgwr._fallback_used
        if fb is not None:
            msg = (f"[m6] MGWR fell back to '{fb}'; results are NOT a true "
                   f"MGWR benchmark")
            if self.strict_fallback:
                raise RuntimeError(msg)
            logger.warning(msg)
        return self

    def predict_proba(self, coords, X):
        X_pca = self.gwpca.transform(np.asarray(coords, dtype=float),
                                     np.asarray(X, dtype=float))
        return np.clip(
            self.mgwr.predict_at(np.asarray(coords, dtype=float), X_pca), 0, 1)

    def get_summary(self):
        return {"gwpca_components": self.gwpca.n_components_,
                "fallback_used": self.mgwr._fallback_used,
                "true_mgwr": self.mgwr._fallback_used is None,
                "mgwr": self.mgwr.get_summary()}

    

MODEL_CLASSES = [
    GWPCASVMModel,
    GWRMGWRModel,
    GWPCAMARSModel,
    SpatialStackingModel,
    GWRMARSSVMPipeline,
    MGWRBenchmarkModel,
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def train_model_suite(df: pd.DataFrame,
                      feature_cols: List[str],
                      coord_cols=("x", "y"),
                      label_col: str = "flood_label",
                      event_col: str = "event",
                      out_dir="outputs/phase3",
                      test_size: float = 0.2,
                      seed: int = 42,
                      max_spatial_samples: int = 2000,
                      run_categories: Optional[List[str]] = None,
                      split: str = "spatial",
                      temporal_holdout: bool = True) -> pd.DataFrame:
    """
    df must contain feature_cols, coord_cols (metric CRS, e.g. EPSG:32651),
    label_col (0/1) and event_col with values from config.AVAILABLE_EVENTS.
    Each category gets its own train/test split and fresh models.

    split: 'spatial' holds out whole spatial blocks (see SPATIAL_CV); 'random'
           is the old stratified pixel split.
    temporal_holdout: also adds 'holdout_<event>' categories that train on the
           other events and test on <event> (leave-one-event-out).
    run_categories: optional subset of category names to train
           (event keys, 'overall', 'holdout_<event>').
    """
    if event_col not in df.columns:
        raise ValueError(f"'{event_col}' column missing; per-event training needs it "
                         f"(values: {EVENT_KEYS}).")
    if split not in ("spatial", "random"):
        raise ValueError("split must be 'spatial' or 'random'")

    out_dir = Path(out_dir)
    categories = {k: df[df[event_col] == k] for k in EVENT_KEYS}
    categories["overall"] = df
    if temporal_holdout:
        known = df[df[event_col].isin(EVENT_KEYS)]
        for k in EVENT_KEYS:
            categories[f"holdout_{k}"] = (known[known[event_col] != k],
                                          known[known[event_col] == k])

    if run_categories is not None:
        unknown = set(run_categories) - set(categories)
        if unknown:
            raise ValueError(f"Unknown categories {sorted(unknown)}; "
                             f"available: {sorted(categories)}")
        categories = {k: v for k, v in categories.items() if k in run_categories}
        logger.info(f"Running only categories: {list(categories)}")

    need = [*feature_cols, *coord_cols, label_col]

    def _arrays(part):
        return (part[feature_cols].to_numpy(float),
                part[list(coord_cols)].to_numpy(float),
                part[label_col].to_numpy(int))

    rows = []
    for cat, d in categories.items():
        if isinstance(d, tuple):
            dtr, dte = (part.dropna(subset=need) for part in d)
            if dtr[label_col].nunique() < 2 or dte[label_col].nunique() < 2:
                logger.error(f"[{cat}] train and test each need both classes; skipping")
                continue
            Xtr, ctr, ytr = _arrays(dtr)
            Xte, cte, yte = _arrays(dte)
            scheme = "temporal_holdout"
        else:
            d = d.dropna(subset=need)
            if d[label_col].nunique() < 2:
                logger.error(f"[{cat}] needs both classes; skipping")
                continue
            X, coords, y = _arrays(d)
            if split == "spatial":
                itr, ite = _spatial_block_split(coords, y, test_size, seed)
                scheme = "spatial_block"
            else:
                itr, ite = train_test_split(np.arange(len(y)), test_size=test_size,
                                            stratify=y, random_state=seed)
                scheme = "random"
            Xtr, ctr, ytr = X[itr], coords[itr], y[itr]
            Xte, cte, yte = X[ite], coords[ite], y[ite]
        logger.info(f"[{cat}] ({scheme}) train={len(ytr)} test={len(yte)} "
                    f"flood_rate train={ytr.mean():.2%} test={yte.mean():.2%}")

        cat_dir = out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        for cls in MODEL_CLASSES:
            try:
                model = cls(random_state=seed, max_spatial_samples=max_spatial_samples)
                model.fit(ctr, Xtr, ytr)
                proba = np.clip(model.predict_proba(cte, Xte), 0, 1)
                pred = (proba >= 0.5).astype(int)
                m = ModelEvaluator.evaluate(yte, pred, proba)
                joblib.dump(model, cat_dir / f"{cls.name}.pkl")
                with open(cat_dir / f"{cls.name}_summary.json", "w") as f:
                    json.dump(model.get_summary(), f, indent=2, default=str)
                rows.append({"category": cat, "model": cls.name, "split": scheme,
                             "n_train": len(ytr), "n_test": len(yte), **m})
                logger.info(f"[{cat}] {cls.name}: AUC={m.get('roc_auc', float('nan')):.4f} "
                            f"F1={m['f1']:.4f} kappa={m['kappa']:.4f}")
            except Exception as e:
                logger.exception(f"[{cat}] {cls.name} failed: {e}")
                rows.append({"category": cat, "model": cls.name, "split": scheme,
                             "error": str(e)})

    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "suite_model_comparison.csv", index=False)
    return results
