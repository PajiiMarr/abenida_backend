"""
Machine Learning Models for Flood Susceptibility
Phase 3: GWR, MGWR, MARS, SVM, and Stacking Ensemble
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Tuple, Optional, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, r2_score,
    mean_squared_error, mean_absolute_error, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _gwr_local_predict(coords_train, X_train_scaled, y_train,
                       coords_test, X_test_scaled,
                       bandwidth, kernel='bisquare', fixed=False,
                       ridge_eps=1e-8):
    """
    Manual GWR out-of-sample prediction via local weighted least squares.

    Why this exists: mgwr's GWR.predict(coords, X) is buggy when applied to
    coordinates outside the training set. Its internal .predictions
    property multiplies a test-point projection matrix by a params matrix
    of mismatched shape and raises
    "operands could not be broadcast together with shapes (n_test, k+1) (n_test, k-1)".

    We compute the local fits ourselves. For each test location s0:
        W(s0) = diag(kernel(d(s0, s_i) / bandwidth))
        beta(s0) = (X^T W X + eps*I)^-1 X^T W y
        y_hat(s0) = [1, x(s0)] @ beta(s0)

    Args:
        coords_train: (n_train, 2) training coordinates (already jittered)
        X_train_scaled: (n_train, k) standardized training features
        y_train: (n_train,) training targets
        coords_test: (n_test, 2) test coordinates
        X_test_scaled: (n_test, k) standardized test features
        bandwidth: k (adaptive) or distance threshold (fixed)
        kernel: 'bisquare', 'gaussian', or 'exponential'
        fixed: False = adaptive (bandwidth is # neighbors), True = fixed distance
        ridge_eps: small ridge added to normal equations for stability

    Returns:
        (n_test,) array of predictions.
    """
    from scipy.spatial.distance import cdist

    coords_train = np.asarray(coords_train, dtype=float)
    coords_test = np.asarray(coords_test, dtype=float)
    X_train_scaled = np.asarray(X_train_scaled, dtype=float)
    X_test_scaled = np.asarray(X_test_scaled, dtype=float)
    y_train = np.asarray(y_train, dtype=float).flatten()

    n_train = coords_train.shape[0]
    n_test = coords_test.shape[0]

    # Design matrices with intercept column prepended.
    Xt = np.column_stack([np.ones(n_train), X_train_scaled])
    Xs = np.column_stack([np.ones(n_test), X_test_scaled])

    # Pairwise distances from each test point to each training point.
    D = cdist(coords_test, coords_train)

    # Build kernel weights W (n_test, n_train).
    if fixed:
        t = D / max(float(bandwidth), 1e-9)
        if kernel == 'gaussian':
            W = np.exp(-0.5 * t ** 2)
        elif kernel == 'exponential':
            W = np.exp(-t)
        else:  # bisquare
            W = np.where(t < 1, (1 - t ** 2) ** 2, 0.0)
    else:
        k_nb = max(2, int(round(float(bandwidth))))
        k_nb = min(k_nb, n_train - 1)
        kth = np.partition(D, k_nb, axis=1)[:, k_nb]
        kth = np.where(kth > 0, kth, 1.0)
        t = D / kth[:, None]
        if kernel == 'bisquare':
            W = np.where(t < 1, (1 - t ** 2) ** 2, 0.0)
        elif kernel == 'gaussian':
            W = np.exp(-0.5 * t ** 2)
        else:  # exponential
            W = np.exp(-t)

    preds = np.zeros(n_test)
    k1 = Xt.shape[1]
    eye = ridge_eps * np.eye(k1)
    for i in range(n_test):
        wi = W[i]
        XtW = Xt.T * wi           # (k+1, n_train)
        XtWX = XtW @ Xt + eye     # (k+1, k+1)
        XtWy = XtW @ y_train      # (k+1,)
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        preds[i] = Xs[i] @ beta
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
    squares (see _gwr_local_predict) rather than through mgwr's own
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
        """
        Genuine out-of-sample predictions at arbitrary coordinates.

        Uses manual local WLS to avoid mgwr's broken GWR.predict(), which
        raises "operands could not be broadcast together" when applied to
        external coordinates.
        """
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
        self._Xd_train = None          # set in fit(); used by predict_at()
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

            logger.info(f"MGWR fitted. R²: {self._r2:.4f}, bandwidths: "
                        f"{dict(zip(['intercept'] + [f'x{i}' for i in range(k - 1)], bw))}")
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

        # Primary path: numpy backfit partial-residual smoother.
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

        # Fallback paths.
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
            logger.info(f"MARS (R earth) fitted: {backend.nterms_} terms")
            return self
        except ImportError as e:
            logger.info(f"rpy2 unavailable ({e}) — trying pymars")
        except Exception as e:
            logger.warning(f"R earth unusable ({type(e).__name__}: {e})")

        try:
            from pymars import Earth
            X_fit, y_fit = self._subsample(X, y)
            deg = 1 if X_fit.shape[1] > 15 else self.max_degree
            self.model = Earth(
                max_degree=deg, penalty=self.penalty, max_terms=self.max_terms,
            )
            self.model.fit(X_fit, y_fit)
            self.is_fitted = True
            self._backend = "pymars"
            nterms = (getattr(self.model, "nterms_", None)
                      or getattr(self.model, "n_terms", None)
                      or len(getattr(self.model, "coef_", [])))
            logger.info(f"MARS (pymars) fitted: {nterms} terms")
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
            logger.info("DecisionTreeRegressor fitted as MARS fallback")
            return self
        except Exception as e:
            logger.error(f"DecisionTree fallback failed: {e}")

        self.model = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=self.random_state,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self._backend = "forest"
        logger.info("RandomForestRegressor fitted as final MARS fallback")
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

    def get_summary(self):
        if not self.is_fitted:
            return {"status": "not_fitted"}
        return {
            "model_type": type(self.model).__name__,
            "backend": self._backend,
            "max_degree": self.max_degree,
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
        self.is_fitted = False
        self.scaler = None

    def fit(self, X, y, tune_hyperparams: bool = True, cv: int = 5):
        logger.info(f"Fitting SVM with {X.shape[0]} samples, {X.shape[1]} features")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if tune_hyperparams:
            base_svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
            grid = GridSearchCV(
                base_svm,
                {'C': [0.1, 1, 10], 'gamma': [0.1, 1]},
                cv=min(cv, 3), scoring='roc_auc', n_jobs=-1, verbose=0,
            )
            grid.fit(X_scaled, y)
            self.best_params = dict(grid.best_params_)
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

        self.model = CalibratedClassifierCV(
            inner_svm, method='sigmoid', cv=3, ensemble=False,
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
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_proba))
            metrics['mae'] = mean_absolute_error(y_true, y_proba)
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