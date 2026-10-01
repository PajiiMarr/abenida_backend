"""
Machine Learning Models for Flood Susceptibility
Phase 3: GWR, MARS, SVM, and Stacking Ensemble
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Tuple, Optional, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV  # ← ADDED
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

import warnings
import numpy as np
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=UserWarning, module="spglm")
warnings.filterwarnings("ignore", category=LinAlgWarning, module="spglm")


class GWRFallbackResults:
    """
    Picklable stand-in for mgwr's GWRResults object, used when the real
    GWR/mgwr fit fails (e.g. singular bandwidth matrix) and we fall back
    to a plain Ridge regression instead.

    NOTE: this MUST be a real module-level class (not created dynamically
    with `type(...)`) or joblib/pickle cannot serialize it, since pickling
    a class requires it to be importable by its module path
    (`app.models.GWRFallbackResults`).
    """

    def __init__(self, predy, params, R2, R2_adj, aicc, n, k):
        self.predy = predy
        self.params = params
        self.R2 = R2
        self.R2_adj = R2_adj
        self.aicc = aicc
        self.n = n
        self.k = k

class GWRModel:
    """
    Geographically Weighted Regression wrapper
    Uses mgwr library for spatial regression
    """

    def __init__(self, bandwidth: Optional[float] = None):
        self.bandwidth = bandwidth
        self.model = None
        self.results = None
        self.is_fitted = False

    def fit(self, coords: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Fit GWR model with fallback for singular matrix issues.
        Includes coordinate jitter for duplicate coordinates and bandwidth validation.
        """
        try:
            from mgwr.gwr import GWR
            from mgwr.sel_bw import Sel_BW

            logger.info(f"Fitting GWR with {X.shape[0]} samples, {X.shape[1]} features")

            # Handle duplicate coordinates
            coords_clean = coords.copy()
            unique_coords = np.unique(coords_clean, axis=0)
            if len(unique_coords) < len(coords_clean):
                logger.warning(f"Found {len(coords_clean) - len(unique_coords)} duplicate coordinates. Adding jitter.")
                noise = np.random.normal(0, 1e-6, coords_clean.shape)
                coords_clean = coords_clean + noise
            
            # Check if all coordinates are identical
            if np.std(coords_clean, axis=0).sum() < 1e-10:
                logger.warning("All coordinates are identical. Adding larger jitter.")
                coords_clean = coords_clean + np.random.normal(0, 0.001, coords_clean.shape)

            # mgwr requires y to be shape (n, 1)
            y_col = np.asarray(y).reshape(-1, 1)

            # Add small regularization to prevent singular matrix
            X_reg = np.column_stack([X, np.random.normal(0, 1e-10, X.shape[0])])

            # Select bandwidth if not provided
            if self.bandwidth is None:
                try:
                    bw_selector = Sel_BW(coords_clean, y_col, X_reg)
                    self.bandwidth = bw_selector.search(criterion='AICc')
                    logger.info(f"Selected bandwidth: {self.bandwidth:.4f}")
                except Exception as e:
                    logger.warning(f"Bandwidth selection failed: {e}, using default")
                    from scipy.spatial import distance_matrix
                    dist_matrix = distance_matrix(coords_clean, coords_clean)
                    self.bandwidth = max(np.percentile(dist_matrix, 30), 0.01)
                    logger.info(f"Using fallback bandwidth: {self.bandwidth:.4f}")

            # Validate bandwidth is not zero
            if self.bandwidth <= 0:
                logger.warning(f"Bandwidth is {self.bandwidth}, setting to 0.1")
                self.bandwidth = 0.1

            # Fit GWR
            self.model = GWR(coords_clean, y_col, X_reg, self.bandwidth)
            self.results = self.model.fit()
            self.is_fitted = True

            logger.info(f"GWR model fitted successfully. R²: {self.results.R2:.4f}")

        except Exception as e:
            logger.error(f"GWR fitting failed: {e}")
            # Use Ridge regression as fallback
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
            ridge.fit(X, y)
            r2 = ridge.score(X, y)
            n, k = X.shape[0], X.shape[1]
            denom = (n - k - 1)
            adj_r2 = r2 - (1 - r2) * k / denom if denom > 0 else r2
            self.results = GWRFallbackResults(
                predy=ridge.predict(X),
                params=ridge.coef_,
                R2=r2,
                R2_adj=adj_r2,
                aicc=None,
                n=n,
                k=k,
            )
            self.is_fitted = True
            logger.info("Using Ridge regression as fallback for GWR")


    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict using fitted GWR"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        if self.results is None:
            raise ValueError("Model not fitted")
        predy = np.asarray(self.results.predy)
        return predy.flatten()


    def get_summary(self) -> Dict:
        """Get model summary"""
        if not self.is_fitted or self.results is None:
            return {"status": "not_fitted"}
        return {
            "bandwidth": self.bandwidth,
            "r2": getattr(self.results, "R2", None),
            "r2_adj": getattr(self.results, "R2_adj", None),
            "aicc": getattr(self.results, "aicc", None),
            "n": getattr(self.results, "n", None),
            "k": getattr(self.results, "k", None),
        }
    
class MARSModeL(BaseEstimator, RegressorMixin):
    """
    Multivariate Adaptive Regression Splines
    Uses scikit-learn's DecisionTreeRegressor as fallback when py-earth
    is not installed.

    Inherits BaseEstimator/RegressorMixin so sklearn.base.clone() and
    cross-validation utilities can work with it correctly. Do NOT rename
    constructor args before storing them (BaseEstimator's get_params
    introspects __init__ directly) or clone() will silently break again.
    """

    def __init__(self, max_degree: int = 2, penalty: float = 3.0, random_state: int = 42):
        self.max_degree = max_degree
        self.penalty = penalty
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self._estimator_type = "regressor"

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit MARS model"""
        try:
            # Try to import Earth
            try:
                from pymars import Earth
                logger.info(f"Fitting MARS with {X.shape[0]} samples, {X.shape[1]} features")
                self.model = Earth(
                    max_degree=self.max_degree,
                    penalty=self.penalty,
                    max_terms=50, 
                )
                self.model.fit(X, y)
                self.is_fitted = True
                logger.info(f"MARS model fitted: {self.model.nterms_} terms")
            except ImportError:
                # Fallback to DecisionTree
                from sklearn.tree import DecisionTreeRegressor
                logger.info("Earth not available, using DecisionTreeRegressor as fallback")
                self.model = DecisionTreeRegressor(
                    max_depth=8,
                    min_samples_split=20,
                    random_state=self.random_state
                )
                self.model.fit(X, y)
                self.is_fitted = True
                logger.info("DecisionTreeRegressor fitted as MARS fallback")
        except Exception as e:
            logger.error(f"MARS fitting failed: {e}")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=self.random_state
            )
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("RandomForestRegressor fitted as final fallback")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (thresholded at 0.5)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        preds = np.asarray(self.model.predict(X)).flatten()
        preds = np.clip(preds, 0, 1)
        # Use 0.5 threshold, but handle edge cases
        return (preds >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict flood probability (continuous, clipped to [0, 1])
        Returns 2D array with shape (n_samples, 2) for sklearn compatibility
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        preds = np.asarray(self.model.predict(X)).flatten()
        preds = np.clip(preds, 0, 1)
        
        # Return 2D array: [prob_class_0, prob_class_1]
        # Class 0 = Non-flood, Class 1 = Flood
        proba = np.column_stack([1 - preds, preds])
        return proba

    def get_summary(self) -> Dict:
        """Get model summary"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        return {
            "model_type": type(self.model).__name__,
            "max_degree": self.max_degree,
            "penalty": self.penalty,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance. Only meaningful for tree-based fallbacks
        (DecisionTreeRegressor / RandomForestRegressor); returns an empty
        DataFrame if the underlying model has no `feature_importances_`
        (e.g. a real Earth/py-earth model, which uses hinge functions
        instead of impurity-based importances).
        """
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": [f"Feature_{i}" for i in range(len(importances))],
            "importance": importances
        }).sort_values("importance", ascending=False)

    def get_hinge_functions(self) -> list:
        """
        Only populated when the real Earth/py-earth model is used.
        Returns an empty list for tree-based fallbacks since they have
        no hinge/basis function representation.
        """
        if not self.is_fitted:
            return []
        if hasattr(self.model, "summary"):
            try:
                return [str(self.model.summary())]
            except Exception:
                return []
        return []

class SVMModel(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine with RBF kernel
    Uses GridSearchCV for hyperparameter tuning

    Inherits BaseEstimator/ClassifierMixin so sklearn.base.clone() works
    correctly during spatial cross-validation. Without this, clone()
    fails (no get_params/set_params) and the CV loop silently falls back
    to reusing the same already-fitted model across every fold, which
    quietly invalidates the cross-validated AUC numbers.
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

    def fit(self, X: np.ndarray, y: np.ndarray,
            tune_hyperparams: bool = True, cv: int = 5):
        """Fit SVM model with optional hyperparameter tuning"""
        logger.info(f"Fitting SVM with {X.shape[0]} samples, {X.shape[1]} features")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if tune_hyperparams:
            # Create base SVM
            base_svm = SVC(
                kernel='rbf',
                class_weight='balanced',
                random_state=42
            )

            # Grid search over SVC parameters first
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': [0.1, 1]
            }

            # Do grid search on SVC first
            grid_search = GridSearchCV(
                base_svm,
                param_grid,
                cv=min(cv, 3),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_scaled, y)
            
            # Get best parameters
            best_C = grid_search.best_params_['C']
            best_gamma = grid_search.best_params_['gamma']
            self.best_params = {'C': best_C, 'gamma': best_gamma}
            
            logger.info(f"Best parameters: {self.best_params}")
            
            # Create final SVM with best parameters
            best_svm = SVC(
                C=best_C,
                gamma=best_gamma,
                kernel='rbf',
                class_weight='balanced',
                random_state=42
            )
            
            # Wrap with CalibratedClassifierCV for probability outputs
            self.model = CalibratedClassifierCV(
                best_svm,
                method='sigmoid',
                cv=3,
                ensemble=False
            )
            self.model.fit(X_scaled, y)
            
        else:
            # Create base SVM without probability
            base_svm = SVC(
                C=self.C,
                gamma=self.gamma,
                kernel=self.kernel,
                class_weight='balanced',
                random_state=42
            )

            # Wrap with CalibratedClassifierCV for probability outputs
            self.model = CalibratedClassifierCV(
                base_svm,
                method='sigmoid',
                cv=3,
                ensemble=False
            )
            self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info("SVM model fitted successfully")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_fitted:
            return pd.DataFrame()

        # Get the base estimator from CalibratedClassifierCV
        if hasattr(self.model, 'base_estimator'):
            base_est = self.model.base_estimator
        elif hasattr(self.model, 'best_estimator_'):
            base_est = self.model.best_estimator_
            if hasattr(base_est, 'base_estimator'):
                base_est = base_est.base_estimator
        else:
            base_est = self.model

        # Get coefficients
        coef = getattr(base_est, 'coef_', None)
        if coef is not None and len(coef.shape) == 2:
            coef = np.abs(coef[0])

        if coef is not None:
            return pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(coef))],
                'importance': coef
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()

    def get_summary(self) -> Dict:
        """Get model summary"""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        # Get base estimator
        if hasattr(self.model, 'base_estimator'):
            base_svm = self.model.base_estimator
        else:
            base_svm = self.model

        return {
            'best_params': self.best_params or {'C': self.C, 'gamma': self.gamma},
            'kernel': getattr(base_svm, 'kernel', self.kernel),
            'n_support_vectors': len(base_svm.support_) if hasattr(base_svm, 'support_') else 0,
            'calibrated': True,
            'calibration_method': 'sigmoid',
        }
  
class StackingEnsemble:
    """
    Stacked Ensemble Model using sklearn's StackingClassifier
    """

    def __init__(self, base_models: List = None):
        self.base_models = base_models
        self.stacked_model = None
        self.is_fitted = False
        self.scaler = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit stacking ensemble"""
        logger.info(f"Fitting Stacking Ensemble with {X_train.shape[0]} samples")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        from sklearn.calibration import CalibratedClassifierCV

        # Create calibrated SVM
        base_svm = SVC(
            kernel='rbf',
            C=10,
            gamma=0.1,
            class_weight='balanced',
            random_state=42
        )
        calibrated_svm = CalibratedClassifierCV(
            base_svm,
            method='sigmoid',
            cv=3,
            ensemble=False
        )

        if self.base_models is None:
            base_models = [
                ('lr', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
                ('svm', calibrated_svm),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'))
            ]
        else:
            if isinstance(self.base_models, dict):
                base_models = list(self.base_models.items())
            elif isinstance(self.base_models, list):
                base_models = self.base_models
            else:
                base_models = [('model', self.base_models)]

        meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')

        self.stacked_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3,
            stack_method='predict_proba'
        )

        self.stacked_model.fit(X_scaled, y_train)
        self.is_fitted = True

        logger.info("Stacking ensemble fitted successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.stacked_model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.stacked_model.predict_proba(X_scaled)

    def get_summary(self) -> Dict:
        """Get model summary"""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        return {
            'n_estimators': len(self.stacked_model.estimators_) if hasattr(self.stacked_model, 'estimators_') else 0,
            'meta_estimator': str(self.stacked_model.final_estimator_) if hasattr(self.stacked_model, 'final_estimator_') else None,
        }

class ModelEvaluator:
    """
    Model evaluation utilities
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None) -> Dict:
        """Compute classification metrics"""
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
    def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """Perform cross-validation"""
        scores = {
            'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
            'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc'),
            'f1': cross_val_score(model, X, y, cv=cv, scoring='f1_weighted'),
        }

        return {
            k: {
                'mean': v.mean() if len(v) > 0 else 0,
                'std': v.std() if len(v) > 0 else 0,
                'scores': v.tolist() if len(v) > 0 else []
            }
            for k, v in scores.items()
        }