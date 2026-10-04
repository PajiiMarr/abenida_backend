"""
Phase 3: Model Training Pipeline
GWR → MARS → SVM → MGWR → Stacking Ensemble

Saves models with pickle.dump() so they can be loaded with pickle.load()
without needing joblib.

Note on feature sets:
- GWR and MGWR require a non-singular design matrix. With 47 heavily
  multicollinear engineered features, the local weight matrices become
  singular. They are trained on the raw, physically distinct conditioning
  factors listed in GWR_FEATURE_SET.
- MARS, SVM, and Stacking are trained on the full engineered feature set.
"""
import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from .models import (
    GWRModel, MGWRModel, MARSModeL, SVMModel,
    StackingEnsemble, ModelEvaluator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _save_model(obj, path: Path):
    """Save a model with pickle.dump so it can be loaded with pickle.load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved (pickle) {path} ({path.stat().st_size:,} bytes)")


class ModelTrainingPipeline:
    """
    Complete Phase 3: Model Training Pipeline

    Implements:
    1. GWR (Geographically Weighted Regression) - Spatial non-stationarity
    2. MARS (Multivariate Adaptive Regression Splines) - Non-linear thresholds
    3. SVM (Support Vector Machine) - Classification with RBF kernel
    4. MGWR (Multiscale GWR) - Benchmark model
    5. Stacking Ensemble - Combines all models
    6. 4-class risk classification (Low to Very High) with equal thresholds
    """

    GWR_FEATURE_SET = [
        "dem",
        "slope",
        "aspect",
        "twi",
        "hand",
        "flow_accumulation",
        "distance_to_river",
        "drainage_density",
        "chirps",
        "gpm",
    ]

    def __init__(self):
        self.output_dir = Path("./outputs/phase3")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.X = None
        self.y = None
        self.coords = None
        self.feature_names = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.coords_train = None
        self.coords_test = None

        self.X_gwr_train: Optional[np.ndarray] = None
        self.X_gwr_test: Optional[np.ndarray] = None
        self.coords_gwr_train: Optional[np.ndarray] = None
        self.coords_gwr_test: Optional[np.ndarray] = None
        self.feature_names_gwr: List[str] = []

        self.models = {}
        self.metrics = {}
        self.cv_results = {}

        self.risk_classes = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        self.risk_bins = [0, 0.25, 0.50, 0.75, 1.0]

        self.model_descriptions = {
            'gwr': (
                "GWR (Geographically Weighted Regression) checks whether the relationship "
                "between flood-related features (like elevation and slope) and "
                "flood risk changes from one part of the city to another, instead of assuming "
                "one fixed relationship everywhere."
            ),
            'mars': (
                "MARS (Multivariate Adaptive Regression Splines) looks for threshold effects — "
                "points where a feature (e.g. slope or TWI) starts to matter a lot "
                "more or less to flood risk — rather than assuming a straight-line relationship."
            ),
            'mgwr': (
                "MGWR (Multiscale GWR) lets each feature have its own local scale of "
                "influence, instead of one shared bandwidth for all features. Here it is "
                "fitted with a custom backfitting routine; the intercept varies at a very "
                "local scale while the other effects are close to constant across the area."
            ),
            'svm': (
                "SVM (Support Vector Machine) with an RBF kernel draws a flexible boundary "
                "between flood-prone and non-flood-prone areas based on all features combined, "
                "and tends to be a strong, stable classifier on this kind of tabular data."
            ),
            'stacking': (
                "The Stacking Ensemble combines predictions from several base models "
                "(Logistic Regression, SVM, Random Forest) through a meta-model, usually giving "
                "the most accurate and reliable overall flood susceptibility score."
            ),
        }

    def load_data(self, phase2_dir: Path = Path("./outputs/phase2")):
        logger.info("=" * 60)
        logger.info("Loading ENGINEERED dataset from Phase 2...")
        logger.info("=" * 60)

        dataset_path = phase2_dir / "full_dataset_engineered_standardized.csv"

        if not dataset_path.exists():
            logger.warning("Engineered dataset not found! Falling back to standardized.")
            dataset_path = phase2_dir / "full_dataset_standardized.csv"

        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")

        exclude_cols = ['sample_id', 'row', 'col', 'flood_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        self.feature_names = feature_cols
        self.X = df[feature_cols].values
        self.y = df['flood_label'].values
        self.coords = df[['row', 'col']].values

        logger.info(f"Loaded dataset with {len(self.feature_names)} features")
        logger.info(f"  Features: {', '.join(self.feature_names[:10])}...")

        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.coords_train, self.coords_test = train_test_split(
            self.X, self.y, self.coords,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")

        # ------------------------------------------------------------------
        # Reduced feature set for GWR / MGWR ONLY.
        # MARS, SVM and Stacking keep the full feature set (self.X_train).
        # ------------------------------------------------------------------
        available = [f for f in self.GWR_FEATURE_SET if f in self.feature_names]
        missing = [f for f in self.GWR_FEATURE_SET if f not in self.feature_names]

        if missing:
            logger.warning(
                f"GWR/MGWR feature(s) not found in dataset, skipping: {missing}"
            )

        idx = [self.feature_names.index(f) for f in available]

        # Drop constant columns for GWR/MGWR: they make the local design
        # matrix singular. Other models are unaffected.
        varying = [i for i in idx if np.nanstd(self.X_train[:, i]) > 1e-8]
        dropped = [self.feature_names[i] for i in idx if i not in varying]
        if dropped:
            logger.warning(f"GWR/MGWR: dropping constant feature(s): {dropped}")
        idx = varying
        available = [self.feature_names[i] for i in idx]

        if len(available) < 3:
            logger.error(
                f"Only {len(available)} usable GWR features available. "
                f"GWR/MGWR will not be trained."
            )
            self.X_gwr_train = None
            self.X_gwr_test = None
            self.coords_gwr_train = None
            self.coords_gwr_test = None
            self.feature_names_gwr = []
        else:
            self.X_gwr_train = self.X_train[:, idx]
            self.X_gwr_test = self.X_test[:, idx]
            self.coords_gwr_train = self.coords_train
            self.coords_gwr_test = self.coords_test
            self.feature_names_gwr = available

            logger.info(
                f"GWR/MGWR reduced feature set: {len(available)} features "
                f"({', '.join(available)})"
            )
            logger.info(f"  X_gwr_train shape: {self.X_gwr_train.shape}")
            logger.info(f"  X_gwr_test shape: {self.X_gwr_test.shape}")


    # =========================================================================
    # GWR — reduced feature set
    # =========================================================================
    def train_gwr(self):
        """Train GWR on the reduced (raw) feature set."""
        logger.info("\n" + "=" * 60)
        logger.info("Training GWR Model (reduced raw feature set)")
        logger.info("=" * 60)

        if self.coords_gwr_train is None or self.X_gwr_train is None:
            logger.warning("GWR feature set not available. Skipping GWR.")
            return

        try:
            X_train = self.X_gwr_train
            X_test = self.X_gwr_test
            coords_train = self.coords_gwr_train
            coords_test = self.coords_gwr_test

            logger.info(f"Using GWR features: {X_train.shape}")
            logger.info(f"  Feature names: {self.feature_names_gwr}")

            gwr = GWRModel()
            gwr.fit(coords_train, X_train, self.y_train)

            # In-sample predictions (at training coordinates).
            y_train_pred = gwr.predict_training()

            # Genuine out-of-sample predictions at the test coordinates.
            # The previous version sliced training predictions and compared
            # them to test labels, which is why test AUC was ~0.51.
            y_test_pred = gwr.predict_at(coords_test, X_test)

            train_metrics = ModelEvaluator.evaluate(
                self.y_train,
                (y_train_pred >= 0.5).astype(int),
                y_train_pred,
            )
            test_metrics = ModelEvaluator.evaluate(
                self.y_test,
                (y_test_pred >= 0.5).astype(int),
                y_test_pred,
            )

            self.models["gwr"] = gwr
            self.metrics["gwr"] = {
                "train": train_metrics,
                "test": test_metrics,
                "summary": gwr.get_summary(),
                "feature_names": self.feature_names_gwr,
                "coords_used": True,
            }

            logger.info(f"GWR Training - AUC: {train_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"GWR Testing - AUC: {test_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"GWR R²: {gwr.get_summary().get('r2', 0):.4f}")

            _save_model(gwr, self.output_dir / "gwr_model.pkl")

        except Exception as e:
            logger.error(f"GWR training failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # MARS — full feature set
    # =========================================================================
    def train_mars(self):
        """Train MARS on the full engineered feature set."""
        logger.info("\n" + "=" * 60)
        logger.info("Training MARS Model (full engineered feature set)")
        logger.info("=" * 60)

        try:
            X_train = self.X_train
            X_test = self.X_test
            logger.info(f"Using features: {X_train.shape}")

            mars = MARSModeL(max_degree=2, penalty=3.0, random_state=42)
            mars.fit(X_train, self.y_train)

            y_train_proba = mars.predict_proba(X_train)
            y_test_proba = mars.predict_proba(X_test)

            y_train_pred = y_train_proba[:, 1] if y_train_proba.ndim == 2 else y_train_proba.flatten()
            y_test_pred = y_test_proba[:, 1] if y_test_proba.ndim == 2 else y_test_proba.flatten()

            y_train_class = (y_train_pred > 0.5).astype(int)
            y_test_class = (y_test_pred > 0.5).astype(int)

            train_metrics = ModelEvaluator.evaluate(self.y_train, y_train_class, y_train_pred)
            test_metrics = ModelEvaluator.evaluate(self.y_test, y_test_class, y_test_pred)

            self.models['mars'] = mars
            self.metrics['mars'] = {
                'train': train_metrics,
                'test': test_metrics,
                'summary': mars.get_summary(),
                'feature_importance': (
                    mars.get_feature_importance().to_dict()
                    if hasattr(mars, 'get_feature_importance') else {}
                ),
                'hinge_functions': (
                    mars.get_hinge_functions()
                    if hasattr(mars, 'get_hinge_functions') else []
                ),
                'feature_names': self.feature_names,
                'cv_results': {},
            }

            logger.info(f"MARS Training - AUC: {train_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"MARS Test - AUC: {test_metrics.get('roc_auc', 0):.4f}")

            _save_model(mars, self.output_dir / "mars_model.pkl")

        except Exception as e:
            logger.error(f"MARS training failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # SVM — full feature set
    # =========================================================================
    def train_svm(self):
        """Train SVM on the full engineered feature set."""
        logger.info("\n" + "=" * 60)
        logger.info("Training SVM Model (full engineered feature set)")
        logger.info("=" * 60)

        try:
            X_train = self.X_train
            X_test = self.X_test
            logger.info(f"Using features: {X_train.shape}")

            svm = SVMModel()
            svm.fit(X_train, self.y_train, tune_hyperparams=True)

            y_train_proba = svm.predict_proba(X_train)
            y_test_proba = svm.predict_proba(X_test)
            y_train_pred = y_train_proba[:, 1] if y_train_proba.shape[1] == 2 else y_train_proba[:, 0]
            y_test_pred = y_test_proba[:, 1] if y_test_proba.shape[1] == 2 else y_test_proba[:, 0]
            y_train_class = svm.predict(X_train)
            y_test_class = svm.predict(X_test)

            train_metrics = ModelEvaluator.evaluate(self.y_train, y_train_class, y_train_pred)
            test_metrics = ModelEvaluator.evaluate(self.y_test, y_test_class, y_test_pred)

            self.models['svm'] = svm
            self.metrics['svm'] = {
                'train': train_metrics,
                'test': test_metrics,
                'summary': svm.get_summary(),
                'best_params': svm.best_params,
                'feature_names': self.feature_names,
                'cv_results': {},
            }

            logger.info(f"SVM Training - AUC: {train_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"SVM Test - AUC: {test_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"SVM Best Params: {svm.best_params}")

            _save_model(svm, self.output_dir / "svm_model.pkl")

        except Exception as e:
            logger.error(f"SVM training failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # MGWR — reduced feature set
    # =========================================================================
    def train_mgwr(self):
        """Train the numpy backfitting multiscale GWR on the reduced feature set."""
        logger.info("\n" + "=" * 60)
        logger.info("Training MGWR (reduced raw feature set)")
        logger.info("=" * 60)

        if self.coords_gwr_train is None or self.X_gwr_train is None:
            logger.warning("MGWR feature set not available. Skipping MGWR.")
            return

        try:
            X_train = self.X_gwr_train
            X_test = self.X_gwr_test
            coords_train = self.coords_gwr_train
            coords_test = self.coords_gwr_test

            logger.info(f"Using MGWR features: {X_train.shape}")
            logger.info(f"  Feature names: {self.feature_names_gwr}")

            mgwr = MGWRModel(multi=True)
            mgwr.fit(coords_train, X_train, self.y_train)

            y_train_pred = mgwr.predict_training()
            y_test_pred = mgwr.predict_at(coords_test, X_test)

            train_metrics = ModelEvaluator.evaluate(
                self.y_train,
                (y_train_pred >= 0.5).astype(int),
                y_train_pred,
            )
            test_metrics = ModelEvaluator.evaluate(
                self.y_test,
                (y_test_pred >= 0.5).astype(int),
                y_test_pred,
            )

            summary = mgwr.get_summary()

            self.models['mgwr'] = mgwr
            self.metrics['mgwr'] = {
                'train': train_metrics,
                'test': test_metrics,
                'summary': summary,
                'feature_names': self.feature_names_gwr,
                'coords_used': True,
            }

            logger.info(f"MGWR Training - AUC: {train_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"MGWR Testing  - AUC: {test_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"MGWR R²: {summary.get('r2', 0):.4f}")
            if summary.get("fallback_used"):
                logger.warning(f"MGWR used fallback: {summary['fallback_used']}")

            _save_model(mgwr, self.output_dir / "mgwr_model.pkl")

        except Exception as e:
            logger.error(f"MGWR training failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # STACKING — full feature set
    # =========================================================================
    def train_stacking(self):
        """Train Stacking Ensemble on the full engineered feature set."""
        logger.info("\n" + "=" * 60)
        logger.info("Training Stacking Ensemble (full engineered feature set)")
        logger.info("=" * 60)

        try:
            X_train = self.X_train
            X_test = self.X_test
            logger.info(f"Using features: {X_train.shape}")

            # Rely on StackingEnsemble's internal defaults. The previous
            # version passed an explicit SVC(probability=True, ...) here,
            # which bypassed the internal probability=False default and
            # triggered the sklearn 1.9 deprecation warning 4× per fit.
            stacking = StackingEnsemble(base_models=None)
            stacking.fit(X_train, self.y_train)

            y_train_proba = stacking.predict_proba(X_train)
            y_test_proba = stacking.predict_proba(X_test)
            y_train_pred = y_train_proba[:, 1] if y_train_proba.shape[1] == 2 else y_train_proba[:, 0]
            y_test_pred = y_test_proba[:, 1] if y_test_proba.shape[1] == 2 else y_test_proba[:, 0]
            y_train_class = stacking.predict(X_train)
            y_test_class = stacking.predict(X_test)

            train_metrics = ModelEvaluator.evaluate(self.y_train, y_train_class, y_train_pred)
            test_metrics = ModelEvaluator.evaluate(self.y_test, y_test_class, y_test_pred)

            self.models['stacking'] = stacking
            self.metrics['stacking'] = {
                'train': train_metrics,
                'test': test_metrics,
                'summary': stacking.get_summary(),
                'feature_names': self.feature_names,
                'cv_results': self.cv_results.get('stacking', {}),
            }

            logger.info(f"Stacking Training - AUC: {train_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"Stacking Test - AUC: {test_metrics.get('roc_auc', 0):.4f}")

            _save_model(stacking, self.output_dir / "stacking_model.pkl")

        except Exception as e:
            logger.error(f"Stacking training failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # RISK CLASSIFICATION
    # =========================================================================
    def classify_risk(self, probabilities: np.ndarray) -> np.ndarray:
        return np.digitize(probabilities, self.risk_bins[1:], right=True)

    def generate_risk_maps(self):
        logger.info("\n" + "=" * 60)
        logger.info("Generating 4-Class Risk Maps (Equal Thresholds)")
        logger.info("=" * 60)
        logger.info(f"Risk Classes: {self.risk_classes}")
        logger.info(f"Thresholds: {self.risk_bins}")

        if 'stacking' in self.models:
            best_model = self.models['stacking']
            logger.info("Using Stacking Ensemble for risk classification")
        elif 'svm' in self.models:
            best_model = self.models['svm']
            logger.info("Using SVM for risk classification")
        elif 'mars' in self.models:
            best_model = self.models['mars']
            logger.info("Using MARS for risk classification")
        else:
            logger.warning("No model available for risk classification")
            return

        X_test = self.X_test
        logger.info(f"Using X_test with shape: {X_test.shape}")

        if X_test is None or len(X_test) == 0:
            logger.warning("No test data available for risk classification")
            return

        try:
            y_proba = best_model.predict_proba(X_test)
            if isinstance(y_proba, (list, tuple)):
                y_proba = np.array(y_proba)

            if hasattr(y_proba, 'ndim'):
                if y_proba.ndim == 1:
                    y_proba = np.array(y_proba).flatten()
                elif y_proba.ndim == 2:
                    if y_proba.shape[1] == 2:
                        y_proba = y_proba[:, 1]
                    elif y_proba.shape[1] > 2:
                        y_proba = y_proba[:, 1]
                    else:
                        y_proba = y_proba.flatten()
                else:
                    y_proba = np.array(y_proba).flatten()
            else:
                y_proba = np.array(y_proba).flatten()

        except (AttributeError, NotImplementedError, TypeError) as e:
            logger.warning(f"predict_proba failed: {e}, using predict")
            y_proba = np.array(best_model.predict(X_test)).flatten()

        y_proba = np.clip(y_proba, 0, 1)
        risk_classes = self.classify_risk(y_proba)
        risk_labels = [self.risk_classes[i] for i in risk_classes]

        risk_df = pd.DataFrame({
            'test_sample': range(len(y_proba)),
            'probability': y_proba,
            'risk_class_number': risk_classes,
            'risk_class_label': risk_labels,
        })
        risk_df['actual_label'] = self.y_test

        risk_df.to_csv(self.output_dir / "risk_classification.csv", index=False)

        class_counts = risk_df['risk_class_label'].value_counts()
        logger.info("Risk Class Distribution (Test Set):")
        for cls in self.risk_classes:
            count = class_counts.get(cls, 0)
            pct = count / len(risk_df) * 100 if len(risk_df) > 0 else 0
            logger.info(f"  {cls}: {count} ({pct:.1f}%)")

        with open(self.output_dir / "risk_class_mapping.json", 'w') as f:
            json.dump({
                'classes': self.risk_classes,
                'bins': self.risk_bins,
                'distribution': class_counts.to_dict(),
                'thresholds': {
                    'Low Risk': [0, 0.25],
                    'Medium Risk': [0.25, 0.50],
                    'High Risk': [0.50, 0.75],
                    'Very High Risk': [0.75, 1.0],
                },
            }, f, indent=2)

        logger.info("Risk classification complete!")

    # =========================================================================
    # SAVE / REPORT
    # =========================================================================
    def save_results(self):
        logger.info("\n" + "=" * 60)
        logger.info("Saving Results")
        logger.info("=" * 60)

        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        with open(self.output_dir / "cv_results.json", 'w') as f:
            json.dump(self.cv_results, f, indent=2, default=str)

        comparison = []
        for model_name, metrics in self.metrics.items():
            if 'test' in metrics and metrics['test'] is not None:
                comparison.append({
                    'model': model_name,
                    'accuracy': metrics['test'].get('accuracy', 0),
                    'precision': metrics['test'].get('precision', 0),
                    'recall': metrics['test'].get('recall', 0),
                    'f1': metrics['test'].get('f1', 0),
                    'roc_auc': metrics['test'].get('roc_auc', 0),
                    'kappa': metrics['test'].get('kappa', 0),
                })
            elif 'train' in metrics:
                comparison.append({
                    'model': model_name,
                    'accuracy': metrics['train'].get('accuracy', 0),
                    'precision': metrics['train'].get('precision', 0),
                    'recall': metrics['train'].get('recall', 0),
                    'f1': metrics['train'].get('f1', 0),
                    'roc_auc': metrics['train'].get('roc_auc', 0),
                    'kappa': metrics['train'].get('kappa', 0),
                })

        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(self.output_dir / "model_comparison.csv", index=False)

        summary_data = []
        for model_name, metrics in self.metrics.items():
            if 'summary' in metrics:
                row = {'model': model_name}
                for k, v in metrics['summary'].items():
                    if isinstance(v, (int, float)):
                        row[k] = v
                summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / "model_summary.csv", index=False)

        logger.info(f"Results saved to {self.output_dir}")

    def generate_report(self):
        logger.info("\n" + "=" * 60)
        logger.info("Generating Visualization Report")
        logger.info("=" * 60)

        try:
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(
                2, 3, figure=fig,
                hspace=0.50, wspace=0.30,
                left=0.06, right=0.97, top=0.94, bottom=0.08,
            )

            ax_metrics = fig.add_subplot(gs[0, 0])
            ax_roc = fig.add_subplot(gs[0, 1])
            ax_risk = fig.add_subplot(gs[0, 2])
            ax_table = fig.add_subplot(gs[1, :])

            comparison_df = pd.read_csv(self.output_dir / "model_comparison.csv")

            if len(comparison_df) > 0:
                metrics_plot = comparison_df.melt(
                    id_vars=['model'],
                    value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                )
                sns.barplot(
                    data=metrics_plot, x='model', y='value',
                    hue='variable', ax=ax_metrics,
                )
                ax_metrics.set_title('Model Performance Comparison', fontsize=13)
                ax_metrics.set_ylabel('Score')
                ax_metrics.set_xlabel('')
                ax_metrics.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                ax_metrics.set_ylim(0, 1)

            roc_values = comparison_df[['model', 'roc_auc']].dropna()
            if len(roc_values) > 0:
                sns.barplot(data=roc_values, x='model', y='roc_auc', ax=ax_roc)
                ax_roc.set_title('ROC AUC by Model', fontsize=13)
                ax_roc.set_ylabel('AUC Score')
                ax_roc.set_xlabel('')
                ax_roc.axhline(y=0.5, color='red', linestyle='--', label='Random')
                ax_roc.set_ylim(0, 1)
                ax_roc.legend(loc='lower right', fontsize=8)

            try:
                risk_df = pd.read_csv(self.output_dir / "risk_classification.csv")
                risk_counts = risk_df['risk_class_label'].value_counts()
                if len(risk_counts) > 0:
                    counts_dict = risk_counts.to_dict()
                    counts = [counts_dict.get(cls, 0) for cls in self.risk_classes]
                    colors = ['green', 'yellow', 'orange', 'red']
                    bars = ax_risk.bar(self.risk_classes, counts, color=colors)
                    ax_risk.set_title('4-Class Risk Distribution (Test Set)', fontsize=13)
                    ax_risk.set_xlabel('Risk Class')
                    ax_risk.set_ylabel('Count')
                    ax_risk.tick_params(axis='x', rotation=45)
                    for bar, count in zip(bars, counts):
                        if count > 0:
                            ax_risk.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 1,
                                str(count), ha='center', va='bottom',
                            )
            except Exception:
                ax_risk.text(
                    0.5, 0.5, 'Risk classification data not available',
                    ha='center', va='center', transform=ax_risk.transAxes,
                )

            ax_table.axis('off')
            if len(comparison_df) > 0:
                table_data = comparison_df.round(4).values
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=comparison_df.columns,
                    loc='center',
                    cellLoc='center',
                )
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.0, 2.2)
                ax_table.set_title('Model Comparison Table', pad=20, fontsize=14)

            plt.savefig(self.output_dir / "model_report.png", dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Report saved to {self.output_dir}/model_report.png")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _grade_auc(self, auc: Optional[float]) -> str:
        if auc is None:
            return "not available"
        if auc >= 0.9:
            return "excellent"
        if auc >= 0.8:
            return "good"
        if auc >= 0.7:
            return "fair"
        if auc >= 0.6:
            return "weak"
        return "poor (close to random guessing)"

    def generate_summary_report(self):
        logger.info("\n" + "=" * 60)
        logger.info("Generating Plain-Language Summary Report")
        logger.info("=" * 60)

        try:
            lines = []
            lines.append("# Flood Susceptibility Model Report\n")
            lines.append(
                "This report summarizes how each model performed at predicting "
                "flood susceptibility, in plain language. Technical terms are kept "
                "so the results can be cross-checked against the raw metrics.\n"
            )

            lines.append("## Model Comparison\n")
            lines.append("| Model | Test AUC | Rating | Test Accuracy | Test F1 |")
            lines.append("|---|---|---|---|---|")

            model_order = ['gwr', 'mars', 'svm', 'mgwr', 'stacking']
            for name in model_order:
                if name not in self.metrics:
                    continue
                m = self.metrics[name]
                test = m.get('test') or m.get('train') or {}
                auc = test.get('roc_auc')
                acc = test.get('accuracy')
                f1 = test.get('f1')
                grade = self._grade_auc(auc)
                auc_str = f"{auc:.3f}" if auc is not None else "N/A"
                acc_str = f"{acc:.3f}" if acc is not None else "N/A"
                f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
                lines.append(f"| {name.upper()} | {auc_str} | {grade} | {acc_str} | {f1_str} |")

            lines.append(
                "\n*AUC (Area Under the ROC Curve) measures how well a model tells "
                "flood-prone areas apart from non-flood-prone areas. 0.5 is random "
                "guessing, 1.0 is a perfect score.*\n"
            )

            lines.append("## What Each Model Does\n")
            for name in model_order:
                if name not in self.metrics:
                    continue
                m = self.metrics[name]
                test = m.get('test') or m.get('train') or {}
                auc = test.get('roc_auc')
                grade = self._grade_auc(auc)
                desc = self.model_descriptions.get(name, "")

                lines.append(f"### {name.upper()}")
                lines.append(desc)
                if auc is not None:
                    lines.append(
                        f"\nOn the test set, {name.upper()} scored an AUC of "
                        f"{auc:.3f}, which is considered **{grade}** for telling "
                        f"flood-prone from non-flood-prone areas."
                    )
                r2 = m.get('summary', {}).get('r2')
                if r2 is not None:
                    lines.append(
                        f" Its R² (how much of the variation in flood risk it "
                        f"explains) was {r2:.3f}."
                    )
                fallback = m.get('summary', {}).get('fallback_used')
                if fallback:
                    lines.append(
                        f" *Note: this run used the {fallback} fallback because "
                        f"the primary spatial-regression fit did not converge.*"
                    )
                lines.append("")

            risk_path = self.output_dir / "risk_classification.csv"
            if risk_path.exists():
                risk_df = pd.read_csv(risk_path)
                counts = risk_df['risk_class_label'].value_counts()
                total = len(risk_df)
                lines.append("## Flood Risk Classification (Test Areas)\n")
                lines.append(
                    "Each test location was sorted into one of 4 risk classes "
                    "with equal thresholds (25% intervals):\n\n"
                    "- **Low Risk**: 0-25% probability\n"
                    "- **Medium Risk**: 25-50% probability\n"
                    "- **High Risk**: 50-75% probability\n"
                    "- **Very High Risk**: 75-100% probability\n\n"
                    "Distribution of test areas:\n"
                )
                for cls in self.risk_classes:
                    count = int(counts.get(cls, 0))
                    pct = (count / total * 100) if total else 0
                    lines.append(f"- **{cls}**: {count} areas ({pct:.1f}% of the test set)")
                lines.append("")

            if self.cv_results:
                lines.append("## Cross-Validation Reliability Check\n")
                lines.append(
                    "Cross-validation re-trains each model on different subsets of "
                    "the data to check that its performance is consistent and not "
                    "a fluke of one particular train/test split.\n"
                )
                for name, results in self.cv_results.items():
                    if (
                        isinstance(results, dict)
                        and 'mean' in results
                        and 'roc_auc' in results['mean']
                    ):
                        mean_auc = results['mean']['roc_auc']
                        std_auc = results['std']['roc_auc']
                        stability = "stable" if std_auc < 0.03 else "somewhat variable"
                        lines.append(
                            f"- **{name.upper()}**: average AUC {mean_auc:.3f} "
                            f"(± {std_auc:.3f} across folds) — {stability} across folds."
                        )
                lines.append("")

            best_name, best_auc = None, -1.0
            for name in model_order:
                if name not in self.metrics:
                    continue
                test = self.metrics[name].get('test') or self.metrics[name].get('train') or {}
                auc = test.get('roc_auc')
                if auc is not None and auc > best_auc:
                    best_name, best_auc = name, auc

            if best_name:
                lines.append("## Bottom Line\n")
                lines.append(
                    f"Based on test-set AUC, **{best_name.upper()}** performed best "
                    f"(AUC = {best_auc:.3f}) and is the most reliable model here for "
                    f"producing the final flood susceptibility map."
                )

            report_text = "\n".join(lines)
            with open(self.output_dir / "model_report_summary.md", "w") as f:
                f.write(report_text)

            logger.info(f"Summary report saved to {self.output_dir}/model_report_summary.md")

        except Exception as e:
            logger.error(f"Summary report generation failed: {e}")
            import traceback
            traceback.print_exc()

    def train_all(self, enable_mgwr: bool = True):
        self.load_data()

        self.train_gwr()       # reduced features
        self.train_mars()      # full features
        self.train_svm()       # full features

        if enable_mgwr:
            self.train_mgwr()  # reduced features — slow, may fail, opt-in
        else:
            logger.info("\n" + "=" * 60)
            logger.info("MGWR SKIPPED (--no-mgwr was passed)")
            logger.info("=" * 60)

        self.train_stacking()  # full features

        self.generate_risk_maps()
        self.save_results()
        self.generate_report()
        self.generate_summary_report()

        return self.metrics
    

def main():
    import sys
    enable_mgwr = "--no-mgwr" not in sys.argv 
    pipeline = ModelTrainingPipeline()
    pipeline.train_all(enable_mgwr=enable_mgwr)


if __name__ == "__main__":
    main()