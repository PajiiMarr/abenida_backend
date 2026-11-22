raise ImportError(
    "DEPRECATED: ModelTrainingPipeline is superseded by "
    "app.training.models.train_model_suite(). See run_suite.py."
)

"""
Phase 3: Model Training Pipeline
SVM → Stacking Ensemble → MARS → GWR → MGWR (per event AND overall)
"""
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
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

from .models import SVMModel, StackingEnsemble, ModelEvaluator
from .models import MARSModeL, GWRModel, MGWRModel, _stratified_subsample

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

    Trains SVM, Stacking Ensemble, MARS, GWR and MGWR for each category
    ('overall' + every event), then classifies 4-level flood risk
    using the best available model (stacking preferred).
    """

    # models trained by train_all(), in order
    MODEL_ORDER = ['svm', 'stacking', 'mars', 'gwr', 'mgwr']

    def __init__(self, category: str = "overall", max_spatial_samples: int = 2000):
        self.category = category
        # GWR / MGWR are O(n^2) in memory -> fit on a stratified subsample,
        # then predict on every point via local weighted least squares.
        self.max_spatial_samples = max_spatial_samples
        self.output_dir = Path(f"./outputs/phase3/{category}")
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

        self.models = {}
        self.metrics = {}
        self.cv_results = {}

        self.risk_classes = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        self.risk_bins = [0, 0.25, 0.50, 0.75, 1.0]

        self.model_descriptions = {
            'svm': (
                "SVM (Support Vector Machine) with an RBF kernel draws a flexible boundary "
                "between flood-prone and non-flood-prone areas based on all features combined."
            ),
            'stacking': (
                "The Stacking Ensemble combines predictions from several base models "
                "(Logistic Regression, SVM, Random Forest) through a meta-model."
            ),
            'mars': (
                "MARS (Multivariate Adaptive Regression Splines) fits piecewise-linear hinge "
                "functions to capture non-linear effects and interactions between features."
            ),
            'gwr': (
                "GWR (Geographically Weighted Regression) fits a separate local regression at "
                "each location so the effect of every feature can vary across space."
            ),
            'mgwr': (
                "MGWR (Multiscale GWR) extends GWR by letting each feature vary over its own "
                "spatial scale (bandwidth)."
            ),
        }

    # =========================================================================
    # DATA
    # =========================================================================
    def load_data(self, phase2_dir: Path = Path("./outputs/phase2")):
        logger.info("=" * 60)
        logger.info(f"Loading ENGINEERED dataset for category: {self.category}")
        logger.info("=" * 60)

        dataset_path = phase2_dir / "full_dataset_engineered.csv"
        if not dataset_path.exists():
            dataset_path = phase2_dir / "full_dataset_engineered_standardized.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found in {phase2_dir}")

        samples_path = phase2_dir / "sample_points/sample_points.shp"
        if not samples_path.exists():
            raise FileNotFoundError(f"Sample points shapefile not found: {samples_path}")

        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} feature rows from {dataset_path.name}")

        logger.info(f"Reading sample points shapefile: {samples_path}")
        samples_gdf = gpd.read_file(samples_path)
        samples_df = pd.DataFrame(samples_gdf.drop(columns='geometry'))
        logger.info(f"Loaded {len(samples_df)} rows from shapefile")

        if 'event' not in df.columns:
            if 'sample_id' in df.columns and 'sample_id' in samples_df.columns:
                df = df.merge(samples_df[['sample_id', 'event']], on='sample_id', how='left')
                logger.info("Merged on sample_id")
            else:
                dedup = samples_df.drop_duplicates(subset=['row', 'col'], keep='first')
                df = df.merge(dedup[['row', 'col', 'event']], on=['row', 'col'], how='left')
                logger.warning("Fell back to (row,col) merge with deduplication")

        logger.info(f"After merge: {len(df)} rows, {df['event'].notna().sum()} matched to an event")

        if self.category != "overall":
            df = df[df['event'] == self.category].reset_index(drop=True)

        logger.info(f"Loaded {len(df)} samples for category '{self.category}'")
        if len(df) < 20:
            raise ValueError(f"Not enough samples ({len(df)}) to train '{self.category}'")

        exclude_cols = ['sample_id', 'row', 'col', 'flood_label', 'event']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        self.feature_names = feature_cols
        self.X = df[feature_cols].values
        self.y = df['flood_label'].values
        self.coords = df[['row', 'col']].values

        if len(np.unique(self.y)) < 2:
            raise ValueError(f"Only one class in '{self.category}' — cannot train")

        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.coords_train, self.coords_test = train_test_split(
            self.X, self.y, self.coords,
            test_size=0.2, random_state=42, stratify=self.y,
        )

        logger.info(f"Train set: {len(self.X_train)} | Test set: {len(self.X_test)}")

    # =========================================================================
    # SVM
    # =========================================================================
    def train_svm(self):
        logger.info(f"\nTraining SVM Model for {self.category}")
        try:
            svm = SVMModel()
            svm.fit(self.X_train, self.y_train, tune_hyperparams=True)

            y_train_proba = svm.predict_proba(self.X_train)
            y_test_proba = svm.predict_proba(self.X_test)
            y_train_pred = y_train_proba[:, 1] if y_train_proba.shape[1] == 2 else y_train_proba[:, 0]
            y_test_pred = y_test_proba[:, 1] if y_test_proba.shape[1] == 2 else y_test_proba[:, 0]
            y_train_class = svm.predict(self.X_train)
            y_test_class = svm.predict(self.X_test)

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
                'status': 'ok',
            }

            logger.info(f"SVM Test AUC: {test_metrics.get('roc_auc', 0):.4f}")
            _save_model(svm, self.output_dir / "svm_model.pkl")

        except Exception as e:
            logger.error(f"SVM training failed for {self.category}: {e}")
            import traceback; traceback.print_exc()
            self.metrics['svm'] = {'status': 'failed', 'error': str(e), 'test': {}}

    # =========================================================================
    # STACKING — trained per event AND overall
    # =========================================================================
    def train_stacking(self):
        logger.info(f"\nTraining Stacking Ensemble for {self.category}")
        try:
            stacking = StackingEnsemble(base_models=None)
            stacking.fit(self.X_train, self.y_train)

            y_train_proba = stacking.predict_proba(self.X_train)
            y_test_proba = stacking.predict_proba(self.X_test)
            y_train_pred = y_train_proba[:, 1] if y_train_proba.shape[1] == 2 else y_train_proba[:, 0]
            y_test_pred = y_test_proba[:, 1] if y_test_proba.shape[1] == 2 else y_test_proba[:, 0]
            y_train_class = stacking.predict(self.X_train)
            y_test_class = stacking.predict(self.X_test)

            train_metrics = ModelEvaluator.evaluate(self.y_train, y_train_class, y_train_pred)
            test_metrics = ModelEvaluator.evaluate(self.y_test, y_test_class, y_test_pred)

            self.models['stacking'] = stacking
            self.metrics['stacking'] = {
                'train': train_metrics,
                'test': test_metrics,
                'summary': stacking.get_summary(),
                'feature_names': self.feature_names,
                'cv_results': self.cv_results.get('stacking', {}),
                'status': 'ok',
            }

            logger.info(f"Stacking Test AUC: {test_metrics.get('roc_auc', 0):.4f}")
            _save_model(stacking, self.output_dir / "stacking_model.pkl")

        except Exception as e:
            logger.error(f"Stacking training failed for {self.category}: {e}")
            import traceback; traceback.print_exc()
            self.metrics['stacking'] = {'status': 'failed', 'error': str(e), 'test': {}}

    # =========================================================================
    # MARS
    # =========================================================================
    def train_mars(self):
        logger.info(f"\nTraining MARS Model for {self.category}")
        try:
            mars = MARSModeL(random_state=42)
            mars.fit(self.X_train, self.y_train.astype(float))

            y_train_proba = mars.predict_proba(self.X_train)[:, 1]
            y_test_proba = mars.predict_proba(self.X_test)[:, 1]
            y_train_class = (y_train_proba >= 0.5).astype(int)
            y_test_class = (y_test_proba >= 0.5).astype(int)

            train_metrics = ModelEvaluator.evaluate(self.y_train, y_train_class, y_train_proba)
            test_metrics = ModelEvaluator.evaluate(self.y_test, y_test_class, y_test_proba)

            self.models['mars'] = mars
            self.metrics['mars'] = {
                'train': train_metrics,
                'test': test_metrics,
                'summary': mars.get_summary(),
                'hinge_functions': mars.get_hinge_functions(),
                'feature_names': self.feature_names,
                'cv_results': {},
                'status': 'ok',
            }

            logger.info(f"MARS Test AUC: {test_metrics.get('roc_auc', 0):.4f} "
                        f"(backend: {mars._backend})")
            try:
                _save_model(mars, self.output_dir / "mars_model.pkl")
            except Exception as e:
                logger.warning(f"MARS trained but could not be pickled: {e}")

        except Exception as e:
            logger.error(f"MARS training failed for {self.category}: {e}")
            import traceback; traceback.print_exc()
            self.metrics['mars'] = {'status': 'failed', 'error': str(e), 'test': {}}

    # =========================================================================
    # GWR / MGWR (shared helpers)
    # =========================================================================
    @staticmethod
    def _predict_at_chunked(model, coords, X, chunk: int = 2000) -> np.ndarray:
        """predict_at() in chunks (keeps the n_test x n_train distance matrix small)."""
        coords = np.asarray(coords, dtype=float)
        X = np.asarray(X, dtype=float)
        out = [model.predict_at(coords[i:i + chunk], X[i:i + chunk])
               for i in range(0, len(X), chunk)]
        return np.clip(np.concatenate(out), 0, 1)

    def _fit_spatial(self, model):
        """Fit a GWR/MGWR on a stratified subsample of the training set."""
        idx = _stratified_subsample(self.y_train, self.max_spatial_samples, 42)
        logger.info(f"  Spatial fit on {len(idx)} / {len(self.y_train)} training samples")
        model.fit(self.coords_train[idx], self.X_train[idx],
                  np.asarray(self.y_train[idx], dtype=float))
        return len(idx)

    def _evaluate_spatial(self, name: str, model, n_fit: int, extra_summary: Dict):
        train_proba = self._predict_at_chunked(model, self.coords_train, self.X_train)
        test_proba = self._predict_at_chunked(model, self.coords_test, self.X_test)
        train_class = (train_proba >= 0.5).astype(int)
        test_class = (test_proba >= 0.5).astype(int)

        train_metrics = ModelEvaluator.evaluate(self.y_train, train_class, train_proba)
        test_metrics = ModelEvaluator.evaluate(self.y_test, test_class, test_proba)

        summary = model.get_summary()
        summary['n_fit_samples'] = int(n_fit)
        summary.update(extra_summary)

        self.models[name] = model
        self.metrics[name] = {
            'train': train_metrics,
            'test': test_metrics,
            'summary': summary,
            'feature_names': self.feature_names,
            'cv_results': {},
            'status': 'ok',
        }
        logger.info(f"{name.upper()} Test AUC: {test_metrics.get('roc_auc', 0):.4f}")

    # =========================================================================
    # GWR
    # =========================================================================
    def train_gwr(self):
        logger.info(f"\nTraining GWR Model for {self.category}")
        try:
            gwr = GWRModel()
            n_fit = self._fit_spatial(gwr)
            if not gwr.is_fitted:
                raise RuntimeError("GWR did not fit")

            # GWRModel silently falls back to Ridge when the spatial fit fails.
            used_ridge = gwr._ridge_fallback is not None
            if used_ridge:
                logger.warning("GWR used Ridge fallback (spatial fit failed)")

            self._evaluate_spatial('gwr', gwr, n_fit,
                                   {'fallback_used': 'ridge' if used_ridge else None})

            # GWRModel.__getstate__ drops the ridge fallback, so a pickled
            # fallback model could not predict. Only pickle real GWR fits.
            if used_ridge:
                logger.warning("Skipping GWR pickle (Ridge fallback is not picklable)")
            else:
                _save_model(gwr, self.output_dir / "gwr_model.pkl")

        except Exception as e:
            logger.error(f"GWR training failed for {self.category}: {e}")
            import traceback; traceback.print_exc()
            self.metrics['gwr'] = {'status': 'failed', 'error': str(e), 'test': {}}

    # =========================================================================
    # MGWR
    # =========================================================================
    def train_mgwr(self):
        logger.info(f"\nTraining MGWR Model for {self.category}")
        try:
            mgwr = MGWRModel()
            n_fit = self._fit_spatial(mgwr)
            if not mgwr.is_fitted:
                raise RuntimeError("MGWR did not fit")

            fallback = mgwr._fallback_used          # None | 'gwr' | 'ridge'
            if fallback:
                logger.warning(f"MGWR used '{fallback}' fallback (backfitting failed)")

            self._evaluate_spatial('mgwr', mgwr, n_fit, {})

            inner_ridge = (mgwr._gwr_fallback is not None
                           and mgwr._gwr_fallback._ridge_fallback is not None)
            if fallback == 'ridge' or inner_ridge:
                logger.warning("Skipping MGWR pickle (Ridge fallback not picklable via GWRModel)")
            else:
                _save_model(mgwr, self.output_dir / "mgwr_model.pkl")

        except Exception as e:
            logger.error(f"MGWR training failed for {self.category}: {e}")
            import traceback; traceback.print_exc()
            self.metrics['mgwr'] = {'status': 'failed', 'error': str(e), 'test': {}}

    # =========================================================================
    # RISK
    # =========================================================================
    def classify_risk(self, probabilities: np.ndarray) -> np.ndarray:
        return np.digitize(probabilities, self.risk_bins[1:], right=True)

    def generate_risk_maps(self):
        logger.info(f"\nGenerating 4-Class Risk Maps for {self.category}")

        if 'stacking' in self.models:
            best_model, best_name = self.models['stacking'], 'stacking'
        elif 'svm' in self.models:
            best_model, best_name = self.models['svm'], 'svm'
        else:
            logger.warning(f"No model available for {self.category} risk classification")
            return

        logger.info(f"  Using '{best_name}' for risk classification (category={self.category})")

        try:
            y_proba = best_model.predict_proba(self.X_test)
            y_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba.flatten()
        except Exception:
            y_proba = np.array(best_model.predict(self.X_test)).flatten()

        y_proba = np.clip(y_proba, 0, 1)
        risk_classes = self.classify_risk(y_proba)
        risk_labels = [self.risk_classes[i] for i in risk_classes]

        risk_df = pd.DataFrame({
            'test_sample': range(len(y_proba)),
            'probability': y_proba,
            'risk_class_number': risk_classes,
            'risk_class_label': risk_labels,
            'actual_label': self.y_test,
            'model_used': best_name,
        })
        risk_df.to_csv(self.output_dir / "risk_classification.csv", index=False)

        class_counts = risk_df['risk_class_label'].value_counts()
        with open(self.output_dir / "risk_class_mapping.json", 'w') as f:
            json.dump({
                'model_used': best_name,
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

        logger.info(f"  Distribution: {class_counts.to_dict()}")

    # =========================================================================
    # SAVE / REPORT
    # =========================================================================
    def save_results(self):
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        comparison = []
        for model_name, metrics in self.metrics.items():
            if 'test' in metrics and metrics.get('status') == 'ok':
                comparison.append({
                    'model': model_name,
                    'accuracy': metrics['test'].get('accuracy', 0),
                    'precision': metrics['test'].get('precision', 0),
                    'recall': metrics['test'].get('recall', 0),
                    'f1': metrics['test'].get('f1', 0),
                    'roc_auc': metrics['test'].get('roc_auc', 0),
                    'kappa': metrics['test'].get('kappa', 0),
                })
            elif metrics.get('status') == 'failed':
                comparison.append({
                    'model': model_name, 'accuracy': 0, 'precision': 0,
                    'recall': 0, 'f1': 0, 'roc_auc': 0, 'kappa': 0,
                })

        pd.DataFrame(comparison).to_csv(self.output_dir / "model_comparison.csv", index=False)

    def generate_report(self):
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.30,
                                   left=0.06, right=0.97, top=0.94, bottom=0.08)
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
                sns.barplot(data=metrics_plot, x='model', y='value', hue='variable', ax=ax_metrics)
                ax_metrics.set_title(f'Model Performance ({self.category})', fontsize=13)
                ax_metrics.set_ylabel('Score'); ax_metrics.set_xlabel('')
                ax_metrics.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                ax_metrics.set_ylim(0, 1)

                roc_values = comparison_df[['model', 'roc_auc']].dropna()
                sns.barplot(data=roc_values, x='model', y='roc_auc', ax=ax_roc)
                ax_roc.set_title('ROC AUC by Model', fontsize=13)
                ax_roc.set_ylabel('AUC Score'); ax_roc.set_xlabel('')
                ax_roc.axhline(y=0.5, color='red', linestyle='--', label='Random')
                ax_roc.set_ylim(0, 1)
                ax_roc.legend(loc='lower right', fontsize=8)

            try:
                risk_df = pd.read_csv(self.output_dir / "risk_classification.csv")
                risk_counts = risk_df['risk_class_label'].value_counts()
                counts_dict = risk_counts.to_dict()
                counts = [counts_dict.get(cls, 0) for cls in self.risk_classes]
                colors = ['green', 'yellow', 'orange', 'red']
                bars = ax_risk.bar(self.risk_classes, counts, color=colors)
                ax_risk.set_title('4-Class Risk Distribution', fontsize=13)
                ax_risk.set_xlabel('Risk Class'); ax_risk.set_ylabel('Count')
                ax_risk.tick_params(axis='x', rotation=45)
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax_risk.text(bar.get_x() + bar.get_width() / 2,
                                     bar.get_height() + 1, str(count),
                                     ha='center', va='bottom')
            except Exception:
                pass

            ax_table.axis('off')
            if len(comparison_df) > 0:
                table_data = comparison_df.round(4).values
                table = ax_table.table(cellText=table_data,
                                       colLabels=comparison_df.columns,
                                       loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.0, 2.2)
                ax_table.set_title('Model Comparison Table', pad=20, fontsize=14)

            plt.savefig(self.output_dir / "model_report.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Report generation failed for {self.category}: {e}")

    def _grade_auc(self, auc: Optional[float]) -> str:
        if auc is None: return "not available"
        if auc >= 0.9: return "excellent"
        if auc >= 0.8: return "good"
        if auc >= 0.7: return "fair"
        if auc >= 0.6: return "weak"
        return "poor"

    def generate_summary_report(self):
        try:
            lines = [f"# Flood Susceptibility Model Report - {self.category.upper()}\n"]
            lines.append("| Model | Status | Test AUC | Rating | Test Accuracy | Test F1 |")
            lines.append("|---|---|---|---|---|---|")

            for name in self.MODEL_ORDER:
                if name not in self.metrics: continue
                m = self.metrics[name]
                status = m.get('status', 'unknown')
                if status != 'ok':
                    lines.append(f"| {name.upper()} | {status} | — | — | — | — |")
                    continue
                test = m.get('test') or {}
                auc = test.get('roc_auc')
                grade = self._grade_auc(auc)
                lines.append(
                    f"| {name.upper()} | ok | {auc:.3f} | {grade} | "
                    f"{test.get('accuracy', 0):.3f} | {test.get('f1', 0):.3f} |"
                )

            risk_path = self.output_dir / "risk_classification.csv"
            if risk_path.exists():
                risk_df = pd.read_csv(risk_path)
                counts = risk_df['risk_class_label'].value_counts()
                total = len(risk_df)
                model_used = risk_df['model_used'].iloc[0] if 'model_used' in risk_df.columns else 'n/a'
                lines.append(f"\n## Flood Risk Classification (model: **{model_used}**)\n")
                for cls in self.risk_classes:
                    count = int(counts.get(cls, 0))
                    pct = (count / total * 100) if total else 0
                    lines.append(f"- **{cls}**: {count} areas ({pct:.1f}%)")

            with open(self.output_dir / "model_report_summary.md", "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.error(f"Summary report generation failed for {self.category}: {e}")

    def train_all(self):
        self.load_data()
        self.train_svm()
        self.train_stacking()          # ← stacking trained per category (event & overall)
        self.train_mars()              # ← MARS
        self.train_gwr()               # ← GWR
        self.train_mgwr()              # ← MGWR
        self.generate_risk_maps()
        self.save_results()
        self.generate_report()
        self.generate_summary_report()
        return self.metrics


def main():
    sample_dir = Path("./outputs/phase2/sample_points")
    samples_path = sample_dir / "sample_points.shp"

    if not samples_path.exists():
        logger.error("Master sample points shapefile not found. Run Phase 2 first.")
        return

    logger.info("Reading master shapefile to extract unique events...")
    samples_gdf = gpd.read_file(samples_path)

    if 'event' not in samples_gdf.columns:
        logger.error("No 'event' column found in sample_points.shp.")
        return

    unique_events = samples_gdf['event'].dropna().unique().tolist()
    categories = ["overall"] + sorted(unique_events)

    logger.info(f"Found {len(categories)} categories to train "
                f"(SVM + Stacking + MARS + GWR + MGWR each): {categories}")

    for category in categories:
        logger.info(f"\n{'='*60}\nStarting all models for category: {category}\n{'='*60}")
        try:
            pipeline = ModelTrainingPipeline(category)
            pipeline.train_all()
        except Exception as e:
            logger.error(f"Training failed for {category}: {e}")
            import traceback; traceback.print_exc()

    logger.info("\nAll training pipelines completed.")


if __name__ == "__main__":
    main()