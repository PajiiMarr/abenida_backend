"""
Phase 3: Model Training Pipeline
SVM → Stacking Ensemble
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
    1. SVM (Support Vector Machine) - Classification with RBF kernel
    2. Stacking Ensemble - Combines multiple base models
    3. 4-class risk classification (Low to Very High) with equal thresholds
    """

    def __init__(self, category: str = "overall"):
        self.category = category
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
        logger.info(f"Loading ENGINEERED dataset for category: {self.category}")
        logger.info("=" * 60)

        # Prefer the freshly generated engineered CSV
        dataset_path = phase2_dir / "full_dataset_engineered.csv"
        if not dataset_path.exists():
            dataset_path = phase2_dir / "full_dataset_engineered_standardized.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found in {phase2_dir}")

        samples_path = phase2_dir / "sample_points/sample_points.shp"
        if not samples_path.exists():
            raise FileNotFoundError(f"Sample points shapefile not found: {samples_path}")

        # 1. Load feature dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} feature rows from {dataset_path.name}")

        # 2. Load event labels from shapefile
        logger.info(f"Reading sample points shapefile: {samples_path}")
        samples_gdf = gpd.read_file(samples_path)
        samples_df = pd.DataFrame(samples_gdf.drop(columns='geometry'))
        logger.info(f"Loaded {len(samples_df)} rows from shapefile")

        # 3. Merge by sample_id (unique key). Fall back to (row,col) only if sample_id is missing.
        if 'event' not in df.columns:
            if 'sample_id' in df.columns and 'sample_id' in samples_df.columns:
                df = df.merge(samples_df[['sample_id', 'event']], on='sample_id', how='left')
                logger.info("Merged on sample_id")
            else:
                # Deduplicate on (row, col) to avoid cartesian product
                dedup = samples_df.drop_duplicates(subset=['row', 'col'], keep='first')
                df = df.merge(dedup[['row', 'col', 'event']], on=['row', 'col'], how='left')
                logger.warning("Fell back to (row,col) merge with deduplication")

        logger.info(f"After merge: {len(df)} rows, {df['event'].notna().sum()} matched to an event")

        # 4. Filter by category
        if self.category != "overall":
            df = df[df['event'] == self.category].reset_index(drop=True)

        logger.info(f"Loaded {len(df)} samples for category '{self.category}'")

        if len(df) < 20:
            raise ValueError(f"Not enough samples ({len(df)}) to train '{self.category}'")

        # 5. Extract features and labels
        exclude_cols = ['sample_id', 'row', 'col', 'flood_label', 'event']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        self.feature_names = feature_cols
        self.X = df[feature_cols].values
        self.y = df['flood_label'].values
        self.coords = df[['row', 'col']].values

        if len(np.unique(self.y)) < 2:
            raise ValueError(f"Only one class in '{self.category}' — cannot train")

        # 6. Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.coords_train, self.coords_test = train_test_split(
            self.X, self.y, self.coords,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        logger.info(f"Train set: {len(self.X_train)} samples | Test set: {len(self.X_test)} samples")

        
    # =========================================================================
    # SVM — full feature set
    # =========================================================================
    def train_svm(self):
        """Train SVM on the full engineered feature set."""
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
            }

            logger.info(f"SVM Test AUC: {test_metrics.get('roc_auc', 0):.4f}")
            _save_model(svm, self.output_dir / "svm_model.pkl")

        except Exception as e:
            logger.error(f"SVM training failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # STACKING — full feature set
    # =========================================================================
    def train_stacking(self):
        """Train Stacking Ensemble on the full engineered feature set."""
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
            }

            logger.info(f"Stacking Test AUC: {test_metrics.get('roc_auc', 0):.4f}")
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
        logger.info(f"\nGenerating 4-Class Risk Maps for {self.category}")
        
        if 'stacking' in self.models:
            best_model = self.models['stacking']
        elif 'svm' in self.models:
            best_model = self.models['svm']
        else:
            logger.warning("No model available for risk classification")
            return

        try:
            y_proba = best_model.predict_proba(self.X_test)
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            else:
                y_proba = y_proba.flatten()
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
        })
        risk_df.to_csv(self.output_dir / "risk_classification.csv", index=False)

        class_counts = risk_df['risk_class_label'].value_counts()
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

    # =========================================================================
    # SAVE / REPORT
    # =========================================================================
    def save_results(self):
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        comparison = []
        for model_name, metrics in self.metrics.items():
            if 'test' in metrics:
                comparison.append({
                    'model': model_name,
                    'accuracy': metrics['test'].get('accuracy', 0),
                    'precision': metrics['test'].get('precision', 0),
                    'recall': metrics['test'].get('recall', 0),
                    'f1': metrics['test'].get('f1', 0),
                    'roc_auc': metrics['test'].get('roc_auc', 0),
                    'kappa': metrics['test'].get('kappa', 0),
                })

        pd.DataFrame(comparison).to_csv(self.output_dir / "model_comparison.csv", index=False)

    def generate_report(self):
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.30, left=0.06, right=0.97, top=0.94, bottom=0.08)
            ax_metrics = fig.add_subplot(gs[0, 0])
            ax_roc = fig.add_subplot(gs[0, 1])
            ax_risk = fig.add_subplot(gs[0, 2])
            ax_table = fig.add_subplot(gs[1, :])

            comparison_df = pd.read_csv(self.output_dir / "model_comparison.csv")

            if len(comparison_df) > 0:
                metrics_plot = comparison_df.melt(id_vars=['model'], value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
                sns.barplot(data=metrics_plot, x='model', y='value', hue='variable', ax=ax_metrics)
                ax_metrics.set_title(f'Model Performance ({self.category})', fontsize=13)
                ax_metrics.set_ylabel('Score')
                ax_metrics.set_xlabel('')
                ax_metrics.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                ax_metrics.set_ylim(0, 1)

                roc_values = comparison_df[['model', 'roc_auc']].dropna()
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
                counts_dict = risk_counts.to_dict()
                counts = [counts_dict.get(cls, 0) for cls in self.risk_classes]
                colors = ['green', 'yellow', 'orange', 'red']
                bars = ax_risk.bar(self.risk_classes, counts, color=colors)
                ax_risk.set_title('4-Class Risk Distribution', fontsize=13)
                ax_risk.set_xlabel('Risk Class')
                ax_risk.set_ylabel('Count')
                ax_risk.tick_params(axis='x', rotation=45)
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax_risk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(count), ha='center', va='bottom')
            except Exception:
                pass

            ax_table.axis('off')
            if len(comparison_df) > 0:
                table_data = comparison_df.round(4).values
                table = ax_table.table(cellText=table_data, colLabels=comparison_df.columns, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.0, 2.2)
                ax_table.set_title('Model Comparison Table', pad=20, fontsize=14)

            plt.savefig(self.output_dir / "model_report.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Report generation failed: {e}")

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
            lines.append("| Model | Test AUC | Rating | Test Accuracy | Test F1 |")
            lines.append("|---|---|---|---|---|")

            for name in ['svm', 'stacking']:
                if name not in self.metrics: continue
                m = self.metrics[name]
                test = m.get('test') or {}
                auc = test.get('roc_auc')
                grade = self._grade_auc(auc)
                lines.append(f"| {name.upper()} | {auc:.3f} | {grade} | {test.get('accuracy', 0):.3f} | {test.get('f1', 0):.3f} |")

            risk_path = self.output_dir / "risk_classification.csv"
            if risk_path.exists():
                risk_df = pd.read_csv(risk_path)
                counts = risk_df['risk_class_label'].value_counts()
                total = len(risk_df)
                lines.append("\n## Flood Risk Classification\n")
                for cls in self.risk_classes:
                    count = int(counts.get(cls, 0))
                    pct = (count / total * 100) if total else 0
                    lines.append(f"- **{cls}**: {count} areas ({pct:.1f}%)")

            with open(self.output_dir / "model_report_summary.md", "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.error(f"Summary report generation failed: {e}")

    def train_all(self):
        self.load_data()
        self.train_svm()
        self.train_stacking()
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

    # Read the master .shp to find all unique events dynamically
    logger.info("Reading master shapefile to extract unique events...")
    samples_gdf = gpd.read_file(samples_path)
    
    if 'event' not in samples_gdf.columns:
        logger.error("No 'event' column found in sample_points.shp.")
        return
        
    unique_events = samples_gdf['event'].dropna().unique().tolist()

    # Ensure 'overall' is first in the list
    categories = ["overall"] + sorted(unique_events)

    logger.info(f"Found {len(categories)} categories to train: {categories}")

    for category in categories:
        logger.info(f"\n{'='*60}\nStarting training for category: {category}\n{'='*60}")
        try:
            pipeline = ModelTrainingPipeline(category)
            pipeline.train_all()
        except Exception as e:
            logger.error(f"Training failed for {category}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\nAll training pipelines completed.")


if __name__ == "__main__":
    main()