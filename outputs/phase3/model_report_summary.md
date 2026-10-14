# Flood Susceptibility Model Report

This report summarizes how each model performed at predicting flood susceptibility, in plain language. Technical terms are kept so the results can be cross-checked against the raw metrics.

## Model Comparison

| Model | Test AUC | Rating | Test Accuracy | Test F1 |
|---|---|---|---|---|
| GWR | 0.676 | weak | 0.605 | 0.605 |
| MARS | 0.670 | weak | 0.605 | 0.602 |
| SVM | 0.679 | weak | 0.621 | 0.621 |
| MGWR | 0.747 | fair | 0.676 | 0.670 |
| STACKING | 0.752 | fair | 0.656 | 0.656 |

*AUC (Area Under the ROC Curve) measures how well a model tells flood-prone areas apart from non-flood-prone areas. 0.5 is random guessing, 1.0 is a perfect score.*

## What Each Model Does

### GWR
GWR (Geographically Weighted Regression) checks whether the relationship between flood-related features (like elevation and slope) and flood risk changes from one part of the city to another, instead of assuming one fixed relationship everywhere.

On the test set, GWR scored an AUC of 0.676, which is considered **weak** for telling flood-prone from non-flood-prone areas.
 Its R² (how much of the variation in flood risk it explains) was 0.156.

### MARS
MARS (Multivariate Adaptive Regression Splines) looks for threshold effects — points where a feature (e.g. slope or TWI) starts to matter a lot more or less to flood risk — rather than assuming a straight-line relationship.

On the test set, MARS scored an AUC of 0.670, which is considered **weak** for telling flood-prone from non-flood-prone areas.

### SVM
SVM (Support Vector Machine) with an RBF kernel draws a flexible boundary between flood-prone and non-flood-prone areas based on all features combined, and tends to be a strong, stable classifier on this kind of tabular data.

On the test set, SVM scored an AUC of 0.679, which is considered **weak** for telling flood-prone from non-flood-prone areas.

### MGWR
MGWR (Multiscale GWR) lets each feature have its own local scale of influence, instead of one shared bandwidth for all features. Here it is fitted with a custom backfitting routine; the intercept varies at a very local scale while the other effects are close to constant across the area.

On the test set, MGWR scored an AUC of 0.747, which is considered **fair** for telling flood-prone from non-flood-prone areas.
 Its R² (how much of the variation in flood risk it explains) was 0.338.

### STACKING
The Stacking Ensemble combines predictions from several base models (Logistic Regression, SVM, Random Forest) through a meta-model, usually giving the most accurate and reliable overall flood susceptibility score.

On the test set, STACKING scored an AUC of 0.752, which is considered **fair** for telling flood-prone from non-flood-prone areas.

## Flood Risk Classification (Test Areas)

Each test location was sorted into one of 4 risk classes with equal thresholds (25% intervals):

- **Low Risk**: 0-25% probability
- **Medium Risk**: 25-50% probability
- **High Risk**: 50-75% probability
- **Very High Risk**: 75-100% probability

Distribution of test areas:

- **Low Risk**: 124 areas (15.5% of the test set)
- **Medium Risk**: 247 areas (30.9% of the test set)
- **High Risk**: 318 areas (39.8% of the test set)
- **Very High Risk**: 111 areas (13.9% of the test set)

## Bottom Line

Based on test-set AUC, **STACKING** performed best (AUC = 0.752) and is the most reliable model here for producing the final flood susceptibility map.