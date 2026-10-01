# Flood Susceptibility Model Report

This report summarizes how each model performed at predicting flood susceptibility, in plain language. Technical terms are kept so the results can be cross-checked against the raw metrics.

## Model Comparison

| Model | Test AUC | Rating | Test Accuracy | Test F1 |
|---|---|---|---|---|
| GWR | 0.525 | poor (close to random guessing) | 0.516 | 0.510 |
| MARS | 0.809 | good | 0.742 | 0.736 |
| SVM | 0.770 | fair | 0.677 | 0.674 |
| STACKING | 0.810 | good | 0.741 | 0.740 |

*AUC (Area Under the ROC Curve) measures how well a model tells flood-prone areas apart from non-flood-prone areas. 0.5 is random guessing, 1.0 is a perfect score.*

## What Each Model Does

### GWR
GWR (Geographically Weighted Regression) checks whether the relationship between flood-related features (like elevation and distance to river) and flood risk changes from one part of the city to another, instead of assuming one fixed relationship everywhere.

On the test set, GWR scored an AUC of 0.525, which is considered **poor (close to random guessing)** for telling flood-prone from non-flood-prone areas.
 Its R² (how much of the variation in flood risk it explains) was 0.281.

### MARS
MARS (Multivariate Adaptive Regression Splines) looks for threshold effects — points where a feature (e.g. slope or drainage density) starts to matter a lot more or less to flood risk — rather than assuming a straight-line relationship.

On the test set, MARS scored an AUC of 0.809, which is considered **good** for telling flood-prone from non-flood-prone areas.

### SVM
SVM (Support Vector Machine) with an RBF kernel draws a flexible boundary between flood-prone and non-flood-prone areas based on all features combined, and tends to be a strong, stable classifier on this kind of tabular data.

On the test set, SVM scored an AUC of 0.770, which is considered **fair** for telling flood-prone from non-flood-prone areas.

### STACKING
The Stacking Ensemble combines predictions from several base models (Logistic Regression, SVM, Random Forest) through a meta-model, usually giving the most accurate and reliable overall flood susceptibility score.

On the test set, STACKING scored an AUC of 0.810, which is considered **good** for telling flood-prone from non-flood-prone areas.

## Flood Risk Classification (Test Areas)

Each test location was sorted into one of 4 risk classes with equal thresholds (25% intervals):

- **Low Risk**: 0-25% probability
- **Medium Risk**: 25-50% probability
- **High Risk**: 50-75% probability
- **Very High Risk**: 75-100% probability

Distribution of test areas:

- **Low Risk**: 203 areas (27.7% of the test set)
- **Medium Risk**: 107 areas (14.6% of the test set)
- **High Risk**: 208 areas (28.4% of the test set)
- **Very High Risk**: 215 areas (29.3% of the test set)

## Bottom Line

Based on test-set AUC, **STACKING** performed best (AUC = 0.810) and is the most reliable model here for producing the final flood susceptibility map.