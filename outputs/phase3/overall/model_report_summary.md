# Flood Susceptibility Model Report - OVERALL

| Model | Status | Test AUC | Rating | Test Accuracy | Test F1 |
|---|---|---|---|---|---|
| SVM | ok | 0.802 | good | 0.737 | 0.739 |
| STACKING | ok | 0.931 | excellent | 0.857 | 0.859 |
| MARS | ok | 0.725 | fair | 0.668 | 0.666 |
| GWR | ok | 0.671 | weak | 0.632 | 0.603 |
| MGWR | ok | 0.809 | good | 0.737 | 0.738 |

## Flood Risk Classification (model: **stacking**)

- **Low Risk**: 4214 areas (48.2%)
- **Medium Risk**: 749 areas (8.6%)
- **High Risk**: 822 areas (9.4%)
- **Very High Risk**: 2958 areas (33.8%)