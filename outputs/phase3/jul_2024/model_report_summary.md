# Flood Susceptibility Model Report - JUL_2024

| Model | Status | Test AUC | Rating | Test Accuracy | Test F1 |
|---|---|---|---|---|---|
| SVM | ok | 0.838 | good | 0.774 | 0.777 |
| STACKING | ok | 0.943 | excellent | 0.880 | 0.882 |
| MARS | ok | 0.809 | good | 0.770 | 0.770 |
| GWR | ok | 0.753 | fair | 0.691 | 0.673 |
| MGWR | ok | 0.868 | good | 0.806 | 0.808 |

## Flood Risk Classification (model: **stacking**)

- **Low Risk**: 1484 areas (53.9%)
- **Medium Risk**: 173 areas (6.3%)
- **High Risk**: 192 areas (7.0%)
- **Very High Risk**: 902 areas (32.8%)