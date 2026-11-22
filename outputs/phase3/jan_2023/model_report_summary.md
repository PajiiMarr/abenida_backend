# Flood Susceptibility Model Report - JAN_2023

| Model | Status | Test AUC | Rating | Test Accuracy | Test F1 |
|---|---|---|---|---|---|
| SVM | ok | 0.785 | fair | 0.722 | 0.724 |
| STACKING | ok | 0.931 | excellent | 0.853 | 0.854 |
| MARS | ok | 0.770 | fair | 0.699 | 0.695 |
| GWR | ok | 0.680 | weak | 0.635 | 0.625 |
| MGWR | ok | 0.844 | good | 0.775 | 0.777 |

## Flood Risk Classification (model: **stacking**)

- **Low Risk**: 1308 areas (44.2%)
- **Medium Risk**: 322 areas (10.9%)
- **High Risk**: 323 areas (10.9%)
- **Very High Risk**: 1004 areas (34.0%)