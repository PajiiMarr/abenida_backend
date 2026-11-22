# Flood Susceptibility Model Report - OCT_2022

| Model | Status | Test AUC | Rating | Test Accuracy | Test F1 |
|---|---|---|---|---|---|
| SVM | ok | 0.780 | fair | 0.717 | 0.719 |
| STACKING | ok | 0.916 | excellent | 0.838 | 0.839 |
| MARS | ok | 0.756 | fair | 0.701 | 0.697 |
| GWR | ok | 0.667 | weak | 0.643 | 0.623 |
| MGWR | ok | 0.852 | good | 0.779 | 0.780 |

## Flood Risk Classification (model: **stacking**)

- **Low Risk**: 1302 areas (42.9%)
- **Medium Risk**: 344 areas (11.3%)
- **High Risk**: 360 areas (11.9%)
- **Very High Risk**: 1030 areas (33.9%)