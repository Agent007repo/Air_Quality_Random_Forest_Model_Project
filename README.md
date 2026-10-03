# Air Quality Prediction with Random Forest Regression

Predict hourly carbon monoxide concentration, `CO(GT)`, from the UCI Air Quality sensor and meteorological dataset using a Random Forest. The notebook includes exploratory plots, missing-value handling, feature selection, and residual analysis.

## Run

Use Python 3.11+ and install `requirements.txt`. Download the semicolon-delimited UCI `AirQualityUCI.csv` and put it in the repository root, or set `AIR_QUALITY_CSV` to its path. Run `Air_Quality_Random_Forest_Model_Samarth.ipynb` from this directory.

```bash
pip install -r requirements.txt
jupyter notebook Air_Quality_Random_Forest_Model_Samarth.ipynb
```

Readings of `-200` are missing values. After sorting by timestamp, the final 20% of observations are held out chronologically. Median imputation and correlation-based feature removal are learned from training data only. The forest learns a log-transformed target, while reported MAE, MSE, RMSE, and R² use the original concentration scale.

## Evaluation status

Earlier README values (R² 0.92 and MAE 0.08) are withdrawn: they came from a random split and transformed-target evaluation and cannot describe the corrected chronological workflow. Notebook outputs are cleared pending a complete dataset rerun. Contemporaneous sensor inputs make this a concentration-estimation experiment; it does not establish future-hour forecasting or deployment readiness.

```bash
python -m unittest discover -s tests -p test_regressions.py -v
```

The regression test executes the notebook's data-cleaning and split/imputation cells against a synthetic CSV with future extremes. It verifies chronological separation and training-only imputation; it does not measure real-data model performance.
