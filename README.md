# Air Quality Prediction with Random Forest Regression

This project predicts hourly carbon monoxide concentration (`CO(GT)`) from air-quality sensor and meteorological data. It is a supervised machine-learning workflow covering data cleaning, exploratory analysis, feature selection, model training, and regression evaluation.

## What This Shows

- End-to-end tabular ML workflow
- Sensor-data cleaning and missing-value handling
- Exploratory analysis and correlation-based feature selection
- Random Forest regression with reproducible train/test evaluation
- Model interpretation through error metrics and residual inspection

## Result

The Random Forest model achieved strong test-set performance:

| Metric | Value |
|---|---:|
| R2 | 0.92 |
| MAE | 0.08 |
| MSE | 0.01 |
| RMSE | 0.12 |

## Dataset

Source: [UCI Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

The dataset contains 9,358 hourly observations from an air-quality multisensor device in an Italian city. Missing values are represented as `-200` and are handled during preprocessing.

## Methodology

1. Load and clean the UCI air-quality dataset.
2. Convert date/time fields and replace invalid readings.
3. Explore distributions, correlations, outliers, and temporal patterns.
4. Select predictive sensor and time features while reducing redundant inputs.
5. Train a `RandomForestRegressor` with a fixed random seed.
6. Evaluate with MAE, MSE, RMSE, R2, actual-vs-predicted plots, and residual checks.

## Repository Contents

| File | Purpose |
|---|---|
| `Air_Quality_Random_Forest_Model_Samarth.ipynb` | Main analysis and model notebook |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignore rules for local artifacts |

## How To Run

```bash
git clone https://github.com/Agent007repo/Air_Quality_Random_Forest_Model_Project.git
cd Air_Quality_Random_Forest_Model_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Air_Quality_Random_Forest_Model_Samarth.ipynb
```

Download the UCI dataset and follow the notebook preprocessing steps before running the model cells.

## Recruiter Signal

This is a practical analyst/ML workflow project. It demonstrates data cleaning, supervised modeling, metric interpretation, and clear communication of predictive performance.
