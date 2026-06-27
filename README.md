# Air Quality Prediction with Random Forest Regression

Classic machine-learning regression project predicting carbon monoxide concentration from the UCI Air Quality dataset.

## Project Maturity

Notebook analysis project. This is useful evidence of a complete supervised ML workflow: data cleaning, EDA, feature selection, model training, evaluation, and interpretation.

## Results

| Metric | Result |
|---|---:|
| R-squared | 0.92 |
| MAE | 0.08 |
| MSE | 0.01 |
| RMSE | 0.12 |

## Dataset

Source: [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

The dataset contains hourly sensor and meteorological readings from an air quality multisensor device in an Italian city. The target variable is `CO(GT)`, the true hourly averaged carbon monoxide concentration.

## Methodology

1. Loaded and cleaned the dataset, including missing-value handling for sentinel values.
2. Explored sensor distributions, temporal patterns, and feature correlations.
3. Selected features to reduce multicollinearity while preserving predictive signal.
4. Trained a Random Forest regressor.
5. Evaluated predictions using standard regression metrics.

## Main Artifact

- `Air_Quality_Random_Forest_Model_Samarth.ipynb`: notebook with the full analysis and executed outputs.

## Local Setup

```bash
git clone https://github.com/Agent007repo/Air_Quality_Random_Forest_Model_Project.git
cd Air_Quality_Random_Forest_Model_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Air_Quality_Random_Forest_Model_Samarth.ipynb
```

Download the UCI Air Quality dataset and place it where the notebook expects the input file.

## Recruiter Signal

This project supports data science and analytics roles by showing a clean end-to-end ML workflow and clear model evaluation.

## Technical Reviewer Signal

The project is solid as a learning notebook. To make it stronger for ML engineering roles, the model should be packaged into reusable training/inference scripts with saved plots, baseline comparisons, and hyperparameter tuning.

## Known Limitations

- The project is notebook-based rather than packaged as a reusable library or service.
- Additional baseline models would strengthen the comparison.
- A production setting would need data validation, drift monitoring, retraining logic, and an inference interface.

## Recommended Next Improvements

- Add exported visualizations under `outputs/`.
- Add `train.py` and `predict.py` scripts.
- Add baseline model comparison.
- Add hyperparameter tuning.
- Add a short model card with assumptions and intended use.
