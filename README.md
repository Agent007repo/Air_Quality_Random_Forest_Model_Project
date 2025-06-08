# Air Quality Prediction using Random Forest Regression

## Project Overview

This project focuses on predicting Carbon Monoxide (CO) concentration levels in the air, specifically the `CO(GT)` target variable, using a Random Forest Regressor model. The dataset comprises hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Multisensor Device, along with meteorological data. The goal was to perform comprehensive Exploratory Data Analysis (EDA), effective feature selection, and build a robust predictive model, evaluating its performance using standard regression metrics.

This project demonstrates a typical machine learning workflow, from data understanding and preparation to model training and evaluation, culminating in a high-performance predictive model.

**Key Achievements:**
- Developed a Random Forest model with an **R-squared value of 0.92** on the test set.
- Achieved low prediction errors: **MAE: 0.08, MSE: 0.01, RMSE: 0.12**.
- Performed thorough EDA and feature selection to handle multicollinearity and identify relevant predictors.

## Dataset

The project utilizes the [Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality) from the UCI Machine Learning Repository. This dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Multisensor Device. The device was located on the field in a significantly polluted area, at road level, within an Italian city.

**Target Variable:** `CO(GT)` - True hourly averaged concentration CO in mg/m^3 (reference analyzer)

## Methodology

The project followed these key steps:

1.  **Data Loading and Initial Cleaning:** Loaded the dataset, handled missing values (often represented as -200 in this dataset), and performed initial data type conversions.
2.  **Exploratory Data Analysis (EDA):**
    *   Analyzed data distributions for key features and the target variable.
    *   Investigated correlations between different sensor readings and meteorological data using heatmaps.
    *   Visualized temporal patterns (e.g., CO levels by hour, month) to understand trends.
    *   Used box plots for outlier detection.
3.  **Feature Selection:**
    *   Based on EDA (particularly correlation analysis), features were selected to reduce multicollinearity and retain predictive power.
    *   The final selected features for modeling were: `PT08.S1(CO)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `Hour`, `Month`.
4.  **Data Splitting:** The dataset was split into training (6139 samples) and testing (1535 samples) sets.
5.  **Model Training:**
    *   A `RandomForestRegressor` from `scikit-learn` was chosen for its robustness and ability to capture non-linear relationships.
    *   The model was initialized with `n_estimators=100` and `random_state=42` for reproducibility.
    *   The model was trained on the selected features from the training set.
6.  **Prediction and Evaluation:**
    *   Predictions were made on the test set.
    *   Model performance was evaluated using:
        *   Mean Absolute Error (MAE)
        *   Mean Squared Error (MSE)
        *   Root Mean Squared Error (RMSE)
        *   R-squared (R2) score

## Results and Key Findings

The Random Forest Regressor demonstrated strong predictive performance on the test set:

*   **R-squared (R2): 0.92** (The model explains 92% of the variance in `CO(GT)` levels)
*   **Mean Absolute Error (MAE): 0.08**
*   **Mean Squared Error (MSE): 0.01**
*   **Root Mean Squared Error (RMSE): 0.12**

These metrics indicate that the model's predictions are highly accurate and closely align with the actual air quality measurements.

### Visualizing Insights

*(This section is where you'll embed or describe your key visualizations. Ensure your Jupyter Notebook, when placed in this repository, has these plots with clear outputs. Alternatively, save them as images, place them in a `visualizations/` subfolder, and embed them here using Markdown: `![Description](visualizations/plot_name.png)`)*

*   **Correlation Heatmap:**
    *   **Discussion:** "The correlation heatmap was instrumental in identifying highly correlated features. For instance, strong correlations were observed between [mention example pair like 'C6H6(GT)' and 'PT08.S2(NMHC)']. This guided the feature selection process to remove redundant features, improving model efficiency and interpretability."
    *   `![Correlation Heatmap](visualizations/correlation_heatmap.png)` (Replace with actual path or describe where to find it in the notebook)

*   **Actual vs. Predicted Plot:**
    *   **Discussion:** "This plot visually confirms the model's high accuracy. The data points cluster tightly around the diagonal line, indicating that the predicted `CO(GT)` values are very close to the actual values. This provides strong visual support for the R2 score of 0.92."
    *   `![Actual vs Predicted Plot](visualizations/actual_vs_predicted.png)` (Replace with actual path or describe where to find it in the notebook)

*   **Residual Plot:**
    *   **Discussion:** "The residual plot (residuals vs. predicted values) shows a random scatter of points around the zero line. This lack of a discernible pattern suggests that the model's errors are homoscedastic and not systematically biased, which is a good indication of model fit."
    *   `![Residual Plot](visualizations/residual_plot.png)` (Replace with actual path or describe where to find it in the notebook)

*   **Temporal Patterns (e.g., CO(GT) vs. Hour):**
    *   **Discussion:** "Visualizing `CO(GT)` levels against the 'Hour' of the day revealed [describe pattern, e.g., 'clear diurnal patterns, with peaks typically observed during morning and evening rush hours']. This underscored the importance of including 'Hour' as a predictive feature."
    *   `![Temporal Plot - CO vs Hour](visualizations/co_vs_hour_plot.png)` (Replace with actual path or describe where to find it in the notebook)

## File Structure

-   `Air_Quality_Random_Forest_Model_Samarth.ipynb`: The main Jupyter Notebook containing all the code for data analysis, model training, and evaluation.
-   `README.md`: This file, providing an overview of the project.
-   `requirements.txt`: A list of Python dependencies required to run the project.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `data/` (Recommended): Create this folder and place your dataset (e.g., `AirQualityUCI.csv`) here. Provide download instructions if not including the file directly.
-   `visualizations/` (Recommended): Create this folder and save your key plots as images here if you want to embed them directly in the README.

## How to Run / Reproduce

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Obtain the dataset:** Download the Air Quality dataset from the UCI ML Repository ([link](https://archive.ics.uci.edu/ml/datasets/Air+Quality)). You will need to place the `AirQualityUCI.xlsx` (or `.csv` if you convert it) file into a `data/` subdirectory within the project. The notebook expects the data to be in `data/AirQualityUCI.xlsx`.
    *   Ensure you handle the preprocessing steps as outlined in the notebook (e.g., converting -200 to NaN, parsing dates/times).
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Air_Quality_Random_Forest_Model_Samarth.ipynb
    ```
    Execute the cells sequentially to see the analysis and model results.

## Future Work

Potential areas for future improvement include:

*   **Hyperparameter Tuning:** Employ techniques like GridSearchCV or RandomizedSearchCV to find the optimal hyperparameters for the Random Forest model.
*   **Advanced Feature Engineering:** Explore creating more complex features, such as interaction terms or lag features if appropriate for time series aspects.
*   **Comparison with Other Models:** Evaluate other regression algorithms (e.g., Gradient Boosting, SVMs, Neural Networks) to compare performance.
*   **Deployment:** Package the model into an API for real-time predictions (e.g., using Flask/FastAPI).

## License

This project is open-sourced under the MIT License. (Consider adding a `LICENSE` file - a common choice is the MIT License. You can create a file named `LICENSE` and paste the MIT License text into it.)
