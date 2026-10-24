
# Manufacturing Quality Prediction

An end-to-end machine learning project that predicts manufacturing inspection outcomes using production-process data. The project covers data inspection, preprocessing, model comparison, cross-validation, hyperparameter tuning, model persistence, a REST API, automated testing, and Docker deployment.

> **Important:** This project uses synthetic manufacturing data for educational purposes. Its performance does not establish real-world factory reliability or suitability for production quality-control decisions.

## Project Objectives

- Explore and prepare manufacturing-process data.
- Compare classification algorithms for predicting inspection outcomes.
- Evaluate performance with particular attention to the minority `Fail` class.
- Tune model hyperparameters using stratified cross-validation.
- Save a trained preprocessing and prediction pipeline.
- Serve predictions through FastAPI.
- Test API behavior with pytest.
- Package and run the API in Docker.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- joblib
- FastAPI and Uvicorn
- pytest
- Docker

## Dataset

The dataset contains 3,000 synthetic manufacturing records. The target is `Inspection_Result`, with the classes `Pass` and `Fail`.

### Features used

**Numerical features**
- `Rated_Power_VA`
- `Input_Voltage_V`
- `Winding_Time_Min`
- `Assembly_Time_Min`
- `Core_Weight_Kg`
- `Copper_Weight_Kg`
- `Winding_Temperature_C`

**Categorical features**
- `Product_Type`
- `Production_Line`
- `Shift`

Columns such as `Defect_Type`, `Voltage_Test_Result`, and `Rework_Required` were excluded from model inputs because they may reveal information recorded during or after inspection. This helps avoid target leakage.

## Project Workflow

1. Inspect the dataset and target distribution.
2. Preprocess numerical and categorical features.
3. Compare baseline classification models.
4. Evaluate models using stratified five-fold cross-validation.
5. Tune Logistic Regression, SVM, and Gaussian Naive Bayes.
6. Compare tuned models on a held-out test split.
7. Save the selected Logistic Regression pipeline.
8. Expose predictions through a FastAPI service.
9. Test endpoints and input validation with pytest.
10. Build and run the API in Docker.

## Model Evaluation

The dataset is imbalanced, so accuracy alone is insufficient. The project also considers balanced accuracy, precision, recall, and F1-score for the `Fail` class.

The tuned-model comparison on the 20% test split produced the following results:

| Metric | Logistic Regression | SVM | Gaussian Naive Bayes |
|---|---:|---:|---:|
| Accuracy | 0.903 | 0.913 | 0.913 |
| Balanced accuracy | 0.762 | 0.734 | 0.734 |
| Fail precision | 0.27 | 0.278 | 0.278 |
| Fail recall | 0.607 | 0.536 | 0.536 |
| Fail F1-score | 0.37 | 0.366 | 0.366 |

Logistic Regression was selected provisionally because it achieved the highest `Fail` recall and `Fail` F1-score among these three tuned candidates on this split. The difference is small, and the test set was used to compare and select the model; these figures should not be treated as an untouched final estimate of generalization.

Because the data is synthetic, these metrics are illustrative rather than evidence of industrial performance.

## API

The FastAPI application is defined in `src/app.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Basic service and model-loaded status |
| POST | `/predict` | Predict `Pass` or `Fail` from manufacturing features |

Interactive documentation is available at `/docs` when the service is running.

The prediction response contains the predicted class and model-estimated class probabilities. These probabilities may not be calibrated and should not be interpreted as guarantees.

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the API from the project root:

```bash
python -m uvicorn src.app:app --reload
```

Open `http://127.0.0.1:8000/docs` to submit a prediction request.

## Run Automated Tests

From the project root:

```bash
python -m pytest -v
```

The API test suite covers the home endpoint, health endpoint, successful predictions, negative numerical input, and missing required fields.

## Run with Docker

Build the image:

```bash
docker build -t manufacturing-quality-api:1.0 .
```

Start the container:

```bash
docker run -d --name manufacturing-quality-container -p 8000:8000 manufacturing-quality-api:1.0
```

Open `http://127.0.0.1:8000/docs` to test the containerized API.

To stop and remove the container:

```bash
docker stop manufacturing-quality-container
docker rm manufacturing-quality-container
```

The trained model is included in the image, and the scikit-learn version is pinned to match the version used to save the model.

## Repository Structure

```text
.
├── data/
├── models/
├── reports/
├── src/
│   ├── app.py
│   ├── compare_models.py
│   ├── compare_tuned_models.py
│   ├── cross_validate_models.py
│   ├── inspect_data.py
│   ├── train_final_model.py
│   ├── train_model.py
│   ├── tune_logistic_regression.py
│   ├── tune_naive_bayes.py
│   └── tune_svm.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Limitations and Future Improvements

- Validate the workflow on real, representative manufacturing data before considering operational use.
- Evaluate the model on a genuinely independent dataset.
- Improve minority-class precision and recall, and assess the cost of false negatives.
- Investigate probability calibration and appropriate decision thresholds.
- Add production-grade logging, monitoring, and deployment configuration.

## Disclaimer

This is an educational portfolio project built with synthetic data. Predictions must not be used as the sole basis for manufacturing safety, inspection, or release decisions.