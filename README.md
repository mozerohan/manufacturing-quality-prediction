
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
- pandas and matplotlib
- scikit-learn
- joblib
- FastAPI and Uvicorn
- pytest
- Docker
- Git and GitHub

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

| Metric            | Logistic Regression |   SVM | Gaussian Naive Bayes |
| ----------------- | ------------------: | ----: | -------------------: |
| Accuracy          |               0.903 | 0.913 |                0.913 |
| Balanced accuracy |               0.762 | 0.734 |                0.734 |
| Fail precision    |               0.270 | 0.278 |                0.278 |
| Fail recall       |               0.607 | 0.536 |                0.536 |
| Fail F1-score     |               0.370 | 0.366 |                0.366 |

Logistic Regression was selected provisionally because it achieved the highest `Fail` recall and `Fail` F1-score among these three tuned candidates on this split. The difference is small, and the test set was used to compare and select the model; these figures should not be treated as an untouched final estimate of generalization.

Because the data is synthetic, these metrics are illustrative rather than evidence of industrial performance.

## API

The FastAPI application is defined in `src/app.py`.

| Method | Endpoint   | Purpose                                              |
| ------ | ---------- | ---------------------------------------------------- |
| GET    | `/`        | API information                                      |
| GET    | `/health`  | Basic service and model-loaded status                |
| POST   | `/predict` | Predict `Pass` or `Fail` from manufacturing features |

Interactive documentation is available at `/docs` when the service is running.

The prediction response contains the predicted class and model-estimated class probabilities. These probabilities may not be calibrated and should not be interpreted as guarantees.


### Example Prediction Request

Use the following sample input to test the `POST /predict` endpoint in Swagger UI.

**Sample input (JSON):**

```json
{
  "Rated_Power_VA": 500,
  "Input_Voltage_V": 230,
  "Winding_Time_Min": 45,
  "Assembly_Time_Min": 30,
  "Core_Weight_Kg": 2.5,
  "Copper_Weight_Kg": 1.2,
  "Winding_Temperature_C": 65,
  "Product_Type": "Control Transformer",
  "Production_Line": "Line A",
  "Shift": "Day"
}
```

**How to test:**

1. Start the API locally or run the Docker container.
2. Open [http://localhost:8000/docs](http://localhost:8000/docs).
3. Expand `POST /predict`.
4. Click **Try it out**.
5. Paste the JSON example into the request body.
6. Click **Execute** to view the response.

The response includes the predicted inspection class (`Pass` or `Fail`) and the model-estimated probabilities for each class.



## Run Locally

Create and activate a virtual environment, then install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn src.app:app --reload
```

Open the interactive API documentation:

**[http://localhost:8000/docs](http://localhost:8000/docs)**

Use the Swagger interface to submit manufacturing data to `POST /predict` and inspect the prediction response.

## Run Automated Tests

From the project root:

```bash
python -m pytest tests/test_api.py -v
```

The API test suite covers the home endpoint, health endpoint, successful predictions, negative numerical input, and missing required fields.

## Run with Docker

Docker runs the API and its required dependencies in a container. The image includes the saved model, and the scikit-learn version is pinned to match the version used to save it.

### 1. Build the image

Run this command from the project root, where the `Dockerfile` is located:

```bash
docker build -t manufacturing-quality-api:1.0 .
```

### 2. Start the container

```bash
docker run -d --name manufacturing-quality-container -p 8000:8000 manufacturing-quality-api:1.0
```

The `-p 8000:8000` option maps port 8000 on your computer to port 8000 inside the container. This makes the API accessible from your local browser.

### 3. Open the API documentation

**[http://localhost:8000/docs](http://localhost:8000/docs)**

The Swagger interface lets you test the containerized API by submitting input data to `POST /predict`.

You can also check the health endpoint:

**[http://localhost:8000/health](http://localhost:8000/health)**

### 4. Check container status and logs

Check running containers:

```bash
docker ps
```

View application logs:

```bash
docker logs manufacturing-quality-container
```

### 5. Stop, restart, or remove the container

Stop the container:

```bash
docker stop manufacturing-quality-container
```

Start the existing container again:

```bash
docker start manufacturing-quality-container
```

Remove the stopped container when it is no longer needed:

```bash
docker rm manufacturing-quality-container
```

Removing a container does not remove the Docker image.

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
- Consider a dedicated web interface if users need a simpler experience than Swagger.

## Disclaimer

This is an educational portfolio project built with synthetic data. Predictions must not be used as the sole basis for manufacturing safety, inspection, or product-release decisions.
