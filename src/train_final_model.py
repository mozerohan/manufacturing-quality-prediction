
from pathlib import Path
import joblib

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "manufacturing_production_data.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "Inspection_Result"

numeric_features = [
    "Rated_Power_VA",
    "Input_Voltage_V",
    "Winding_Time_Min",
    "Assembly_Time_Min",
    "Core_Weight_Kg",
    "Copper_Weight_Kg",
    "Winding_Temperature_C",
]

categorical_features = [
    "Product_Type",
    "Production_Line",
    "Shift",
]

features = numeric_features + categorical_features

df = df.dropna(subset=[target]).copy()

X = df[features]
y = df[target].astype(str).str.strip().str.title()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Training records:", len(X_train))
print("Test records reserved for final evaluation:", len(X_test))

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    sparse_threshold=0,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    )),
])

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["lbfgs"],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=make_scorer(
        f1_score,
        pos_label="Fail",
        zero_division=0,
    ),
    cv=cv,
    refit=True,
    n_jobs=-1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest mean cross-validation Fail F1:")
print(round(grid_search.best_score_, 3))

cv_results = pd.DataFrame(grid_search.cv_results_)

selected_results = cv_results[
    [
        "param_classifier__C",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "rank_test_score",
    ]
].sort_values("rank_test_score")

selected_results.to_csv(
    REPORTS_PATH / "logistic_regression_tuning.csv",
    index=False,
)

print("\nTuning results:")
print(selected_results.round(3).to_string(index=False))


MODEL_PATH = PROJECT_ROOT / "models" / "final_quality_model.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

final_model = grid_search.best_estimator_

joblib.dump(final_model, MODEL_PATH)

print("Selected model: Logistic Regression")
print("Best parameters:", grid_search.best_params_)
print(f"Saved final pipeline to: {MODEL_PATH.resolve()}")
print("Model file exists:", MODEL_PATH.is_file())
