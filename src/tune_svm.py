
from pathlib import Path

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
from sklearn.svm import SVC
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



svm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            SVC(
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


param_grid = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__kernel": ["linear", "rbf"],
    "classifier__gamma": ["scale", "auto"],
}



fail_f1 = make_scorer(
    f1_score,
    pos_label="Fail",
    zero_division=0,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    scoring=fail_f1,
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

best_svm = grid_search.best_estimator_

# Predict on the training data for diagnosis only
train_predictions = best_svm.predict(X_train)

print("\nTraining classification report:")
print(
    classification_report(
        y_train,
        train_predictions,
        labels=["Pass", "Fail"],
        zero_division=0,
    )
)

print("\nTraining confusion matrix (Pass, Fail):")
print(
    confusion_matrix(
        y_train,
        train_predictions,
        labels=["Pass", "Fail"],
    )
)

print("\nTraining prediction counts:")
print(pd.Series(train_predictions).value_counts())

# Export every tested configuration
results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        "param_classifier__C",
        "param_classifier__kernel",
        "param_classifier__gamma",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "rank_test_score",
    ]
].sort_values("rank_test_score")

csv_path = REPORTS_PATH / "svm_tuning.csv"
results.to_csv(csv_path, index=False)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation Fail F1:", round(grid_search.best_score_, 3))
print(f"CSV saved to: {csv_path.resolve()}")
print("CSV file exists:", csv_path.exists())
print("Number of configurations saved:", len(results))
print(results.to_string(index=False))
