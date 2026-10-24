
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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    make_scorer,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


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

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fail_f1 = make_scorer(
    f1_score,
    pos_label="Fail",
    zero_division=0,
)

model_grids = {
    "Logistic Regression": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            )),
        ]),
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs"],
        },
    ),

    "SVM": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"],
        },
    ),

    "Gaussian Naive Bayes": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB()),
        ]),
        {
            "classifier__var_smoothing": [
                1e-12, 1e-11, 1e-10, 1e-9,
                1e-8, 1e-7, 1e-6,
            ],
        },
    ),
}

results = []
confusion_matrices = {}

for model_name, (pipeline, param_grid) in model_grids.items():
    print(f"\n{'=' * 60}")
    print(f"Tuning and evaluating: {model_name}")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=fail_f1,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    test_predictions = best_model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=["Pass", "Fail"],
    )
    confusion_matrices[model_name] = matrix

    results.append({
        "Model": model_name,
        "Best Parameters": str(grid_search.best_params_),
        "CV Fail F1": grid_search.best_score_,
        "Test Accuracy": accuracy_score(y_test, test_predictions),
        "Test Balanced Accuracy": balanced_accuracy_score(
            y_test, test_predictions
        ),
        "Test Fail Precision": precision_score(
            y_test, test_predictions,
            pos_label="Fail", zero_division=0,
        ),
        "Test Fail Recall": recall_score(
            y_test, test_predictions,
            pos_label="Fail", zero_division=0,
        ),
        "Test Fail F1": f1_score(
            y_test, test_predictions,
            pos_label="Fail", zero_division=0,
        ),
    })

    print("Best parameters:", grid_search.best_params_)
    print("CV Fail F1:", round(grid_search.best_score_, 3))
    print("\nTest classification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=["Pass", "Fail"],
        zero_division=0,
    ))
    print("Test confusion matrix (Pass, Fail):")
    print(matrix)

comparison = pd.DataFrame(results).sort_values(
    "Test Fail F1",
    ascending=False,
)

csv_path = REPORTS_PATH / "tuned_model_comparison.csv"
comparison.to_csv(csv_path, index=False)

print("\nFINAL TUNED MODEL COMPARISON")
print(comparison.round(3).to_string(index=False))
print(f"\nComparison CSV saved to: {csv_path.resolve()}")

for model_name, matrix in confusion_matrices.items():
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Pass", "Actual Fail"],
        columns=["Predicted Pass", "Predicted Fail"],
    )
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_test_confusion_matrix.csv"
    )
    matrix_df.to_csv(REPORTS_PATH / filename)
