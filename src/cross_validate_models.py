from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

# Keep the test set untouched during cross-validation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Training records:", len(X_train))
print("Held-out test records:", len(X_test))
print("\nTraining target distribution:")
print(y_train.value_counts())

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

models = {
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        random_state=42,
    ),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Support Vector Machine": SVC(
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
}

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)



scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "fail_precision": make_scorer(
        precision_score, pos_label="Fail", zero_division=0
    ),
    "fail_recall": make_scorer(
        recall_score, pos_label="Fail", zero_division=0
    ),
    "fail_f1": make_scorer(
        f1_score, pos_label="Fail", zero_division=0
    ),
}

results = []

for name, classifier in models.items():
    print(f"\nCross-validating: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        error_score="raise",
        n_jobs=1,
    )

    results.append({
        "Model": name,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Balanced Accuracy Mean": scores["test_balanced_accuracy"].mean(),
        "Fail Precision Mean": scores["test_fail_precision"].mean(),
        "Fail Recall Mean": scores["test_fail_recall"].mean(),
        "Fail F1 Mean": scores["test_fail_f1"].mean(),
        "Balanced Accuracy Std": scores["test_balanced_accuracy"].std(),
        "Fail Recall Std": scores["test_fail_recall"].std(),
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Balanced Accuracy Mean",
    ascending=False,
)

print("\n===== CROSS-VALIDATION RESULTS =====")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    REPORTS_PATH / "cross_validation_results.csv",
    index=False,
)