from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    ConfusionMatrixDisplay,
    f1_score,
)

from sklearn.model_selection import train_test_split
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

# Target - Final Inspection outcome
target = "Inspection_Result"

numeric_features = [
    "Rated_Power_VA",
    "Input_Voltage_V",
    "Winding_Time_Min",
    "Assembly_Time_Min",
    "Core_Weight_Kg",
    "Copper_Weight_Kg",
    "Winding_Temperature_C"
]

categorical_features = [
    "Product_Type",
    "Production_Line",
    "Shift"
]

features = numeric_features + categorical_features

# Validate Column names before training
required_columns = features + [target]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Remove rows without a known target
df= df.dropna(subset=[target]).copy()

X = df[features]
y = df[target].astype(str).str.strip().str.title()

print("Target distribution:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,  # Preserve class proportion in both sets
    random_state=42
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
],  sparse_threshold=0)

models = {
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        )),
    ]),

    "K-Nearest Neighbors": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),

    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )),
    ]),

    "Neural Network": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42
        )),
    ]),

    "Naive Bayes": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB()),
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ]),

    "Support Vector Machine": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        )),
    ]),
}

print("Seven model pipelines created successfully.")


results = []

for name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Detailed classification report
    print("\nClassification Report:")
    report=classification_report(
        y_test,
        y_pred,
        labels=["Fail", "Pass"],
        output_dict=True,
        zero_division=0
    )

    print(report)

    metrics = pd.DataFrame(report).loc[
        ["precision", "recall", "f1-score"], ["Fail", "Pass"]
    ]

    metrics.plot(kind="bar", figsize=(8, 5))

    plt.title(f"Model Performance by Class {name}")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        REPORTS_PATH / f"{name.lower().replace(' ', '_')}_class_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Fail", "Pass"]
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(
        REPORTS_PATH / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


    # Overall evaluation metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, y_pred
        ),
        "Fail Recall": recall_score(
            y_test,
            y_pred,
            pos_label="Fail",
            zero_division=0
        ),
        "Fail F1": f1_score(
            y_test,
            y_pred,
            pos_label="Fail",
            zero_division=0
        ),
    })

# Compare models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Balanced Accuracy",
    ascending=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    REPORTS_PATH / "model_comparison.csv",
    index=False
)