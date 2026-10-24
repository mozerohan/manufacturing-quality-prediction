from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "manufacturing_production_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"
REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Target : Final Inspection outcome
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
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("Balanced Accuracy:")
print(round(balanced_accuracy_score(y_test, predictions),4))

print("\n Classification report:")
report = classification_report(
    y_test,
    predictions,
    labels=["Fail", "Pass"],
    output_dict=True,
    zero_division=0
)
print(report)
metrics = pd.DataFrame(report).loc[
    ["precision", "recall", "f1-score"], ["Fail", "Pass"]
]

metrics.plot(kind="bar", figsize=(8, 5))

plt.title("Model Performance by Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.tight_layout()
plt.savefig(REPORTS_PATH / "class_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n Confusion matrix (rows = actual, colums = predicted):")
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["Fail", "Pass"])
print(cm)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS_PATH / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)


