
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "manufacturing_production_data.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic overview
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- NUMERICAL SUMMARY ---")
print(df.describe().T)

print("\n--- CATEGORICAL SUMMARY ---")
print(df.describe(include="object").T)



print("\n--- DEFECT TYPE COUNTS ---")
print(df["Defect_Type"].value_counts(dropna=False))

print("\n--- INSPECTION RESULT COUNTS ---")
print(df["Inspection_Result"].value_counts(dropna=False))

print("\n--- VOLTAGE TEST RESULT COUNTS ---")
print(df["Voltage_Test_Result"].value_counts(dropna=False))

print("\n--- REWORK REQUIRED COUNTS ---")
print(df["Rework_Required"].value_counts(dropna=False))

print("\n--- DEFECT TYPE BY INSPECTION RESULT ---")
print(pd.crosstab(
    df["Defect_Type"].fillna("Missing"),
    df["Inspection_Result"],
    margins=True
))

print("\n--- INSPECTION RESULT DISTRIBUTION ---")
print(df["Inspection_Result"].value_counts(dropna=False))

print("\n--- INSPECTION RESULT PERCENTAGES ---")
print(
    df["Inspection_Result"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

print("\n--- ALL COLUMN NAMES ---")
for column in df.columns:
    print(column)
