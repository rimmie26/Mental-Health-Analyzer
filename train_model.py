import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# 1. FILE SETTINGS
# -----------------------------
file_path = "PR project dataset.xlsx"
sheet_name = None

# -----------------------------
# 2. CHECK FILE EXISTS
# -----------------------------
if not os.path.exists(file_path):
    print(f"Error: File not found -> {file_path}")
    raise FileNotFoundError(file_path)

# -----------------------------
# 3. LOAD DATA
# -----------------------------
try:
    excel_file = pd.ExcelFile(file_path, engine="openpyxl")
    print("Available sheet names:", excel_file.sheet_names)

    if sheet_name is None:
        sheet_name = excel_file.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

except ImportError:
    print("openpyxl is not installed. Run:")
    print("pip install openpyxl")
    raise

except Exception as e:
    print("Error while reading Excel file:", str(e))
    raise

print("\nDataset loaded successfully.")
print("Using sheet:", sheet_name)
print("Shape:", df.shape)

# -----------------------------
# 4. CLEAN COLUMN NAMES
# -----------------------------
df.columns = df.columns.str.strip()

print("\nColumns in dataset:")
for col in df.columns:
    print("-", col)

# -----------------------------
# 5. REQUIRED COLUMNS
# -----------------------------
selected_features = [
    "Gender",
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness"
]

target_column = "Depression"

missing_columns = [col for col in selected_features + [target_column] if col not in df.columns]
if missing_columns:
    print("\nMissing required columns:")
    for col in missing_columns:
        print("-", col)
    raise ValueError("Dataset columns do not match expected columns.")

df = df[selected_features + [target_column]].copy()

# -----------------------------
# 6. BASIC CLEANING
# -----------------------------
# Replace invalid markers first
df.replace(["?", " ?", "? ", "", "nan", "NaN", "None", "null", "NULL"], np.nan, inplace=True)

# Strip spaces in object columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# -----------------------------
# 7. FORCE NUMERIC COLUMNS
# -----------------------------
numeric_features = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Financial Stress"
]

categorical_features = [
    "Gender",
    "Sleep Duration",
    "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

# Drop rows where target is missing
df = df.dropna(subset=[target_column])

# Target as integer
df[target_column] = df[target_column].astype(int)

print("\nData types after cleaning:")
print(df.dtypes)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df[target_column].value_counts())

# -----------------------------
# 8. SPLIT FEATURES AND TARGET
# -----------------------------
X = df[selected_features]
y = df[target_column]

# -----------------------------
# 9. PREPROCESSING
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------
# 10. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# 11. MODELS
# -----------------------------
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# 12. EVALUATION FUNCTION
# -----------------------------
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------------
# 13. TRAIN MODELS
# -----------------------------
print("\nTraining Logistic Regression...")
log_reg_pipeline.fit(X_train, y_train)

print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)

# -----------------------------
# 14. EVALUATE MODELS
# -----------------------------
log_results = evaluate_model(log_reg_pipeline, X_test, y_test, "Logistic Regression")
rf_results = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")

# -----------------------------
# 15. SELECT FINAL MODEL
# -----------------------------
if log_results["recall"] >= rf_results["recall"]:
    final_model = log_reg_pipeline
    final_results = log_results
else:
    final_model = rf_pipeline
    final_results = rf_results

print("\nFinal selected model:", final_results["model_name"])
print("Chosen because higher recall is preferred for screening.")

# -----------------------------
# 16. SAVE FILES
# -----------------------------
joblib.dump(final_model, "mental_health_risk_model.pkl")
joblib.dump(selected_features, "selected_features.pkl")

risk_thresholds = {
    "low_max": 0.35,
    "moderate_max": 0.65
}
joblib.dump(risk_thresholds, "risk_thresholds.pkl")

print("\nSaved successfully:")
print("- mental_health_risk_model.pkl")
print("- selected_features.pkl")
print("- risk_thresholds.pkl")

# -----------------------------
# 17. SAMPLE PREDICTIONS
# -----------------------------
sample_probs = final_model.predict_proba(X_test.iloc[:10])[:, 1]

print("\nSample prediction probabilities:")
for i, prob in enumerate(sample_probs, start=1):
    if prob < 0.35:
        level = "Low"
    elif prob < 0.65:
        level = "Moderate"
    else:
        level = "High"

    print(f"Sample {i}: probability={prob:.4f}, mapped_level={level}")

print("\nTraining completed successfully.")