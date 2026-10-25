
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_FILE = Path("gesture_data.csv")
MODEL_FILE = Path("gesture_model.pkl")

# 1. Load the collected dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        "gesture_data.csv not found. Run collect_data.py first."
    )

df = pd.read_csv(DATA_FILE)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

df = df.dropna()

X = df.drop(columns=["label"])
y = df["label"]

print("\n--- Dataset overview ---")
print("Total valid samples:", len(df))
print("Number of features:", X.shape[1])
print("\nSamples per gesture:")
print(y.value_counts().sort_index())

if y.nunique() < 2:
    raise ValueError("At least two gesture classes are required.")

if y.value_counts().min() < 2:
    raise ValueError(
        "Each gesture needs at least two samples for a split."
    )

# 2. Split into training and test sets
# Stratification keeps class proportions similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Train a baseline classifier
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining the model...")
model.fit(X_train, y_train)

# 4. Evaluate on held-out samples
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n--- Evaluation results ---")
print(f"Test accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

labels = sorted(y.unique())

print("Confusion matrix")
print("Rows = actual gesture; columns = predicted gesture")
print("Gesture order:", labels)
print(confusion_matrix(y_test, predictions, labels=labels))

# 5. Save the trained model and feature order
joblib.dump(
    {
        "model": model,
        "feature_columns": list(X.columns),
        "labels": labels,
    },
    MODEL_FILE,
)

print(f"\nSaved trained model to: {MODEL_FILE.resolve()}")
