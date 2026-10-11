
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Get the ai-service folder regardless of where the script is run.
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "bsl_landmarks.csv"
MODEL_PATH = BASE_DIR / "model" / "gesture_model.pkl"
MATRIX_PATH = BASE_DIR / "model" / "confusion_matrix.png"


def normalize_landmarks(values):
    """Normalize hand landmarks by position and scale."""
    points = np.asarray(values, dtype=float).reshape(21, 3).copy()

    # Use the wrist as the origin.
    points = points - points[0].copy()

    # Normalize using the wrist-to-middle-finger MCP distance.
    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


def main():
    # Check paths before starting.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load and validate dataset.
    data = pd.read_csv(DATASET_PATH)

    if "label" not in data.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    if data.isnull().any().any():
        raise ValueError(
            "Dataset contains missing values. Clean it before training."
        )

    X_raw = data.drop(columns=["label"]).to_numpy(dtype=float)
    y = data["label"].astype(str)

    if X_raw.shape[1] != 63:
        raise ValueError(
            f"Expected 63 landmark features, got {X_raw.shape[1]}"
        )

    if y.nunique() < 2:
        raise ValueError("At least two different labels are required.")

    # Every class needs enough samples for a stratified split.
    counts = y.value_counts()

    if counts.min() < 2:
        raise ValueError(
            "Every letter needs at least two samples for training."
        )

    print("Dataset loaded successfully.")
    print("Dataset path:", DATASET_PATH)
    print("Total samples:", len(data))
    print("Features per sample:", X_raw.shape[1])
    print("Number of classes:", y.nunique())
    print("\nSamples per letter:")
    print(counts.sort_index().to_string())

    # Apply the same normalization used by the prediction service.
    X = np.array([
        normalize_landmarks(row)
        for row in X_raw
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining normalized landmark model...")
    model.fit(X_train, y_train)

    # Evaluate before saving the new model.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("NORMALIZED MODEL EVALUATION")
    print("==============================")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy percentage: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    labels = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(xticks_rotation="vertical")
    plt.title("ASL Recognition - Normalized Landmarks")
    plt.tight_layout()
    plt.savefig(MATRIX_PATH, dpi=150)
    plt.close()

    # Save to a temporary file first, then replace the model.
    # The existing model remains in place if training fails earlier.
    temporary_model_path = MODEL_PATH.with_name(
        "gesture_model_new.pkl"
    )

    try:
        joblib.dump(model, temporary_model_path)
        temporary_model_path.replace(MODEL_PATH)
    finally:
        if temporary_model_path.exists():
            temporary_model_path.unlink()

    print("\nModel saved successfully:", MODEL_PATH)
    print("Confusion matrix saved:", MATRIX_PATH)


if __name__ == "__main__":
    main()
