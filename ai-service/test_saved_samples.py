import pandas as pd
import joblib
from src.train_model import normalize_landmarks

data = pd.read_csv("dataset/bsl_landmarks.csv")
model = joblib.load("model/gesture_model.pkl")

feature_columns = [c for c in data.columns if c != "label"]

for letter in ["U", "V"]:
    samples = data[data["label"] == letter]
    correct = 0

    print(f"\nTesting actual letter {letter}:")

    for _, sample in samples.iterrows():
        features = normalize_landmarks(
            sample[feature_columns].to_numpy()
        ).reshape(1, -1)

        prediction = model.predict(features)[0]

        if prediction == letter:
            correct += 1

    print(f"Correct: {correct}/{len(samples)}")
    print(f"Accuracy: {correct / len(samples) * 100:.2f}%")