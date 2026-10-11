
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------------
# FILE PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "gesture_model.pkl"


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

print("GestureType model loaded successfully.")


# -----------------------------------
# CREATE FASTAPI APP
# -----------------------------------

app = FastAPI(title="GestureType ASL Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gesturetype.netlify.app",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# -----------------------------------
# REQUEST FORMAT
# -----------------------------------

class PredictionRequest(BaseModel):
    landmarks: list[float] = Field(
        ...,
        min_length=63,
        max_length=63,
    )


# -----------------------------------
# NORMALIZE LANDMARKS
# Must match train_model.py
# -----------------------------------

def normalize_landmarks(values):
    points = np.asarray(values, dtype=float).reshape(21, 3).copy()

    # Use the wrist as the origin.
    points = points - points[0].copy()

    # Normalize using wrist-to-middle-finger MCP distance.
    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


# -----------------------------------
# HEALTH CHECK ENDPOINTS
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "GestureType ASL Recognition API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------------
# PREDICTION ENDPOINT
# -----------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        raw_landmarks = np.asarray(
            request.landmarks,
            dtype=float,
        )

        if not np.isfinite(raw_landmarks).all():
            raise HTTPException(
                status_code=400,
                detail="Landmarks must contain finite numbers.",
            )

        features = normalize_landmarks(raw_landmarks)

        probabilities = model.predict_proba([features])[0]

        top_indices = np.argsort(probabilities)[::-1][:3]

        best_index = int(top_indices[0])
        prediction = str(model.classes_[best_index])
        confidence = float(probabilities[best_index] * 100)

        top_predictions = [
            {
                "letter": str(model.classes_[int(index)]),
                "confidence": round(
                    float(probabilities[index] * 100), 2
                ),
            }
            for index in top_indices
        ]

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )
