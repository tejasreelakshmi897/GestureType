
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------
# FILE PATHS AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "gesture_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# FASTAPI SETUP
# --------------------------------------------------

app = FastAPI(
    title="GestureType AI Service",
    description="Recognizes ASL letters from hand landmarks",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# REQUEST VALIDATION
# --------------------------------------------------

class GestureRequest(BaseModel):
    landmarks: list[float] = Field(
        ...,
        min_length=63,
        max_length=63,
    )

    @field_validator("landmarks")
    @classmethod
    def validate_landmarks(cls, values):
        if not np.isfinite(values).all():
            raise ValueError("Landmarks must contain only finite numbers.")
        return values

# --------------------------------------------------
# LANDMARK NORMALIZATION
# Must match train_model.py
# --------------------------------------------------

def normalize_landmarks(values):
    points = np.asarray(values, dtype=float).reshape(21, 3).copy()

    # Move the wrist to the origin.
    points = points - points[0].copy()

    # Normalize hand size using landmark 9.
    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        scale = 1.0

    return (points / scale).flatten()

# --------------------------------------------------
# HEALTH ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "GestureType AI Service is running",
        "model_loaded": True,
        "classes": len(model.classes_),
        "available_letters": [str(letter) for letter in model.classes_],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }

# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_gesture(request: GestureRequest):
    try:
        values = np.asarray(request.landmarks, dtype=float)
        features = normalize_landmarks(values)

        # Get probabilities for every letter.
        probabilities = model.predict_proba([features])[0]

        # Highest-probability prediction.
        best_index = int(np.argmax(probabilities))
        prediction = str(model.classes_[best_index])
        confidence = round(float(probabilities[best_index] * 100), 2)

        # Return the three most likely letters for debugging.
        top_indices = np.argsort(probabilities)[::-1][:3]

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
            "confidence": confidence,
            "top_predictions": top_predictions,
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"Prediction error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the AI service terminal.",
        ) from error
