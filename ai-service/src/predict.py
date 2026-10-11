
import cv2
import mediapipe as mp
import joblib
import numpy as np

from collections import deque, Counter


# -----------------------------------
# FILE PATHS
# -----------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(BASE_DIR / "model" / "gesture_model.pkl")
TASK_PATH = str(BASE_DIR / "hand_landmarker.task")

# -----------------------------------
# LANDMARK NORMALIZATION
# Must match train_model.py
# -----------------------------------

def normalize_landmarks(values):
    points = np.asarray(values, dtype=float).reshape(21, 3).copy()

    # Use the wrist as the origin.
    points = points - points[0].copy()

    # Normalize the hand size using wrist-to-middle-finger MCP distance.
    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------

model = joblib.load(MODEL_PATH)

print("Normalized model loaded successfully.")


# -----------------------------------
# MEDIAPIPE SETUP
# -----------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=TASK_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# -----------------------------------
# CAMERA SETUP
# -----------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access the camera.")
    landmarker.close()
    raise SystemExit(1)

print("Camera started.")
print("Show one hand to the camera.")
print("Press Q to quit.")


# Keep the last 10 accepted predictions.
prediction_history = deque(maxlen=10)

# Minimum confidence required to accept a frame.
CONFIDENCE_THRESHOLD = 45.0


# -----------------------------------
# MAIN WEBCAM LOOP
# -----------------------------------

try:
    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read from the camera.")
            break

        # Mirror the webcam view.
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect the hand.
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Collect 63 landmark coordinates.
            raw_features = []

            for landmark in hand:
                raw_features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Apply the same normalization as training.
            features = normalize_landmarks(raw_features)

            # Predict the current frame.
            probabilities = model.predict_proba(
                [features]
            )[0]

            best_index = int(np.argmax(probabilities))

            raw_prediction = str(
                model.classes_[best_index]
            )

            confidence = float(
                probabilities[best_index] * 100
            )

            # Accept only sufficiently confident frames.
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_history.append(raw_prediction)

            # Choose the most frequent recent prediction.
            if prediction_history:
                prediction, votes = Counter(
                    prediction_history
                ).most_common(1)[0]

                vote_text = (
                    f"Recent votes: {votes}/"
                    f"{len(prediction_history)}"
                )
            else:
                prediction = "?"
                vote_text = "Waiting for clear handshape"

            # Draw detected hand landmarks.
            height, width, _ = frame.shape

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Display stabilized prediction.
            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display current frame confidence.
            cv2.putText(
                frame,
                f"Frame confidence: {confidence:.2f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                vote_text,
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Hand detected",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        else:

            # Clear old predictions when the hand disappears.
            prediction_history.clear()

            cv2.putText(
                frame,
                "Show one hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Show the webcam.
        cv2.imshow(
            "GestureType - ASL Prediction",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()

print("Prediction program stopped.")