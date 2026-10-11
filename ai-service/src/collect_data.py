
import cv2
import mediapipe as mp
import csv
from pathlib import Path
import string

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

# Locate the ai-service folder using this script's location.
BASE_DIR = Path(__file__).resolve().parent.parent

TASK_PATH = BASE_DIR / "hand_landmarker.task"
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "bsl_landmarks.csv"

# Create the dataset folder if it does not exist.
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Check that the MediaPipe task file exists.
if not TASK_PATH.exists():
    raise FileNotFoundError(
        f"Hand landmarker model not found: {TASK_PATH}\n"
        "Make sure hand_landmarker.task is inside the ai-service folder."
    )

# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(TASK_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# --------------------------------------------------
# DATASET SETUP
# --------------------------------------------------

# Create the CSV with a header only if it does not exist.
if not DATASET_PATH.exists():
    columns = ["label"]

    for landmark in range(1, 22):
        columns.extend([
            f"x{landmark}",
            f"y{landmark}",
            f"z{landmark}"
        ])

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

# --------------------------------------------------
# SELECT LETTER
# --------------------------------------------------

label = input(
    "Enter the ASL letter to collect (A-Z): "
).strip().upper()

if len(label) != 1 or label not in string.ascii_uppercase:
    print("Invalid input. Enter one letter from A to Z.")
    landmarker.close()
    raise SystemExit(1)

print()
print(f"Collecting ASL samples for letter: {label}")
print("Show ONE hand clearly to the camera.")
print("Keep the selected letter's handshape correct.")
print("Press SPACE to capture a sample.")
print("Press Q to quit.")
print()

# --------------------------------------------------
# OPEN CAMERA
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access the camera. Close other camera apps and try again.")
    landmarker.close()
    raise SystemExit(1)

sample_count = 0
window_name = "GestureType - ASL Data Collection"

# --------------------------------------------------
# MAIN CAMERA LOOP
# --------------------------------------------------

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame from the camera.")
            break

        # Mirror the frame, matching the existing collection process.
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        hand = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            height, width, _ = frame.shape

            for point in hand:
                x = int(point.x * width)
                y = int(point.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            hand_text = "Hand detected - ready to capture"
            hand_color = (0, 255, 0)
        else:
            hand_text = "No hand detected - show one hand"
            hand_color = (0, 0, 255)

        # Display instructions and sample count.
        cv2.putText(
            frame,
            f"ASL Letter: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"New samples captured: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Capture | Q = Quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            hand_text,
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            hand_color,
            2
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        # Capture a sample only when a hand is detected.
        if key == ord(" "):
            if hand is None:
                print("No hand detected. Show your hand and try again.")
                continue

            row = [label]

            for point in hand:
                row.extend([
                    point.x,
                    point.y,
                    point.z
                ])

            if len(row) != 64:
                print("Incorrect feature count. Sample was not saved.")
                continue

            # Append to the existing dataset; do not overwrite it.
            with open(
                DATASET_PATH,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(row)

            sample_count += 1

            print(
                f"Saved sample {sample_count} for {label}"
            )

        if key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()

print()
print(f"Finished collecting {sample_count} new samples for {label}.")
print(f"Dataset location: {DATASET_PATH}")
