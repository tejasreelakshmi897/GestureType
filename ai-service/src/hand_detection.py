import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the camera")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                height, width, _ = frame.shape

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("GestureType - Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
landmarker.close()