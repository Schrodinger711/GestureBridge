
import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # Draw connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame, points[start], points[end],
            (0, 255, 0), 2
        )

    # Draw landmark points
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


# Configure MediaPipe Tasks
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

with vision.HandLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    print("GestureBridge hand tracking started.")
    print("Press Q to quit.")

    start_time = time.monotonic()
    last_timestamp_ms = -1

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp_ms = max(
                timestamp_ms, last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            results = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            if results.hand_landmarks:
                for landmarks in results.hand_landmarks:
                    draw_hand(frame, landmarks)

                message = "Hand detected"
                color = (0, 255, 0)
            else:
                message = "Show your hand"
                color = (0, 0, 255)

            cv2.putText(
                frame, message, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2
            )

            cv2.imshow(
                "GestureBridge - Hand Tracking", frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
