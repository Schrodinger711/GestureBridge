
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
DATA_FILE = Path("gesture_data.csv")

GESTURES = ["HELLO", "YES", "NO", "OK", "PEACE"]
SAMPLES_PER_GESTURE = 200

# MediaPipe hand landmark detector
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Create CSV with 63 landmark-coordinate columns and a label
columns = (
    [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
    + ["label"]
)

if not DATA_FILE.exists():
    with DATA_FILE.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)


def extract_features(landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32
    )

    # Translate so the wrist is the origin
    points -= points[0]

    # Normalize by hand size to reduce distance effects
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points /= scale

    return points.flatten()


with vision.HandLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("GestureBridge dataset collector")
    print("Press Q to quit at any time.")
    print("Press SPACE to start collecting a gesture.")

    try:
        for gesture in GESTURES:
            collected = 0
            print(f"\nNext gesture: {gesture}")
            print("Make the gesture and press SPACE to collect.")

            while collected < SAMPLES_PER_GESTURE:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(time.monotonic() * 1000)
                result = detector.detect_for_video(image, timestamp)

                features = None
                if result.hand_landmarks:
                    features = extract_features(
                        result.hand_landmarks[0]
                    )

                    pts = result.hand_landmarks[0]
                    height, width, _ = frame.shape
                    coords = [
                        (int(p.x * width), int(p.y * height))
                        for p in pts
                    ]

                    for start, end in [
                        (0,1),(1,2),(2,3),(3,4),(0,5),
                        (5,6),(6,7),(7,8),(5,9),(9,10),
                        (10,11),(11,12),(9,13),(13,14),
                        (14,15),(15,16),(13,17),(17,18),
                        (18,19),(19,20),(0,17)
                    ]:
                        cv2.line(
                            frame, coords[start], coords[end],
                            (0, 255, 0), 2
                        )

                    for x, y in coords:
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                cv2.putText(
                    frame, f"Gesture: {gesture}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Samples: {collected}/{SAMPLES_PER_GESTURE}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, "SPACE: capture | Q: quit",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                cv2.imshow("GestureBridge Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    raise SystemExit

                if key == ord(" ") and features is not None:
                    row = list(features) + [gesture]
                    with DATA_FILE.open(
                        "a", newline="", encoding="utf-8"
                    ) as f:
                        csv.writer(f).writerow(row)

                    collected += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

print(f"\nCollection finished. Data saved to {DATA_FILE}")
