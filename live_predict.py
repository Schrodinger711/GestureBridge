
import cv2
import mediapipe as mp
import numpy as np
import joblib
import subprocess
import threading
import queue
import time

from pathlib import Path
from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. FILE PATHS AND SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "gesture_model.pkl"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 0.65
STABLE_SECONDS = 0.5
SMOOTHING_FRAMES = 7

GESTURE_LABELS = ["HELLO", "YES", "NO", "OK", "PEACE"]


# =========================================================
# 2. LOAD TRAINED GESTURE MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model file not found: {HAND_MODEL_PATH}"
    )

# Load the saved model bundle
bundle = joblib.load(MODEL_PATH)

# Extract the trained classifier and feature information
model = bundle["model"]
feature_columns = bundle["feature_columns"]
labels = bundle["labels"]

print("[INFO] Gesture model loaded successfully.")
print("[INFO] Labels:", labels)


# =========================================================
# 3. SPEECH QUEUE AND WORKER
# =========================================================

speech_queue = queue.Queue()


def speech_worker():
    """Speak queued gesture labels using Windows PowerShell."""

    while True:
        word = speech_queue.get()

        try:
            if word is None:
                return

            print(f"[VOICE] Speaking: {word}")

            # The labels are fixed, known gesture names.
            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object "
                "System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{word}'); "
                "$s.Dispose()"
            )

            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

            if result.returncode == 0:
                print(f"[VOICE] Finished: {word}")
            else:
                print("[VOICE ERROR]", result.stderr.strip())

        except Exception as error:
            print(f"[VOICE ERROR] {error}")

        finally:
            speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True,
)

speech_thread.start()


# =========================================================
# 4. MEDIAPIPE HAND LANDMARKER
# =========================================================

BaseOptions = python.BaseOptions

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = vision.HandLandmarker.create_from_options(options)


# =========================================================
# 5. FEATURE EXTRACTION
# =========================================================

def extract_features(hand_landmarks):
    """
    Convert 21 hand landmarks into the same 63-feature
    representation used during training.
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )

    # Make the wrist the origin.
    points = points - points[0]

    # Normalize using the largest XY distance from the wrist.
    distances = np.linalg.norm(points[:, :2], axis=1)
    scale = np.max(distances)

    if scale > 1e-6:
        points = points / scale

    # Preserve interleaved x, y, z ordering.
    return points.flatten().reshape(1, 63)


# =========================================================
# 6. PREDICTION SMOOTHING
# =========================================================

prediction_history = deque(maxlen=SMOOTHING_FRAMES)

candidate_label = None
candidate_start_time = 0.0
last_spoken_label = None


def reset_prediction_state():
    global candidate_label, candidate_start_time
    global last_spoken_label

    prediction_history.clear()
    candidate_label = None
    candidate_start_time = 0.0
    last_spoken_label = None


# =========================================================
# 7. START WEBCAM
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    landmarker.close()
    speech_queue.put(None)
    raise RuntimeError(
        "Could not open webcam. Check your camera connection."
    )

print("[INFO] Webcam started.")
print("[INFO] Show one of your five trained gestures.")
print("[INFO] Press Q to quit.")


# =========================================================
# 8. MAIN RECOGNITION LOOP
# =========================================================

start_time = time.monotonic()
last_timestamp_ms = -1

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("[WARNING] Could not read webcam frame.")
            break

        # Mirror the webcam image.
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # VIDEO mode requires increasing timestamps.
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )
        timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms

        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        display_label = "No hand detected"
        display_confidence = 0.0

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            features = extract_features(hand_landmarks)

            probabilities = model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))

            raw_label = str(model.classes_[best_index])
            raw_confidence = float(probabilities[best_index])

            display_label = raw_label
            display_confidence = raw_confidence

            # Only accept sufficiently confident predictions.
            if raw_confidence >= CONFIDENCE_THRESHOLD:
                prediction_history.append(raw_label)
            else:
                prediction_history.clear()
                candidate_label = None
                candidate_start_time = 0.0

            # Determine the most common recent prediction.
            if prediction_history:
                counts = Counter(prediction_history)
                stable_label, count = counts.most_common(1)[0]

                # Require a majority of the smoothing window.
                required_count = (len(prediction_history) // 2) + 1

                if count >= required_count:
                    now = time.monotonic()

                    if stable_label != candidate_label:
                        candidate_label = stable_label
                        candidate_start_time = now

                    # Speak only after the prediction remains stable.
                    stable_duration = now - candidate_start_time

                    if (
                        stable_duration >= STABLE_SECONDS
                        and stable_label != last_spoken_label
                    ):
                        speech_queue.put(stable_label)
                        last_spoken_label = stable_label

                else:
                    candidate_label = None
                    candidate_start_time = 0.0

            # Draw hand landmarks.
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1,
                )

        else:
            # Allow the same gesture to be spoken again
            # after the hand leaves the camera view.
            reset_prediction_state()

        # Display current prediction and confidence.
        cv2.putText(
            frame,
            f"Gesture: {display_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {display_confidence:.0%}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("GestureBridge", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Release the webcam and MediaPipe resources.
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Finish queued speech before stopping the worker.
    speech_queue.put(None)
    speech_thread.join(timeout=20)

    print("[INFO] GestureBridge stopped.")
