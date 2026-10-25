
import cv2
import mediapipe as mp
import numpy as np
import joblib
import subprocess
import threading
import queue
import time
import tkinter as tk

from pathlib import Path
from collections import deque, Counter
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gesture_model.pkl"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 0.65
STABLE_SECONDS = 0.5
SMOOTHING_FRAMES = 7

BG = "#0B1120"
PANEL = "#151F32"
PANEL_LIGHT = "#1C2940"
ACCENT = "#42D9C8"
TEXT = "#F3F6FC"
MUTED = "#93A4BC"
GREEN = "#35D49A"
RED = "#FF6B78"
YELLOW = "#F4C76B"

GESTURE_LABELS = ["HELLO", "YES", "NO", "OK", "PEACE"]


# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {HAND_MODEL_PATH}"
    )

bundle = joblib.load(MODEL_PATH)

if isinstance(bundle, dict):
    model = bundle.get("model")
    if model is None:
        raise ValueError("The model bundle has no 'model' entry.")
else:
    model = bundle

if not hasattr(model, "predict_proba"):
    raise TypeError("The loaded model does not support predict_proba().")

print("[INFO] GestureBridge model loaded.")


# =========================================================
# WINDOWS SPEECH WORKER
# =========================================================

speech_queue = queue.Queue()
speech_events = queue.Queue()


def speech_worker():
    while True:
        word = speech_queue.get()

        try:
            if word is None:
                return

            speech_events.put(("speaking", word))

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
                speech_events.put(("finished", word))
            else:
                error = result.stderr.strip()
                print("[VOICE ERROR]", error)
                speech_events.put(("error", error or "Speech failed"))

        except Exception as error:
            print("[VOICE ERROR]", error)
            speech_events.put(("error", str(error)))

        finally:
            speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True,
)
speech_thread.start()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(hand_landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )

    points = points - points[0]

    distances = np.linalg.norm(points[:, :2], axis=1)
    scale = np.max(distances)

    if scale > 1e-6:
        points = points / scale

    return points.flatten().reshape(1, 63)


# =========================================================
# DESKTOP APPLICATION
# =========================================================

class GestureBridgeApp:

    def __init__(self, root):
        self.root = root
        self.root.title("GestureBridge")
        self.root.geometry("1180x780")
        self.root.minsize(940, 650)
        self.root.configure(bg=BG)

        self.camera = None
        self.landmarker = None
        self.running = False
        self.closing = False

        self.frame_job = None
        self.event_job = None
        self.current_photo = None

        self.start_time = 0.0
        self.last_timestamp_ms = -1

        self.prediction_history = deque(maxlen=SMOOTHING_FRAMES)
        self.candidate_label = None
        self.candidate_start_time = 0.0
        self.last_spoken_label = None

        self.last_spoken_text = "No gesture spoken yet"
        self.speech_state = "idle"

        self.build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.event_job = self.root.after(100, self.poll_speech_events)

    # -----------------------------------------------------
    # UI HELPERS
    # -----------------------------------------------------

    def make_label(self, parent, text, size=10, color=TEXT,
                   weight="normal", **kwargs):
        return tk.Label(
            parent,
            text=text,
            font=("Segoe UI", size, weight),
            fg=color,
            bg=kwargs.pop("bg", PANEL),
            **kwargs,
        )

    def make_card(self, parent, padx=18, pady=18):
        return tk.Frame(
            parent,
            bg=PANEL,
            padx=padx,
            pady=pady,
            highlightbackground="#26344C",
            highlightthickness=1,
        )

    # -----------------------------------------------------
    # BUILD INTERFACE
    # -----------------------------------------------------

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "GB.Horizontal.TProgressbar",
            troughcolor="#29364C",
            background=ACCENT,
            bordercolor="#29364C",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=8,
        )

        style.configure(
            "GB.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 10),
            background=PANEL_LIGHT,
            foreground=TEXT,
            borderwidth=0,
        )
        style.map(
            "GB.TButton",
            background=[("active", "#30415D"), ("disabled", "#202A3B")],
            foreground=[("disabled", "#718098")],
        )

        style.configure(
            "GB.Start.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 10),
            background=ACCENT,
            foreground="#07151B",
            borderwidth=0,
        )
        style.map(
            "GB.Start.TButton",
            background=[("active", "#67E9DB"), ("disabled", "#30415D")],
            foreground=[("disabled", "#8291A8")],
        )

        # Header
        header = tk.Frame(self.root, bg=BG, padx=28, pady=20)
        header.pack(fill="x")

        brand_row = tk.Frame(header, bg=BG)
        brand_row.pack(fill="x")

        logo = tk.Frame(
            brand_row,
            bg=ACCENT,
            width=44,
            height=44,
        )
        logo.pack(side="left", padx=(0, 12))
        logo.pack_propagate(False)

        self.make_label(
            logo, "GB", 15, "#07151B", "bold", bg=ACCENT
        ).place(relx=0.5, rely=0.5, anchor="center")

        title_block = tk.Frame(brand_row, bg=BG)
        title_block.pack(side="left")

        self.make_label(
            title_block, "GestureBridge", 22, TEXT, "bold", bg=BG
        ).pack(anchor="w")

        self.make_label(
            title_block,
            "HAND GESTURE RECOGNITION  /  VOICE ASSISTANT",
            9,
            MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(2, 0))

        self.header_status = tk.Label(
            brand_row,
            text="●  CAMERA OFF",
            font=("Segoe UI", 9, "bold"),
            fg=MUTED,
            bg=BG,
        )
        self.header_status.pack(side="right", anchor="center")

        # Thin accent line
        tk.Frame(header, bg="#26344C", height=1).pack(
            fill="x", pady=(18, 0)
        )

        # Main content
        content = tk.Frame(self.root, bg=BG, padx=28, pady=4)
        content.pack(fill="both", expand=True)

        content.grid_columnconfigure(0, weight=7)
        content.grid_columnconfigure(1, weight=3)
        content.grid_rowconfigure(0, weight=1)

        # Camera card
        camera_card = self.make_card(content, 16, 16)
        camera_card.grid(
            row=0, column=0, sticky="nsew", padx=(0, 12)
        )

        camera_header = tk.Frame(camera_card, bg=PANEL)
        camera_header.pack(fill="x", pady=(0, 14))

        self.make_label(
            camera_header, "LIVE CAMERA", 11, TEXT, "bold"
        ).pack(side="left")

        self.camera_indicator = tk.Label(
            camera_header,
            text="● STANDBY",
            font=("Segoe UI", 9, "bold"),
            fg=MUTED,
            bg=PANEL,
        )
        self.camera_indicator.pack(side="right")

        self.camera_label = tk.Label(
            camera_card,
            text="",
            bg="#070D17",
            fg=MUTED,
            font=("Segoe UI", 14),
            justify="center",
            bd=0,
        )
        self.camera_label.pack(fill="both", expand=True)

        self.camera_label.configure(
            text="CAMERA STANDBY\n\n"
                 "Start the camera to begin recognition",
        )

        camera_footer = tk.Frame(camera_card, bg=PANEL)
        camera_footer.pack(fill="x", pady=(12, 0))

        self.make_label(
            camera_footer,
            "Keep your hand clearly visible",
            9,
            MUTED,
        ).pack(side="left")

        self.make_label(
            camera_footer,
            "5 GESTURES",
            9,
            ACCENT,
            "bold",
        ).pack(side="right")

        # Right-hand column
        side = tk.Frame(content, bg=BG)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(0, weight=1)

        # Recognition card
        recognition = self.make_card(side, 18, 18)
        recognition.grid(row=0, column=0, sticky="new")

        self.make_label(
            recognition, "RECOGNIZED GESTURE", 10, MUTED, "bold"
        ).pack(anchor="w")

        self.gesture_var = tk.StringVar(value="—")
        self.make_label(
            recognition,
            "",
            1,
        )  # Keeps layout independent of font scaling.

        self.gesture_label = self.make_label(
            recognition, "—", 29, ACCENT, "bold", wraplength=260
        )
        self.gesture_label.pack(anchor="w", pady=(12, 4))
        self.gesture_label.configure(textvariable=self.gesture_var)

        self.make_label(
            recognition, "MODEL CONFIDENCE", 9, MUTED, "bold"
        ).pack(anchor="w", pady=(18, 7))

        confidence_row = tk.Frame(recognition, bg=PANEL)
        confidence_row.pack(fill="x")

        self.confidence_var = tk.StringVar(value="0%")
        self.make_label(
            confidence_row, "", 1
        ).pack(side="left")

        self.confidence_label = tk.Label(
            confidence_row,
            textvariable=self.confidence_var,
            font=("Segoe UI", 15, "bold"),
            fg=TEXT,
            bg=PANEL,
        )
        self.confidence_label.pack(side="left")

        self.confidence_bar = ttk.Progressbar(
            recognition,
            style="GB.Horizontal.TProgressbar",
            maximum=100,
            mode="determinate",
        )
        self.confidence_bar.pack(fill="x", pady=(10, 0))

        # Speech card
        speech = self.make_card(side, 18, 16)
        speech.grid(row=1, column=0, sticky="new", pady=(12, 0))

        speech_header = tk.Frame(speech, bg=PANEL)
        speech_header.pack(fill="x")

        self.make_label(
            speech_header, "VOICE OUTPUT", 10, MUTED, "bold"
        ).pack(side="left")

        self.speech_dot = tk.Label(
            speech_header,
            text="●",
            font=("Segoe UI", 14),
            fg=MUTED,
            bg=PANEL,
        )
        self.speech_dot.pack(side="right")

        self.speech_var = tk.StringVar(value="Waiting for gesture")
        tk.Label(
            speech,
            textvariable=self.speech_var,
            font=("Segoe UI", 11, "bold"),
            fg=TEXT,
            bg=PANEL,
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(12, 5))

        self.last_spoken_var = tk.StringVar(
            value="No gesture spoken yet"
        )
        tk.Label(
            speech,
            textvariable=self.last_spoken_var,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=PANEL,
            wraplength=250,
            justify="left",
        ).pack(anchor="w")

        # Supported gestures
        supported = self.make_card(side, 18, 14)
        supported.grid(row=2, column=0, sticky="new", pady=(12, 0))

        self.make_label(
            supported, "SUPPORTED GESTURES", 9, MUTED, "bold"
        ).pack(anchor="w", pady=(0, 10))

        chips = tk.Frame(supported, bg=PANEL)
        chips.pack(fill="x")

        for gesture in GESTURE_LABELS:
            tk.Label(
                chips,
                text=gesture,
                font=("Segoe UI", 8, "bold"),
                fg=ACCENT,
                bg=PANEL_LIGHT,
                padx=7,
                pady=5,
            ).pack(side="left", padx=(0, 5), pady=2)

        # Bottom controls
        controls = tk.Frame(self.root, bg=BG, padx=28, pady=18)
        controls.pack(fill="x")

        self.start_button = ttk.Button(
            controls,
            text="▶  START CAMERA",
            style="GB.Start.TButton",
            command=self.start_camera,
        )
        self.start_button.pack(side="left", padx=(0, 8))

        self.stop_button = ttk.Button(
            controls,
            text="■  STOP CAMERA",
            style="GB.TButton",
            command=self.stop_camera,
            state="disabled",
        )
        self.stop_button.pack(side="left")

        ttk.Button(
            controls,
            text="EXIT",
            style="GB.TButton",
            command=self.close_app,
        ).pack(side="right")

        # Footer/status bar
        footer = tk.Frame(self.root, bg=PANEL, padx=28, pady=10)
        footer.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready to start")
        tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=PANEL,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            footer,
            text="GESTUREBRIDGE  •  LOCAL PROCESSING",
            font=("Segoe UI", 8, "bold"),
            fg="#71829B",
            bg=PANEL,
        ).pack(side="right")

    # -----------------------------------------------------
    # CAMERA START
    # -----------------------------------------------------

    def start_camera(self):
        if self.running or self.closing:
            return

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera.release()
            messagebox.showerror(
                "Camera Error",
                "Could not open the webcam.\n"
                "Close other applications using the camera and retry.",
            )
            return

        try:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(HAND_MODEL_PATH)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            landmarker = vision.HandLandmarker.create_from_options(
                options
            )

        except Exception as error:
            camera.release()
            messagebox.showerror("MediaPipe Error", str(error))
            return

        self.camera = camera
        self.landmarker = landmarker
        self.running = True

        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1
        self.reset_prediction_state()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self.header_status.configure(
            text="●  CAMERA ON", fg=GREEN
        )
        self.camera_indicator.configure(
            text="● LIVE", fg=GREEN
        )
        self.status_var.set("Camera running — show a gesture")
        self.speech_var.set("Waiting for gesture")
        self.speech_dot.configure(fg=MUTED)

        self.update_frame()

    # -----------------------------------------------------
    # PREDICTION STATE
    # -----------------------------------------------------

    def reset_prediction_state(self):
        self.prediction_history.clear()
        self.candidate_label = None
        self.candidate_start_time = 0.0
        self.last_spoken_label = None

    # -----------------------------------------------------
    # CAMERA LOOP
    # -----------------------------------------------------

    def update_frame(self):
        if not self.running or self.closing:
            return

        success, frame = self.camera.read()

        if not success:
            self.status_var.set("Camera frame unavailable")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )
        timestamp_ms = max(
            timestamp_ms, self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        results = self.landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        label = "No hand detected"
        confidence = 0.0

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            features = extract_features(landmarks)

            probabilities = model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))

            raw_label = str(model.classes_[best_index])
            raw_confidence = float(probabilities[best_index])

            label = raw_label
            confidence = raw_confidence

            height, width = frame.shape[:2]
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(
                    frame, (x, y), 4, (66, 217, 200), -1
                )

            self.process_prediction(raw_label, raw_confidence)

        else:
            self.reset_prediction_state()
            self.gesture_var.set("—")
            self.speech_var.set("Waiting for gesture")
            self.speech_dot.configure(fg=MUTED)

        self.confidence_var.set(f"{confidence * 100:.1f}%")
        self.confidence_bar["value"] = confidence * 100

        if confidence >= CONFIDENCE_THRESHOLD:
            self.confidence_label.configure(fg=GREEN)
        else:
            self.confidence_label.configure(fg=YELLOW)

        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(display_rgb)

        available_width = max(self.camera_label.winfo_width(), 400)
        available_height = max(self.camera_label.winfo_height(), 300)
        image.thumbnail((available_width, available_height))

        self.current_photo = ImageTk.PhotoImage(image)
        self.camera_label.configure(
            image=self.current_photo,
            text="",
        )

        self.frame_job = self.root.after(15, self.update_frame)

    # -----------------------------------------------------
    # STABILITY AND SPEECH
    # -----------------------------------------------------

    def process_prediction(self, raw_label, confidence):
        if confidence < CONFIDENCE_THRESHOLD:
            self.prediction_history.clear()
            self.candidate_label = None
            self.candidate_start_time = 0.0
            self.gesture_var.set("Uncertain")
            self.speech_var.set("Low confidence")
            self.speech_dot.configure(fg=YELLOW)
            return

        self.prediction_history.append(raw_label)

        counts = Counter(self.prediction_history)
        stable_label, count = counts.most_common(1)[0]

        required_count = len(self.prediction_history) // 2 + 1

        if count < required_count:
            self.candidate_label = None
            self.candidate_start_time = 0.0
            self.gesture_var.set("Detecting…")
            self.speech_var.set("Stabilizing gesture")
            self.speech_dot.configure(fg=YELLOW)
            return

        now = time.monotonic()

        if stable_label != self.candidate_label:
            self.candidate_label = stable_label
            self.candidate_start_time = now

        if now - self.candidate_start_time < STABLE_SECONDS:
            self.gesture_var.set(stable_label)
            self.speech_var.set("Confirming gesture…")
            self.speech_dot.configure(fg=YELLOW)
            return

        self.gesture_var.set(stable_label)

        if stable_label != self.last_spoken_label:
            speech_queue.put(stable_label)
            self.last_spoken_label = stable_label
            self.speech_var.set(f"Queued: {stable_label}")
            self.speech_dot.configure(fg=YELLOW)
            self.status_var.set(f"Recognized {stable_label} — speech queued")

    # -----------------------------------------------------
    # SPEECH STATUS EVENTS
    # -----------------------------------------------------

    def poll_speech_events(self):
        if self.closing:
            return

        while True:
            try:
                event, value = speech_events.get_nowait()
            except queue.Empty:
                break

            if event == "speaking":
                self.speech_var.set(f"Speaking: {value}")
                self.speech_dot.configure(fg=GREEN)
                self.last_spoken_var.set(f"Now speaking: {value}")

            elif event == "finished":
                self.speech_var.set("Speech complete")
                self.speech_dot.configure(fg=ACCENT)
                self.last_spoken_var.set(f"Last spoken: {value}")

            elif event == "error":
                self.speech_var.set("Speech error")
                self.speech_dot.configure(fg=RED)
                self.last_spoken_var.set(str(value))

        self.event_job = self.root.after(
            100, self.poll_speech_events
        )

    # -----------------------------------------------------
    # STOP CAMERA
    # -----------------------------------------------------

    def stop_camera(self):
        self.running = False

        if self.frame_job is not None:
            try:
                self.root.after_cancel(self.frame_job)
            except tk.TclError:
                pass
            self.frame_job = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        self.current_photo = None
        self.camera_label.configure(
            image="",
            text="CAMERA STANDBY\n\nStart the camera to begin recognition",
        )

        self.gesture_var.set("—")
        self.confidence_var.set("0%")
        self.confidence_bar["value"] = 0
        self.confidence_label.configure(fg=TEXT)

        self.speech_var.set("Waiting for gesture")
        self.speech_dot.configure(fg=MUTED)

        self.header_status.configure(
            text="●  CAMERA OFF", fg=MUTED
        )
        self.camera_indicator.configure(
            text="● STANDBY", fg=MUTED
        )

        self.status_var.set("Camera stopped")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

        self.reset_prediction_state()

    # -----------------------------------------------------
    # EXIT
    # -----------------------------------------------------

    def close_app(self):
        if self.closing:
            return

        self.closing = True
        self.running = False

        if self.frame_job is not None:
            try:
                self.root.after_cancel(self.frame_job)
            except tk.TclError:
                pass

        if self.event_job is not None:
            try:
                self.root.after_cancel(self.event_job)
            except tk.TclError:
                pass

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        speech_queue.put(None)
        self.root.destroy()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureBridgeApp(root)
    root.mainloop()
