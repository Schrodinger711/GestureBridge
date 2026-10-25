# GestureBridge

**Windows-only hand gesture recognition and voice feedback prototype.**

GestureBridge uses a webcam to recognize five trained hand gestures and speaks the recognized label using Windows speech capabilities.

## Supported Gestures

- HELLO
- YES
- NO
- OK
- PEACE

## Features

- Webcam-based hand tracking
- Machine-learning gesture classification
- Prediction labels and confidence display
- Voice feedback for stable predictions
- Local processing of webcam frames

## Requirements

- Compatible 64-bit Windows computer
- Working webcam
- Python and project dependencies for running from source
- Windows PowerShell and `Windows.System.Speech` support as required by the application

## Run the Packaged Application

1. Download the Windows release ZIP from GitHub Releases.
2. Extract the entire ZIP.
3. Keep `GestureBridge.exe` and its `_internal` folder together.
4. Run `GestureBridge.exe`.
5. Click **START CAMERA** and show a supported gesture.
6. Click **STOP CAMERA** when finished.

Windows SmartScreen may warn that the application is unrecognized because it is not signed by a recognized publisher. Only run software from a source you trust.

## Project Files

- `desktop_app.py` — desktop application
- `live_predict.py` — gesture prediction
- `hand_tracking.py` — hand tracking logic
- `gesture_model.pkl` — trained gesture model
- `hand_landmarker.task` — hand landmark model
- `collect_data.py` — gesture data collection
- `train_model.py` — model training
- `export_model.py` — model export utility

## Privacy and Limitations

GestureBridge is an educational prototype. Recognition accuracy may vary with lighting, camera quality, and hand position. Predictions are not guaranteed to be correct.

The application is designed to process webcam frames locally. Do not record or monitor other people without their knowledge and consent.

## Platform

**Windows desktop only.** No mobile or Android version is included in this project.
