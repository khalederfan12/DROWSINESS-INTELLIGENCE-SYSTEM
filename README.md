# Drowsiness Intelligence System — Streamlit Dashboard

نظام لحظي لكشف النعاس من الكاميرا: EAR للعين + MAR للفم (تثاؤب) + تقدير وضع الرأس
(Pitch/Yaw) → محرك دمج يحسب Drowsiness Score من 0 إلى 100 مع إنذار صوتي عند الخطر.

## Files
- `streamlit_app.py` — main app: camera loop, dashboard, alarm, session state, FPS.
- `mouth_analysis.py`, `head_pose_estimator.py`, `drowsiness_engine.py` — teammates' modules (unchanged, imported as-is).
- `face_landmarker.task` — MediaPipe face landmark model (must sit next to `streamlit_app.py`).
- `alarm.wav` — alarm sound, played automatically in-browser when status is CRITICAL.

## v2 additions
- **Eye / mouth tracking points**: the exact landmarks used for EAR and MAR are
  drawn as small ring markers on the live feed (toggle in the sidebar).
- **Background blur**: uses MediaPipe's `selfie_segmentation` to keep the
  person sharp and Gaussian-blur everything behind them (toggle + strength
  slider in the sidebar). Off by default since it costs extra CPU per frame.
- Redesigned dark dashboard with a circular conic-gradient score gauge
  instead of a plain progress bar. English-only UI.

## Fixes applied (v2.1)
- **Import bug**: `streamlit_app.py` was importing a class called `DrowsinessEngine`
  that doesn't exist in `drowsiness_engine.py` (the real class is
  `DrowsinessIntelligenceEngine`). This caused an immediate crash on startup — fixed.
- **Asset paths**: the app was looking for the model/alarm files inside a nonexistent
  `assets/` subfolder. `face_landmarker.task` and `alarm.wav` must sit directly next
  to `streamlit_app.py` (as this README already said) — path fixed to match.
- **Camera resource leak (the "disconnects after a while" bug)**: the live-video
  `while` loop held the camera open with no `try/finally`. Any Streamlit rerun
  (moving a sidebar slider, etc.) interrupts the script mid-loop, so
  `cap.release()` was being skipped — the camera stayed locked and the next
  "Start" click failed or the app appeared to hang/error. The loop is now wrapped
  in `try/finally` so the camera is always released, plus a per-frame
  `try/except` so one bad frame just gets skipped instead of killing the session.
- **Native Deploy button**: the custom CSS was hiding Streamlit's entire
  `header`, which also hides its built-in **Deploy** button and toolbar. The CSS
  now only hides the hamburger menu/footer, so the Deploy button is visible again
  (top-right of the app, both locally and once pushed to Streamlit Community Cloud).
- **`mediapipe` version crash**: `requirements.txt` had `mediapipe>=0.10.0` with no
  upper bound, so `pip` installed the newest release — which dropped the legacy
  `mp.solutions` API used by the background-blur feature
  (`AttributeError: module 'mediapipe' has no attribute 'solutions'`, usually
  surfacing the first time you touch a widget and Streamlit reruns the script).
  `requirements.txt` now pins `mediapipe==0.10.14`, a version that still ships
  both the legacy `solutions` API and the Tasks API this app relies on. The app
  also now degrades gracefully (blur just turns itself off with a warning
  instead of crashing) if a future mediapipe version removes `solutions` again.
- **Python version pinned to 3.11**: added `runtime.txt` (`python-3.11`) so
  Streamlit Community Cloud (and other buildpack-based hosts) provision Python
  3.11 specifically — `mediapipe` wheels aren't reliably available for newer
  Python versions yet. Run locally with `python3.11` / `py -3.11` as well.

## Run locally
```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 -m streamlit run streamlit_app.py
```
(On macOS/Linux use `python3.11` instead of `py -3.11`.)

Grant the browser/OS webcam permission, press **Start**.

## Deployment notes (Streamlit Community Cloud / any server)
- `cv2.VideoCapture(0)` reads a **server-side** camera. On Streamlit Cloud there is
  no physical webcam attached to the server, so live video will not work there
  out of the box — this app is built to be demoed **locally** or on a machine
  with an attached camera (e.g. during a viva/presentation).
- `packages.txt` is included so `opencv-python-headless` and `mediapipe` have the
  system libraries they need (`libgl1`, `libglib2.0-0`, etc.) if you do deploy it.
- If browser-side webcam access on the cloud is required later, swap the capture
  loop for `streamlit-webrtc` (`VideoProcessorBase`), which streams frames from
  the visitor's own browser instead of the server. The processing functions in
  this file (`eye_aspect_ratio`, module calls, `push_alarm`) can be reused as-is
  inside a `recv(frame)` callback.

## Performance notes
- Sidebar slider **"Frame-skip"** lets you run mouth/head-pose analysis every
  1–3 frames instead of every frame (EAR/eye-closure always runs every frame,
  since it's the cheapest and most safety-critical signal).
- FPS is measured live and shown in the metrics panel.
- The score history chart updates from an in-memory rolling deque (last 150
  points) instead of growing unbounded.
