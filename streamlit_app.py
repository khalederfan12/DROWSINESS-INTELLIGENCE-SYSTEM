import os
import time
import base64
import collections

import cv2
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from mouth_analysis import MouthAnalyzer
from head_pose_estimator import HeadPoseEstimator
from drowsiness_engine import DrowsinessIntelligenceEngine
# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

MOUTH_POINTS = [61, 291, 13, 14, 81, 178, 311, 402]  # corners + inner-lip pairs

EYE_COLOR = (255, 210, 60)     # cyan-ish blue in BGR
MOUTH_COLOR = (60, 210, 255)   # amber in BGR

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Drowsiness Intelligence System", page_icon="◍", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    /* Keep the native Streamlit header visible so the "Deploy" button still shows;
       just hide the hamburger menu icon inside it. */
    header [data-testid="stMainMenu"] {visibility: hidden;}
    .stApp {
        background: radial-gradient(circle at 20% 0%, #14161d 0%, #0b0c10 55%, #08090c 100%);
    }
    .app-title {
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 30px; font-weight: 800; letter-spacing: 1px;
        color: #f2f3f5; margin-bottom: 0;
    }
    .app-sub { color: #7c8291; font-size: 14px; margin-top: -4px; }

    .panel {
        background: linear-gradient(160deg, #16181f 0%, #101218 100%);
        border: 1px solid #262a35;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .panel-title {
        font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase;
        color: #6f7686; margin-bottom: 10px;
    }

    .gauge-wrap { display: flex; flex-direction: column; align-items: center; }
    .gauge {
        width: 168px; height: 168px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        position: relative;
    }
    .gauge-inner {
        width: 128px; height: 128px; border-radius: 50%;
        background: #0e0f14;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .gauge-score { font-size: 34px; font-weight: 800; color: #f2f3f5; line-height: 1; }
    .gauge-label { font-size: 11px; letter-spacing: 1px; color: #6f7686; margin-top: 4px; text-transform: uppercase; }

    .pill {
        display: inline-block; padding: 7px 18px; border-radius: 999px;
        font-size: 13px; font-weight: 700; letter-spacing: 0.5px; color: #0b0c10;
        margin-top: 12px;
    }
    .action-text { color: #a7adba; font-size: 13px; margin-top: 8px; }

    .stat-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 9px 2px; border-bottom: 1px solid #1e212a; font-size: 14px;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-key { color: #8a90a0; }
    .stat-val { color: #f2f3f5; font-weight: 700; font-variant-numeric: tabular-nums; }

    div.stButton > button {
        border-radius: 10px; border: 1px solid #2b2f3a; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_landmarker(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


@st.cache_resource(show_spinner=False)
def load_segmenter():
    """Legacy MediaPipe Selfie Segmentation — used to blur everything
    except the person for the background-blur toggle.
    Returns None (and the caller disables blur) if this legacy API isn't
    available in the installed mediapipe build."""
    if not hasattr(mp, "solutions"):
        return None
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


@st.cache_data(show_spinner=False)
def get_alarm_base64(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def eye_aspect_ratio(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def blur_background(frame, segmenter, blur_strength=35):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = segmenter.process(rgb).segmentation_mask
    condition = mask > 0.5
    condition = np.stack((condition,) * 3, axis=-1)
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return np.where(condition, frame, blurred)


def draw_tracking_points(frame, landmarks, w, h):
    """Draw small ring markers over the eye and mouth landmarks used for
    EAR / MAR so the detection is visible on the live feed."""
    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 4, EYE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 1, EYE_COLOR, -1, cv2.LINE_AA)

    for idx in MOUTH_POINTS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 4, MOUTH_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 1, MOUTH_COLOR, -1, cv2.LINE_AA)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
defaults = {
    "run_camera": False,
    "score_history": collections.deque(maxlen=150),
    "eyes_closed_start": None,
    "alarm_playing": False,
    "last_alarm_push": 0.0,
    "frame_count": 0,
    "fps": 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.markdown("### Detection Settings")
ear_threshold = st.sidebar.slider("Eye Closure Threshold (EAR)", 0.10, 0.35, 0.20, 0.01)
mar_threshold = st.sidebar.slider("Mouth Open Threshold (MAR)", 0.30, 0.90, 0.55, 0.01)
pitch_thresh = st.sidebar.slider("Head-Down Pitch Threshold", -40.0, -5.0, -18.0, 1.0)
yaw_thresh = st.sidebar.slider("Head-Turn Yaw Threshold", 10.0, 45.0, 22.0, 1.0)
alert_thresh = st.sidebar.slider("Early Warning Threshold", 10.0, 60.0, 35.0, 1.0)
drowsy_thresh = st.sidebar.slider("Critical Threshold", 50.0, 100.0, 70.0, 1.0)

st.sidebar.markdown("### Camera & Performance")
camera_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
process_every_n = st.sidebar.slider("Frame-skip (mouth/head stages)", 1, 3, 1)
show_points = st.sidebar.checkbox("Show eye / mouth tracking points", value=True)
blur_bg = st.sidebar.checkbox("Blur background", value=False)
blur_strength = st.sidebar.slider("Blur strength", 15, 65, 35, 2, disabled=not blur_bg)

mouth_analyzer = MouthAnalyzer(mar_threshold=mar_threshold, yawn_time_threshold=1.5)
pose_estimator = HeadPoseEstimator(pitch_down_thresh=pitch_thresh, yaw_thresh=yaw_thresh)
drowsiness_engine = DrowsinessIntelligenceEngine(alert_thresh=alert_thresh, drowsy_thresh=drowsy_thresh)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown('<p class="app-title">DROWSINESS INTELLIGENCE</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Real-time driver monitoring · eye, mouth & head-pose fusion</p>', unsafe_allow_html=True)
with head_r:
    c1, c2 = st.columns(2)
    if c1.button("Start", use_container_width=True, type="primary"):
        st.session_state.run_camera = True
    if c2.button("Stop", use_container_width=True):
        st.session_state.run_camera = False

st.write("")

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
video_col, dash_col = st.columns([3, 2])

with video_col:
    st.markdown('<div class="panel"><div class="panel-title">Live Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Score Trend</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with dash_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    gauge_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Live Metrics</div>', unsafe_allow_html=True)
    metrics_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

alarm_placeholder = st.empty()
alarm_b64 = get_alarm_base64(ALARM_PATH)

LEVEL_COLOR = {"SAFE": "#22c55e", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}
LEVEL_LABEL = {"SAFE": "ALERT & FOCUSED", "WARNING": "EARLY DROWSINESS", "CRITICAL": "CRITICAL — PULL OVER"}


def render_gauge(level, score, action_text):
    color = LEVEL_COLOR[level]
    angle = min(max(score, 0), 100) * 3.6
    gauge_placeholder.markdown(
        f"""
        <div class="gauge-wrap">
          <div class="gauge" style="background: conic-gradient({color} {angle}deg, #1c1f28 0deg);">
            <div class="gauge-inner">
              <div class="gauge-score">{score:.0f}%</div>
              <div class="gauge-label">Drowsiness</div>
            </div>
          </div>
          <div class="pill" style="background:{color};">{LEVEL_LABEL[level]}</div>
          <div class="action-text">{action_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(m):
    rows = "".join(
        f"""<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>"""
        for k, v in m.items()
    )
    metrics_placeholder.markdown(rows, unsafe_allow_html=True)


def push_alarm():
    if alarm_b64 is None:
        return
    now = time.time()
    if now - st.session_state.last_alarm_push > 2.5:
        alarm_placeholder.markdown(
            f'<audio autoplay><source src="data:audio/wav;base64,{alarm_b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
        st.session_state.last_alarm_push = now


def stop_alarm():
    alarm_placeholder.empty()
    st.session_state.alarm_playing = False


# ------------------------------------------------------------------
# Main real-time loop
# ------------------------------------------------------------------
if st.session_state.run_camera:
    if not os.path.exists(MODEL_PATH):
        st.error(
            "face_landmarker.task model file not found. "
            "Make sure it sits in the same folder as streamlit_app.py."
        )
        st.session_state.run_camera = False
    else:
        landmarker = load_landmarker(MODEL_PATH)
        segmenter = load_segmenter() if blur_bg else None
        if blur_bg and segmenter is None:
            st.sidebar.warning(
                "Background blur isn't available with the installed mediapipe "
                "version (legacy `solutions` API missing) — continuing without it."
            )
        cap = cv2.VideoCapture(int(camera_idx))

        if not cap.isOpened():
            st.error("Could not open the camera. Check the connection / permissions.")
            st.session_state.run_camera = False
        else:
            timestamp_ms = 0
            fps_time = time.time()
            fps_counter = 0

            try:
                while st.session_state.run_camera:
                    success, frame = cap.read()
                    if not success:
                        st.warning("Lost camera stream.")
                        break

                    try:
                        frame = cv2.flip(frame, 1)
                        h, w, _ = frame.shape
                        st.session_state.frame_count += 1
                        run_heavy = (st.session_state.frame_count % process_every_n) == 0

                        if blur_bg and segmenter is not None:
                            frame = blur_background(frame, segmenter, blur_strength)

                        # Convert the current frame to MediaPipe Image once.
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb_frame
                        )

                        # VIDEO mode requires a strictly increasing timestamp.
                        timestamp_ms += 33
                        results = landmarker.detect_for_video(mp_image, timestamp_ms)

                        if results.face_landmarks:
                            landmarks = results.face_landmarks[0]

                            if show_points:
                                draw_tracking_points(frame, landmarks, w, h)

                            def get_pts(indices):
                                return [
                                    (landmarks[i].x * w, landmarks[i].y * h)
                                    for i in indices
                                ]

                            left_ear = eye_aspect_ratio(get_pts(LEFT_EYE_INDICES))
                            right_ear = eye_aspect_ratio(get_pts(RIGHT_EYE_INDICES))
                            ear = (left_ear + right_ear) / 2.0

                            if ear < ear_threshold:
                                if st.session_state.eyes_closed_start is None:
                                    st.session_state.eyes_closed_start = time.time()
                                closed_duration = (
                                    time.time() - st.session_state.eyes_closed_start
                                )
                            else:
                                st.session_state.eyes_closed_start = None
                                closed_duration = 0.0

                            eye_data = {
                                "ear": ear,
                                "is_closed": ear < ear_threshold,
                                "closed_duration": closed_duration,
                            }

                            if run_heavy:
                                mouth_data = mouth_analyzer.update(landmarks, w, h)
                                head_pose_data = pose_estimator.estimate_pose(landmarks, w, h)
                            else:
                                mouth_data = {
                                    "mar": 0.0,
                                    "is_yawning": False,
                                    "yawn_duration": 0.0,
                                    "total_yawns": mouth_analyzer.total_yawn_count,
                                }
                                head_pose_data = {
                                    "pitch": 0.0,
                                    "yaw": 0.0,
                                    "roll": 0.0,
                                    "is_head_down": False,
                                    "is_looking_away": False,
                                    "pose_status": "-",
                                }

                            result = drowsiness_engine.compute_drowsiness_score(
                                eye_data,
                                mouth_data,
                                head_pose_data,
                            )
                            score = result["drowsiness_score"]
                            level = result["level"]
                            st.session_state.score_history.append(score)

                            render_gauge(level, score, result["action"])
                            render_metrics({
                                "EAR": f"{ear:.3f}",
                                "MAR": f"{mouth_data['mar']:.3f}",
                                "Yawn Count": mouth_data["total_yawns"],
                                "Pitch": f"{head_pose_data['pitch']:.1f}°",
                                "Yaw": f"{head_pose_data['yaw']:.1f}°",
                                "Head Pose": head_pose_data["pose_status"],
                                "FPS": f"{st.session_state.fps:.1f}",
                            })

                            if level == "CRITICAL":
                                push_alarm()
                                st.session_state.alarm_playing = True
                            elif st.session_state.alarm_playing:
                                stop_alarm()
                        else:
                            render_gauge("SAFE", 0, "No face detected in frame")
                            stop_alarm()

                        fps_counter += 1
                        elapsed = time.time() - fps_time
                        if elapsed >= 1.0:
                            st.session_state.fps = fps_counter / elapsed
                            fps_counter = 0
                            fps_time = time.time()

                        video_placeholder.image(
                            frame,
                            channels="BGR",
                            use_container_width=True,
                        )

                        if st.session_state.score_history:
                            chart_placeholder.line_chart(
                                list(st.session_state.score_history),
                                height=160,
                            )
                    except Exception as frame_err:
                        st.warning(f"Skipped a bad frame: {frame_err}")

            finally:
                cap.release()
else:
    video_placeholder.info("Click **Start** to begin real-time monitoring.")
    render_gauge("SAFE", 0, "Camera is idle")
    stop_alarm()



###py -3.11 -m streamlit run streamlit_app.py