import os
import time
import base64
import collections
import threading

import av
import cv2
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from mouth_analysis import MouthAnalyzer
from head_pose_estimator import HeadPoseEstimator
from drowsiness_engine import DrowsinessIntelligenceEngine

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")

LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS      = [61, 291, 13, 14, 81, 178, 311, 402]

EYE_COLOR   = (255, 210, 60)
MOUTH_COLOR = (60, 210, 255)

# Processing resolution — downscale before ML inference to save RAM & CPU
PROC_WIDTH  = 320
PROC_HEIGHT = 240

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Drowsiness Intelligence System", page_icon="◍", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
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
    div.stButton > button { border-radius: 10px; border: 1px solid #2b2f3a; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Resources — use IMAGE mode (stateless) instead of VIDEO to save RAM
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_landmarker(model_path: str):
    """
    IMAGE running mode is stateless between calls — significantly lower
    memory footprint than VIDEO mode which keeps internal tracking buffers.
    """
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,   # ← key memory saving
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,               # ← skip blendshapes output
        output_facial_transformation_matrixes=False, # ← skip matrices output
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

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

def draw_tracking_points(frame, landmarks, w, h):
    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 3, EYE_COLOR,   2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 1, EYE_COLOR,  -1, cv2.LINE_AA)
    for idx in MOUTH_POINTS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 3, MOUTH_COLOR,  2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 1, MOUTH_COLOR, -1, cv2.LINE_AA)

# ------------------------------------------------------------------
# Video Processor — memory-optimised
# ------------------------------------------------------------------
class DrowsinessVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmarker = load_landmarker(MODEL_PATH)   # shared cached instance

        self.mouth_analyzer    = MouthAnalyzer(mar_threshold=0.55, yawn_time_threshold=1.5)
        self.pose_estimator    = HeadPoseEstimator(pitch_down_thresh=-18.0, yaw_thresh=22.0)
        self.drowsiness_engine = DrowsinessIntelligenceEngine(alert_thresh=35.0, drowsy_thresh=70.0)

        # Settings written from UI thread — simple Python scalars (GIL-safe to read)
        self.ear_threshold   = 0.20
        self.mar_threshold   = 0.55
        self.pitch_thresh    = -18.0
        self.yaw_thresh      = 22.0
        self.alert_thresh    = 35.0
        self.drowsy_thresh   = 70.0
        self.process_every_n = 2    # default skip=2 to halve CPU/memory pressure
        self.show_points     = True

        # Internal recv-thread state
        self.frame_count       = 0
        self.eyes_closed_start = None
        self._last_mouth       = None
        self._last_pose        = None
        self.fps_counter       = 0
        self.fps_time          = time.time()

        # Output — protected by lock
        self.lock        = threading.Lock()
        self.score       = 0.0
        self.level       = "SAFE"
        self.action_text = "Analyzing..."
        self.metrics     = {}
        self.fps         = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Decode incoming frame
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        disp_h, disp_w = img.shape[:2]

        self.frame_count += 1
        run_heavy = (self.frame_count % max(1, self.process_every_n)) == 0
        ear_thr   = self.ear_threshold

        # ── Downscale for ML inference ──────────────────────────────
        small = cv2.resize(img, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

        try:
            results = self.landmarker.detect(mp_image)   # IMAGE mode — no timestamp
        except Exception:
            results = None
        finally:
            del small, rgb_small, mp_image   # free immediately

        if results and results.face_landmarks:
            landmarks = results.face_landmarks[0]

            if self.show_points:
                # Draw on display-res frame using display coords
                draw_tracking_points(img, landmarks, disp_w, disp_h)

            def get_pts(indices):
                return [(landmarks[i].x * PROC_WIDTH, landmarks[i].y * PROC_HEIGHT) for i in indices]

            ear = (eye_aspect_ratio(get_pts(LEFT_EYE_INDICES)) +
                   eye_aspect_ratio(get_pts(RIGHT_EYE_INDICES))) / 2.0

            if ear < ear_thr:
                if self.eyes_closed_start is None:
                    self.eyes_closed_start = time.time()
                closed_dur = time.time() - self.eyes_closed_start
            else:
                self.eyes_closed_start = None
                closed_dur = 0.0

            eye_data = {"ear": ear, "is_closed": ear < ear_thr, "closed_duration": closed_dur}

            if run_heavy:
                mouth_data     = self.mouth_analyzer.update(landmarks, PROC_WIDTH, PROC_HEIGHT)
                head_pose_data = self.pose_estimator.estimate_pose(landmarks, PROC_WIDTH, PROC_HEIGHT)
                self._last_mouth = mouth_data
                self._last_pose  = head_pose_data
            else:
                mouth_data     = self._last_mouth or {
                    "mar": 0.0, "is_yawning": False, "yawn_duration": 0.0,
                    "total_yawns": self.mouth_analyzer.total_yawn_count,
                }
                head_pose_data = self._last_pose or {
                    "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                    "is_head_down": False, "is_looking_away": False, "pose_status": "-",
                }

            result = self.drowsiness_engine.compute_drowsiness_score(eye_data, mouth_data, head_pose_data)

            # FPS
            self.fps_counter += 1
            elapsed = time.time() - self.fps_time
            if elapsed >= 1.0:
                fps_now = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_time    = time.time()
            else:
                fps_now = self.fps

            new_metrics = {
                "EAR":        f"{ear:.3f}",
                "MAR":        f"{mouth_data['mar']:.3f}",
                "Yawn Count": mouth_data["total_yawns"],
                "Pitch":      f"{head_pose_data['pitch']:.1f}°",
                "Yaw":        f"{head_pose_data['yaw']:.1f}°",
                "Head Pose":  head_pose_data["pose_status"],
                "FPS":        f"{fps_now:.1f}",
            }

            with self.lock:
                self.score       = result["drowsiness_score"]
                self.level       = result["level"]
                self.action_text = result["action"]
                self.metrics     = new_metrics
                self.fps         = fps_now
        else:
            with self.lock:
                self.score       = 0.0
                self.level       = "SAFE"
                self.action_text = "No face detected in frame"
                self.metrics     = {}

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
defaults = {
    "score_history": collections.deque(maxlen=60),  # reduced from 150 → 60
    "alarm_active":  False,
    "last_alarm_ts": 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.markdown("### Detection Settings")
ear_threshold  = st.sidebar.slider("Eye Closure Threshold (EAR)", 0.10, 0.35, 0.20, 0.01)
mar_threshold  = st.sidebar.slider("Mouth Open Threshold (MAR)",  0.30, 0.90, 0.55, 0.01)
pitch_thresh   = st.sidebar.slider("Head-Down Pitch Threshold",  -40.0, -5.0, -18.0, 1.0)
yaw_thresh     = st.sidebar.slider("Head-Turn Yaw Threshold",     10.0, 45.0,  22.0, 1.0)
alert_thresh   = st.sidebar.slider("Early Warning Threshold",     10.0, 60.0,  35.0, 1.0)
drowsy_thresh  = st.sidebar.slider("Critical Threshold",          50.0,100.0,  70.0, 1.0)

st.sidebar.markdown("### Performance")
process_every_n = st.sidebar.slider("Frame-skip (higher = less CPU/RAM)", 1, 4, 2)
show_points     = st.sidebar.checkbox("Show eye / mouth tracking points", value=True)

st.sidebar.markdown(
    "<small style='color:#6f7686'>ℹ️ Background blur is disabled to stay within cloud memory limits.</small>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
head_l, _ = st.columns([3, 1])
with head_l:
    st.markdown('<p class="app-title">DROWSINESS INTELLIGENCE</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Real-time driver monitoring · eye, mouth & head-pose fusion</p>', unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
video_col, dash_col = st.columns([3, 2])

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
    if not m:
        metrics_placeholder.markdown("<div class='action-text'>No data yet</div>", unsafe_allow_html=True)
        return
    rows = "".join(
        f'<div class="stat-row"><span class="stat-key">{k}</span>'
        f'<span class="stat-val">{v}</span></div>'
        for k, v in m.items()
    )
    metrics_placeholder.markdown(rows, unsafe_allow_html=True)


def trigger_alarm():
    """Use JS Audio API — bypasses browser autoplay policy."""
    if alarm_b64 is None:
        return
    now = time.time()
    if now - st.session_state.last_alarm_ts > 3.0:
        st.session_state.last_alarm_ts = now
        alarm_placeholder.markdown(
            f"""
            <script>
            (function() {{
                var a = new Audio("data:audio/wav;base64,{alarm_b64}");
                a.volume = 1.0;
                a.play().catch(function(e) {{ console.warn("Alarm play error:", e); }});
            }})();
            </script>
            """,
            unsafe_allow_html=True,
        )


def clear_alarm():
    alarm_placeholder.empty()
    st.session_state.alarm_active = False


# ------------------------------------------------------------------
# WebRTC streamer
# ------------------------------------------------------------------
with video_col:
    st.markdown('<div class="panel"><div class="panel-title">Live Feed</div>', unsafe_allow_html=True)
    if not os.path.exists(MODEL_PATH):
        st.error("face_landmarker.task model file not found.")
        ctx = None
    else:
        ctx = webrtc_streamer(
            key="drowsiness-cam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=DrowsinessVideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
                "audio": False,
            },
            async_processing=False,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Score Trend</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Push slider values into the processor (UI thread)
# ------------------------------------------------------------------
if ctx and ctx.video_processor:
    p = ctx.video_processor
    p.ear_threshold   = ear_threshold
    p.mar_threshold   = mar_threshold
    p.pitch_thresh    = pitch_thresh
    p.yaw_thresh      = yaw_thresh
    p.alert_thresh    = alert_thresh
    p.drowsy_thresh   = drowsy_thresh
    p.process_every_n = process_every_n
    p.show_points     = show_points

# ------------------------------------------------------------------
# Dashboard refresh — 1.5 s interval (gentler on server)
# ------------------------------------------------------------------
if ctx and ctx.state.playing:
    st_autorefresh(interval=1500, key="dashboard_refresh")

    if ctx.video_processor:
        with ctx.video_processor.lock:
            score       = ctx.video_processor.score
            level       = ctx.video_processor.level
            action_text = ctx.video_processor.action_text
            metrics     = ctx.video_processor.metrics.copy()

        st.session_state.score_history.append(score)
        render_gauge(level, score, action_text)
        render_metrics(metrics)

        if st.session_state.score_history:
            chart_placeholder.line_chart(list(st.session_state.score_history), height=160)

        if level == "CRITICAL":
            trigger_alarm()
            st.session_state.alarm_active = True
        elif st.session_state.alarm_active:
            clear_alarm()
else:
    render_gauge("SAFE", 0, "Camera is idle — click START above")
    render_metrics({})
    clear_alarm()
