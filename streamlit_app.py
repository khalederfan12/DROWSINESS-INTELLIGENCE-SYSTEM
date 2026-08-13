import os
import time
import base64
# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")
EYE_COLOR   = (255, 210, 60)
MOUTH_COLOR = (60, 210, 255)
# Processing resolution — downscale before ML inference to save RAM & CPU
PROC_WIDTH  = 320
PROC_HEIGHT = 240
# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
    .gauge {
        width: 168px; height: 168px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        position: relative;
    }
    .gauge-inner {
        width: 128px; height: 128px; border-radius: 50%;
    .stat-row:last-child { border-bottom: none; }
    .stat-key { color: #8a90a0; }
    .stat-val { color: #f2f3f5; font-weight: 700; font-variant-numeric: tabular-nums; }
    div.stButton > button {
        border-radius: 10px; border: 1px solid #2b2f3a; font-weight: 700;
    }
    div.stButton > button { border-radius: 10px; border: 1px solid #2b2f3a; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)
# ------------------------------------------------------------------
# Resources
