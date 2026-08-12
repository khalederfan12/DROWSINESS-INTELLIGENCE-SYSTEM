import cv2
import numpy as np

class HeadPoseEstimator:
    """
    Module for Head Pose Estimation (Pitch, Yaw, Roll) using OpenCV solvePnP.
    """

    # 2D Landmark Indices in MediaPipe Face Mesh
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    # 3D Model Generic Facial Landmarks (in millimeters)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    def __init__(self, pitch_down_thresh=-18.0, yaw_thresh=22.0, roll_thresh=20.0):
        """
        :param pitch_down_thresh: Angle in degrees below which head is considered dropping down (sleeping).
        :param yaw_thresh: Angle in degrees beyond which head is turned left/right (distraction).
        :param roll_thresh: Angle in degrees beyond which head is tilted left/right.
        """
        self.pitch_down_thresh = pitch_down_thresh
        self.yaw_thresh = yaw_thresh
        self.roll_thresh = roll_thresh

    def estimate_pose(self, landmarks, frame_w, frame_h):
        """
        Estimates Pitch, Yaw, Roll from 2D landmarks.
        :param landmarks: MediaPipe facial landmarks list.
        :param frame_w: Image width.
        :param frame_h: Image height.
        :return: Dict containing pitch, yaw, roll, and flags (is_head_down, is_looking_away, pose_status).
        """
        if not landmarks or len(landmarks) < 468:
            return {
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "is_head_down": False, "is_looking_away": False,
                "pose_status": "No Face"
            }

        # Extract 2D points
        def get_2d(idx):
            lm = landmarks[idx]
            return (lm.x * frame_w, lm.y * frame_h)

        image_points_2d = np.array([
            get_2d(self.NOSE_TIP),
            get_2d(self.CHIN),
            get_2d(self.LEFT_EYE_CORNER),
            get_2d(self.RIGHT_EYE_CORNER),
            get_2d(self.LEFT_MOUTH_CORNER),
            get_2d(self.RIGHT_MOUTH_CORNER)
        ], dtype=np.float64)

        # Approximate Camera Intrinsic Matrix
        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion Coefficients (Assuming zero lens distortion)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point (PnP)
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "is_head_down": False, "is_looking_away": False,
                "pose_status": "Error"
            }

        # Convert Rotation Vector to Rotation Matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose Rotation Matrix to get Euler Angles (Pitch, Yaw, Roll)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # RQDecomp3x3 returns angles in degrees
        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # Evaluate Posture Anomalies
        is_head_down = pitch <= self.pitch_down_thresh
        is_looking_away = abs(yaw) >= self.yaw_thresh
        is_head_tilted = abs(roll) >= self.roll_thresh

        if is_head_down:
            pose_status = "Head Drop (Sleeping)"
        elif is_looking_away:
            pose_status = "Distracted (Looking Away)"
        elif is_head_tilted:
            pose_status = "Head Tilted"
        else:
            pose_status = "Forward (Focused)"

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "is_head_down": is_head_down,
            "is_looking_away": is_looking_away,
            "is_head_tilted": is_head_tilted,
            "pose_status": pose_status
        }
