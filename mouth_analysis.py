import math
import time

class MouthAnalyzer:
    """
    Module for Yawning Detection using Mouth Aspect Ratio (MAR).
    """

    # Inner Mouth Landmark Indices for MediaPipe FaceMesh / FaceLandmarker (478 points)
    # Corners
    MOUTH_CORNER_LEFT = 61
    MOUTH_CORNER_RIGHT = 291

    # Vertical Inner Lip Pairs (Upper, Lower)
    INNER_LIP_CENTER_TOP = 13
    INNER_LIP_CENTER_BOTTOM = 14

    INNER_LIP_LEFT_TOP = 81
    INNER_LIP_LEFT_BOTTOM = 178

    INNER_LIP_RIGHT_TOP = 311
    INNER_LIP_RIGHT_BOTTOM = 402

    def __init__(self, mar_threshold=0.55, yawn_time_threshold=1.5):
        """
        :param mar_threshold: MAR threshold above which mouth is considered open/yawning.
        :param yawn_time_threshold: Duration in seconds mouth must stay open to register a yawn.
        """
        self.mar_threshold = mar_threshold
        self.yawn_time_threshold = yawn_time_threshold
        
        self.yawn_start_time = None
        self.is_yawning = False
        self.current_yawn_duration = 0.0
        self.total_yawn_count = 0
        self._was_yawning_last_frame = False

    @staticmethod
    def _euclidean_distance(p1, p2):
        """Calculates 2D Euclidean distance between two points (x, y)."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_mar(self, landmarks, frame_w, frame_h):
        """
        Calculates Mouth Aspect Ratio (MAR) from MediaPipe landmarks.
        :param landmarks: List of facial landmarks from MediaPipe.
        :param frame_w: Frame width in pixels.
        :param frame_h: Frame height in pixels.
        :return: Computed MAR float value.
        """
        if not landmarks or len(landmarks) < 468:
            return 0.0

        def get_pt(idx):
            lm = landmarks[idx]
            return (lm.x * frame_w, lm.y * frame_h)

        c_left = get_pt(self.MOUTH_CORNER_LEFT)
        c_right = get_pt(self.MOUTH_CORNER_RIGHT)

        v1_top = get_pt(self.INNER_LIP_CENTER_TOP)
        v1_bot = get_pt(self.INNER_LIP_CENTER_BOTTOM)

        v2_top = get_pt(self.INNER_LIP_LEFT_TOP)
        v2_bot = get_pt(self.INNER_LIP_LEFT_BOTTOM)

        v3_top = get_pt(self.INNER_LIP_RIGHT_TOP)
        v3_bot = get_pt(self.INNER_LIP_RIGHT_BOTTOM)

        # Vertical distances
        d_vert1 = self._euclidean_distance(v1_top, v1_bot)
        d_vert2 = self._euclidean_distance(v2_top, v2_bot)
        d_vert3 = self._euclidean_distance(v3_top, v3_bot)

        # Horizontal distance
        d_horiz = self._euclidean_distance(c_left, c_right)

        if d_horiz == 0:
            return 0.0

        mar = (d_vert1 + d_vert2 + d_vert3) / (2.0 * d_horiz)
        return mar

    def update(self, landmarks, frame_w, frame_h):
        """
        Updates mouth state per frame.
        :return: Dictionary containing MAR, yawn status, duration, and count.
        """
        mar = self.calculate_mar(landmarks, frame_w, frame_h)
        current_time = time.time()

        if mar >= self.mar_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time

            self.current_yawn_duration = current_time - self.yawn_start_time

            if self.current_yawn_duration >= self.yawn_time_threshold:
                self.is_yawning = True
                if not self._was_yawning_last_frame:
                    self.total_yawn_count += 1
                    self._was_yawning_last_frame = True
        else:
            self.yawn_start_time = None
            self.current_yawn_duration = 0.0
            self.is_yawning = False
            self._was_yawning_last_frame = False

        return {
            "mar": mar,
            "is_yawning": self.is_yawning,
            "yawn_duration": self.current_yawn_duration,
            "total_yawns": self.total_yawn_count
        }
