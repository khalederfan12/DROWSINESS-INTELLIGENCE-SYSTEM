class DrowsinessIntelligenceEngine:
    """
    Multimodal Intelligence Engine for Drowsiness Detection & Score Calculation.    
    Fuses:
    - Eye Analysis (EAR & Eye Closure Duration)
    - Yawn Analysis (MAR & Yawn Duration/Count)
    - Head Pose (Pitch, Yaw, Roll & Head Drop/Distraction)
    """

    # Status Constants
    STATUS_ALERT = "Alert "
    STATUS_WARNING = "Warning "
    STATUS_DROWSY = "Drowsy "

    def __init__(self, alert_thresh=35.0, drowsy_thresh=70.0):
        """
        :param alert_thresh: Score threshold below which driver is ALERT.
        :param drowsy_thresh: Score threshold above which driver is DROWSY.
        """
        self.alert_thresh = alert_thresh
        self.drowsy_thresh = drowsy_thresh

    def compute_drowsiness_score(self, eye_data, mouth_data, head_pose_data):
        """
        Computes weighted Drowsiness Score (0 to 100) and determines driver status.
        
        :param eye_data: Dict with keys: 'ear', 'is_closed', 'closed_duration'
        :param mouth_data: Dict with keys: 'mar', 'is_yawning', 'yawn_duration', 'total_yawns'
        :param head_pose_data: Dict with keys: 'pitch', 'yaw', 'roll', 'is_head_down', 'is_looking_away'
        :return: Dict with score, status, category scores, and recommended action.
        """
        # 1. Eye Component Score (Max 50 points)
        eye_score = 0.0
        closed_duration = eye_data.get("closed_duration", 0.0)
        ear = eye_data.get("ear", 0.3)

        if closed_duration >= 2.0:
            eye_score = 50.0
        elif closed_duration >= 1.0:
            eye_score = 35.0 + (closed_duration - 1.0) * 15.0  # 35 to 50
        elif closed_duration >= 0.4:
            eye_score = 20.0 + (closed_duration - 0.4) * 25.0  # 20 to 35
        elif ear < 0.20:
            eye_score = 15.0  # Low EAR / partial closure

        # 2. Yawn Component Score (Max 30 points)
        yawn_score = 0.0
        is_yawning = mouth_data.get("is_yawning", False)
        yawn_duration = mouth_data.get("yawn_duration", 0.0)
        total_yawns = mouth_data.get("total_yawns", 0)

        if is_yawning:
            if yawn_duration >= 2.0:
                yawn_score += 20.0
            else:
                yawn_score += 12.0

        # Add points for cumulative yawning history
        if total_yawns >= 3:
            yawn_score += 10.0
        elif total_yawns >= 1:
            yawn_score += 5.0

        yawn_score = min(yawn_score, 30.0)

        # 3. Head Pose Component Score (Max 20 points)
        head_score = 0.0
        is_head_down = head_pose_data.get("is_head_down", False)
        is_looking_away = head_pose_data.get("is_looking_away", False)

        if is_head_down:
            head_score += 20.0  # Head drop is a critical sleep indicator
        elif is_looking_away:
            head_score += 12.0  # Distraction indicator

        head_score = min(head_score, 20.0)

        # Total Fusion Score (0 - 100)
        total_score = min(eye_score + yawn_score + head_score, 100.0)

        # Status Classification
        if total_score >= self.drowsy_thresh:
            status = self.STATUS_DROWSY
            level = "CRITICAL"
            action = "PULL OVER & REST IMMEDIATELY! "
        elif total_score >= self.alert_thresh:
            status = self.STATUS_WARNING
            level = "WARNING"
            action = "EARLY DROWSINESS DETECTED: TAKE A BREAK "
        else:
            status = self.STATUS_ALERT
            level = "SAFE"
            action = "DRIVER IS ALERT & FOCUSED "

        return {
            "drowsiness_score": round(total_score, 1),
            "status": status,
            "level": level,
            "action": action,
            "breakdown": {
                "eye_score": round(eye_score, 1),
                "yawn_score": round(yawn_score, 1),
                "head_score": round(head_score, 1)
            }
        }
