# fatigue_detection/fatigue_detector.py

import cv2
import mediapipe as mp
from .landmarks_utils import landmarks_to_points, calculate_ear

class FatigueDetector:
    # Eye landmark indices from MediaPipe FaceMesh
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    def __init__(self, threshold=0.25, fatigue_frames=20):
        self.ear_threshold = threshold
        self.fatigue_frame_threshold = fatigue_frames
        self.fatigue_counter = 0
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        Main method to process each frame and detect fatigue.
        Returns: updated frame, is_fatigued (bool), EAR (float)
        """
        image_h, image_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        ear = None
        fatigue_flag = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get EAR for both eyes
                left_eye_points = landmarks_to_points(face_landmarks.landmark, self.LEFT_EYE, image_w, image_h)
                right_eye_points = landmarks_to_points(face_landmarks.landmark, self.RIGHT_EYE, image_w, image_h)

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)

                ear = (left_ear + right_ear) / 2.0

                # Draw EAR value on frame
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Fatigue logic
                if ear < self.ear_threshold:
                    self.fatigue_counter += 1
                    if self.fatigue_counter >= self.fatigue_frame_threshold:
                        fatigue_flag = True
                        cv2.putText(frame, "😴 Fatigue Detected!", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    self.fatigue_counter = 0

        return frame, fatigue_flag, ear
