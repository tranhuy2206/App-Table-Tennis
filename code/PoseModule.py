import cv2
import mediapipe as mp
import math


class poseDetector:

    def __init__(self, mode=False, modelComplexity=1, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.modelComplexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            connection_style = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=2)
            node_style = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=node_style,
                                       connection_drawing_spec=connection_style)
        return img

    def findPosition(self, img, draw=True, use_3d=False):
        self.lmList = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            
            if use_3d and self.results.pose_world_landmarks:
                # Sử dụng pose_world_landmarks cho tọa độ 3D chính xác (metric coordinates)
                # pose_world_landmarks trả về tọa độ trong world space (meters)
                world_landmarks = self.results.pose_world_landmarks.landmark
                for id, (lm, wlm) in enumerate(zip(self.results.pose_landmarks.landmark, world_landmarks)):
                    # Lấy x, y từ pose_landmarks (để vẽ trên image)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Lấy x, y, z từ pose_world_landmarks (metric coordinates, chính xác hơn)
                    wx, wy, wz = float(wlm.x), float(wlm.y), float(wlm.z)
                    visibility = float(getattr(lm, "visibility", 1.0))
                    presence = float(getattr(lm, "presence", 1.0))
                    # Keep the historical coordinate positions intact and append
                    # confidence metadata for consumers that can use it.
                    self.lmList.append([id, cx, cy, wx, wy, wz, visibility, presence])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            elif use_3d:
                # Rare fallback when world landmarks are unavailable. Preserve
                # the 3D-compatible layout using normalized image coordinates.
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    visibility = float(getattr(lm, "visibility", 1.0))
                    presence = float(getattr(lm, "presence", 1.0))
                    self.lmList.append([
                        id, cx, cy, float(lm.x), float(lm.y), float(lm.z),
                        visibility, presence,
                    ])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            else:
                # Mode 2D: chỉ dùng pose_landmarks
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    visibility = float(getattr(lm, "visibility", 1.0))
                    presence = float(getattr(lm, "presence", 1.0))
                    self.lmList.append([id, cx, cy, visibility, presence])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def close(self):
        """Release MediaPipe native resources explicitly."""
        if getattr(self, "pose", None) is not None:
            self.pose.close()

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle



