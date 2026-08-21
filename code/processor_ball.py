import cv2
import numpy as np
import os
from processor_base import ProcessorBase

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKER_VERSION = "ball-tracker-v4-counting-enabled"


def _first_existing_path(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0] if paths else None


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class BallProcessor(ProcessorBase):
    """
    Automatic MVP ball counter for wider table-tennis videos.

    The old implementation counted the largest moving contour crossing the
    center of the frame. This version first estimates the table region, filters
    small ball-like candidates inside/near that region, tracks the selected
    candidate with a Kalman filter, then counts crossings around the estimated
    net position.
    """

    def __init__(
        self,
        frame_w=640,
        frame_h=480,
        net_buffer=8,
        table_update_interval=30,
        count_cooldown_frames=10,
        max_missing_predict_frames=None,
        use_yolo=True,
        yolo_model_path=None,
        yolo_conf=None,
        yolo_every_n_frames=None,
        yolo_imgsz=None,
        yolo_device=None,
        detection_only=None,
    ):
        self.FRAME_WIDTH = frame_w
        self.FRAME_HEIGHT = frame_h
        self.NET_BUFFER = net_buffer
        self.size = (self.FRAME_WIDTH, self.FRAME_HEIGHT)

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=180,
            varThreshold=35,
            detectShadows=False,
        )

        self.ball_count = 0
        self.prev_ball_pos = None
        self.ball_direction = 0
        self.last_ball_side = None
        self.last_stable_ball_side = None
        self.net_x_position = self.FRAME_WIDTH // 2

        self.frame_index = 0
        self.table_update_interval = table_update_interval
        self.count_cooldown_frames = count_cooldown_frames
        self.last_count_frame = -count_cooldown_frames
        self.missing_frames = 0
        self.max_missing_predict_frames = (
            max_missing_predict_frames
            if max_missing_predict_frames is not None
            else _env_int("BALL_MAX_MISSING_PREDICT_FRAMES", 15)
        )
        self.tracking_gate_px = _env_int("BALL_TRACKING_GATE_PX", 90)
        self.reacquire_gate_px = _env_int("BALL_REACQUIRE_GATE_PX", 170)
        self.count_side_margin_px = _env_int("BALL_COUNT_SIDE_MARGIN_PX", 24)
        self.prev_gray = None

        self.table_bbox = None  # (x, y, w, h)
        self.table_mask = np.ones((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.uint8) * 255
        self.last_strict_mask = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.uint8)

        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)

        self.kalman = self._create_kalman_filter()
        self.kalman_initialized = False

        self.use_yolo = use_yolo
        self.detection_only = (
            detection_only
            if detection_only is not None
            else _env_bool("BALL_DETECTION_ONLY", False)
        )
        self.count_original_size = _env_bool("BALL_COUNT_ORIGINAL_SIZE", True)
        self.detection_only_original_size = _env_bool("BALL_DETECTION_ONLY_ORIGINAL_SIZE", True)
        self.yolo_conf = _env_float("BALL_YOLO_CONF", 0.20 if yolo_conf is None else yolo_conf)
        self.yolo_probe_conf = _env_float("BALL_YOLO_PROBE_CONF", -1.0)
        self.yolo_acquire_conf = _env_float("BALL_YOLO_ACQUIRE_CONF", self.yolo_conf)
        self.allow_opencv_fallback = _env_bool("BALL_ALLOW_OPENCV_FALLBACK", False)
        self.yolo_every_n_frames = max(
            1,
            yolo_every_n_frames
            if yolo_every_n_frames is not None
            else _env_int("BALL_YOLO_EVERY_N_FRAMES", 1),
        )
        self.yolo_imgsz = max(128, yolo_imgsz if yolo_imgsz is not None else _env_int("BALL_YOLO_IMGSZ", 1280))
        self.yolo_min_area_ratio = _env_float("BALL_YOLO_MIN_AREA_RATIO", 0.000003)
        self.yolo_max_area_ratio = _env_float("BALL_YOLO_MAX_AREA_RATIO", 0.005)
        self.yolo_max_side_ratio = _env_float("BALL_YOLO_MAX_SIDE_RATIO", 0.12)
        self.yolo_device = yolo_device or os.getenv("BALL_YOLO_DEVICE")
        default_yolo_model_path = _first_existing_path(
            os.path.join(PROJECT_ROOT, "backend", "models", "ver6.pt"),
            os.path.join(PROJECT_ROOT, "backend", "models", "ball_yolo.pt"),
        )
        self.yolo_model_path = yolo_model_path or os.getenv(
            "BALL_YOLO_MODEL_PATH",
            default_yolo_model_path,
        )
        self.yolo_model = None
        self.yolo_available = False
        self.yolo_status = "off"
        self.last_yolo_bbox = None
        self._load_yolo_model()

    def _reset_tracking_state(self):
        self.prev_ball_pos = None
        self.ball_direction = 0
        self.last_ball_side = None
        self.last_stable_ball_side = None
        self.missing_frames = 0
        self.prev_gray = None
        self.table_bbox = None
        self.last_yolo_bbox = None
        self.kalman = self._create_kalman_filter()
        self.kalman_initialized = False

    def _sync_frame_geometry(self, frame):
        height, width = frame.shape[:2]
        if width == self.FRAME_WIDTH and height == self.FRAME_HEIGHT:
            return

        self.FRAME_WIDTH = width
        self.FRAME_HEIGHT = height
        self.size = (self.FRAME_WIDTH, self.FRAME_HEIGHT)
        self.net_x_position = self.FRAME_WIDTH // 2
        self.table_mask = np.ones((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.uint8) * 255
        self.last_strict_mask = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.uint8)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=180,
            varThreshold=35,
            detectShadows=False,
        )
        self._reset_tracking_state()

    def _detect_yolo_raw(self, frame, conf=None):
        if not self.yolo_available or self.yolo_model is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        try:
            predict_kwargs = {
                "conf": self.yolo_conf if conf is None else conf,
                "imgsz": self.yolo_imgsz,
                "verbose": False,
            }
            if self.yolo_device:
                predict_kwargs["device"] = self.yolo_device
            results = self.yolo_model.predict(frame, **predict_kwargs)
        except Exception as exc:
            print(f"YOLO prediction failed: {exc}")
            return []

        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].detach().cpu().numpy()]
                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w - 1, x2))
                y2 = max(0, min(frame_h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
                detections.append(
                    {
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "conf": conf,
                    }
                )
        detections.sort(key=lambda item: item["conf"], reverse=True)
        return detections

    def _process_detection_only(self, frame_bgr):
        frame = frame_bgr.copy() if self.detection_only_original_size else cv2.resize(frame_bgr, self.size)
        self.frame_index += 1
        detections = self._detect_yolo_raw(frame)
        probe_detections = []
        if not detections and self.yolo_probe_conf >= 0:
            probe_detections = self._detect_yolo_raw(frame, conf=self.yolo_probe_conf)

        for index, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            color = (0, 255, 0) if index == 0 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        return frame, {
            "count": 0,
            "detections": detections,
            "probe_detections": probe_detections,
            "detection_only": True,
            "yolo_available": self.yolo_available,
            "yolo_status": self.yolo_status,
        }

    def _load_yolo_model(self):
        if not self.use_yolo:
            self.yolo_status = "disabled"
            return
        if not self.yolo_model_path or not os.path.exists(self.yolo_model_path):
            self.yolo_status = "missing-model"
            return
        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_available = True
            self.yolo_status = "on"
        except Exception as exc:
            print(f"YOLO ball detector is unavailable, falling back to OpenCV: {exc}")
            self.yolo_model = None
            self.yolo_available = False
            self.yolo_status = "load-failed"

    def _create_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.18
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.35
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        return kalman

    def _detect_table_region(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green/blue table surfaces in HSV. These ranges are intentionally broad
        # for MVP use across phones, venues, and lighting conditions.
        green_mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([85, 35, 35]), np.array([135, 255, 255]))
        mask = cv2.bitwise_or(green_mask, blue_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_medium, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        min_area = self.FRAME_WIDTH * self.FRAME_HEIGHT * 0.04
        candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not candidates:
            return

        contour = max(candidates, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        pad_x = int(w * 0.20)
        pad_y_top = int(h * 0.60)
        pad_y_bottom = int(h * 0.32)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y_top)
        x2 = min(self.FRAME_WIDTH, x + w + pad_x)
        y2 = min(self.FRAME_HEIGHT, y + h + pad_y_bottom)

        self.table_bbox = (x1, y1, x2 - x1, y2 - y1)
        self.net_x_position = x1 + (x2 - x1) // 2

        # The table estimate is useful for the net line, but it is unreliable
        # as a hard detection mask in wide/low-angle videos. Keep tracking in
        # the full frame so a valid ball is not discarded outside table ROI.
        self.table_mask = np.ones((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.uint8) * 255

    def _build_ball_candidate_mask(self, frame, fgmask, predicted_point):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        white_mask = cv2.inRange(hsv, np.array([0, 0, 115]), np.array([179, 130, 255]))
        orange_mask = cv2.inRange(hsv, np.array([3, 45, 75]), np.array([35, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 35, 90]), np.array([50, 255, 255]))
        bright_mask = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
        color_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, orange_mask), yellow_mask)
        color_mask = cv2.bitwise_or(color_mask, bright_mask)

        mog_motion = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)[1]
        frame_motion = np.zeros_like(mog_motion)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            frame_motion = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)[1]
            frame_motion = cv2.dilate(frame_motion, self.kernel_small, iterations=1)
        self.prev_gray = gray

        motion_mask = cv2.bitwise_or(mog_motion, frame_motion)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.kernel_small)

        # A real ball is usually both moving and bright/orange. Motion-only
        # evidence is used only near the Kalman prediction to avoid following
        # the player, paddle, or table edges.
        strict_mask = cv2.bitwise_and(motion_mask, color_mask)
        self.last_strict_mask = cv2.bitwise_and(strict_mask, self.table_mask)
        candidate_mask = strict_mask
        if self.missing_frames > self.max_missing_predict_frames:
            candidate_mask = cv2.bitwise_or(candidate_mask, color_mask)
        if predicted_point is not None and self.kalman_initialized:
            gate = self.reacquire_gate_px if self.missing_frames > 3 else self.tracking_gate_px
            search_mask = np.zeros_like(motion_mask)
            cv2.circle(search_mask, predicted_point, gate, 255, -1)
            near_prediction = cv2.bitwise_and(motion_mask, search_mask)
            candidate_mask = cv2.bitwise_or(candidate_mask, near_prediction)
        candidate_mask = cv2.bitwise_and(candidate_mask, self.table_mask)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, self.kernel_small)
        return candidate_mask, color_mask

    def _candidate_score(self, contour, color_mask, predicted_point):
        area = cv2.contourArea(contour)
        max_area = 1800 if self.missing_frames > self.max_missing_predict_frames else 650
        if area < 2 or area > max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return None

        aspect = w / float(h)
        max_aspect = 4.0 if self.missing_frames > self.max_missing_predict_frames else 2.6
        if aspect < 0.25 or aspect > max_aspect:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        min_circularity = 0.08 if self.missing_frames > self.max_missing_predict_frames else 0.18
        if circularity < min_circularity:
            return None

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        roi = color_mask[y : y + h, x : x + w]
        color_ratio = cv2.countNonZero(roi) / float(max(1, w * h))

        dist = None
        if (
            predicted_point is not None
            and self.kalman_initialized
            and self.missing_frames <= self.max_missing_predict_frames
        ):
            dist = np.linalg.norm(np.array(center) - np.array(predicted_point))
            gate = self.reacquire_gate_px if self.missing_frames > 3 else self.tracking_gate_px
            if dist > gate:
                return None

        if color_ratio < 0.14 and dist is None:
            return None
        if color_ratio < 0.03 and dist is not None:
            return None

        score = area * 0.08 + circularity * 12 + color_ratio * 8
        if dist is not None:
            score += max(0, 90 - dist) * 0.35

        return score, center, (x, y, w, h)

    def _select_ball_candidate(self, candidate_mask, color_mask, predicted_point):
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            candidate = self._candidate_score(contour, color_mask, predicted_point)
            if candidate is None:
                continue
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None, None
        _, center, bbox = best
        return center, bbox

    def _detect_ball_yolo(self, frame, predicted_point):
        if not self.yolo_available or self.yolo_model is None:
            return None, None
        if self.frame_index % self.yolo_every_n_frames != 0 and self.last_yolo_bbox is not None:
            return None, None

        inference_frame = frame
        offset_x = 0
        offset_y = 0
        mask_points = cv2.findNonZero(self.table_mask)
        if mask_points is not None:
            x, y, w, h = cv2.boundingRect(mask_points)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(self.FRAME_WIDTH, x + w)
            y2 = min(self.FRAME_HEIGHT, y + h)
            if x2 > x1 and y2 > y1:
                inference_frame = frame[y1:y2, x1:x2]
                offset_x = x1
                offset_y = y1

        try:
            predict_kwargs = {
                "conf": self.yolo_conf,
                "imgsz": self.yolo_imgsz,
                "verbose": False,
            }
            if self.yolo_device:
                predict_kwargs["device"] = self.yolo_device
            results = self.yolo_model.predict(inference_frame, **predict_kwargs)
        except Exception as exc:
            print(f"YOLO prediction failed, falling back to OpenCV for this frame: {exc}")
            return None, None

        best = None
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1 += offset_x
                x2 += offset_x
                y1 += offset_y
                y2 += offset_y
                x1 = max(0, min(self.FRAME_WIDTH - 1, x1))
                y1 = max(0, min(self.FRAME_HEIGHT - 1, y1))
                x2 = max(0, min(self.FRAME_WIDTH - 1, x2))
                y2 = max(0, min(self.FRAME_HEIGHT - 1, y2))
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue

                area = w * h
                frame_area = max(1, self.FRAME_WIDTH * self.FRAME_HEIGHT)
                min_area = max(3, int(frame_area * self.yolo_min_area_ratio))
                max_area = max(900, int(frame_area * self.yolo_max_area_ratio))
                max_side = max(24, int(min(self.FRAME_WIDTH, self.FRAME_HEIGHT) * self.yolo_max_side_ratio))
                if area < min_area or area > max_area:
                    continue
                if max(w, h) > max_side:
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if self.table_mask[center[1], center[0]] == 0:
                    continue

                conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
                if (
                    (not self.kalman_initialized or self.missing_frames > self.max_missing_predict_frames)
                    and conf < self.yolo_acquire_conf
                ):
                    continue
                aspect = w / float(h)
                aspect_penalty = abs(1.0 - aspect)
                score = conf * 100 - aspect_penalty * 10
                if predicted_point is not None and self.missing_frames <= self.max_missing_predict_frames:
                    dist = np.linalg.norm(np.array(center) - np.array(predicted_point))
                    gate = self.reacquire_gate_px if self.missing_frames > 3 else self.tracking_gate_px
                    if self.kalman_initialized and dist > gate:
                        continue
                    score += max(0, gate - dist) * 0.25

                if best is None or score > best[0]:
                    best = (score, center, (x1, y1, w, h), (x1, y1, x2, y2))

        if best is None:
            return None, None

        _, center, bbox, xyxy = best
        self.last_yolo_bbox = xyxy
        return center, bbox

    def _update_tracker(self, measurement):
        prediction = self.kalman.predict()
        predicted_point = (int(prediction[0][0]), int(prediction[1][0]))

        if measurement is None:
            self.missing_frames += 1
            if self.kalman_initialized and self.missing_frames <= self.max_missing_predict_frames:
                return predicted_point, False
            return None, False

        mx, my = measurement
        measured = np.array([[np.float32(mx)], [np.float32(my)]])
        should_reinitialize = (
            not self.kalman_initialized
            or self.missing_frames > self.max_missing_predict_frames
        )
        if should_reinitialize:
            self.kalman.statePost = np.array(
                [[np.float32(mx)], [np.float32(my)], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.kalman_initialized = True
        corrected = self.kalman.correct(measured)
        self.missing_frames = 0
        return (int(corrected[0][0]), int(corrected[1][0])), True

    def _update_count(self, point):
        if point is None:
            return

        curr_x = point[0]
        side_margin = max(self.NET_BUFFER, self.count_side_margin_px, int(self.FRAME_WIDTH * 0.015))
        left_line = self.net_x_position - side_margin
        right_line = self.net_x_position + side_margin
        can_count = (self.frame_index - self.last_count_frame) >= self.count_cooldown_frames

        current_side = 0
        if curr_x < left_line:
            current_side = -1
        elif curr_x > right_line:
            current_side = 1

        # Do not count the first time the ball is seen. Establish the side
        # first, then count only when the ball is confirmed past the net on the
        # opposite side. A hit into the net that remains on the same side does
        # not change current_side, so it is not counted.
        if current_side != 0:
            if self.last_stable_ball_side is None:
                self.last_stable_ball_side = current_side
            elif current_side != self.last_stable_ball_side and can_count:
                self.ball_count += 1
                self.last_count_frame = self.frame_index
                self.ball_direction = current_side
                self.last_stable_ball_side = current_side
            elif current_side == self.last_stable_ball_side:
                self.ball_direction = current_side

            self.last_ball_side = current_side

        self.prev_ball_pos = point

    def _draw_debug(self, frame, fgmask, candidate_mask, point, measurement_bbox, detection_source):
        if measurement_bbox is not None:
            bx, by, bw, bh = measurement_bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        cv2.putText(frame, f"Count: {self.ball_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)

        return {
            "mask": fgmask,
            "candidate_mask": candidate_mask,
            "count": self.ball_count,
            "table_bbox": self.table_bbox,
            "net_x_position": self.net_x_position,
            "tracked_ball": point,
        }

    def process(self, frame_bgr):
        if self.detection_only:
            return self._process_detection_only(frame_bgr)

        frame = frame_bgr.copy() if self.count_original_size else cv2.resize(frame_bgr, self.size)
        self._sync_frame_geometry(frame)
        self.frame_index += 1

        if self.table_bbox is None or self.frame_index % self.table_update_interval == 1:
            self._detect_table_region(frame)

        prediction = None
        if self.kalman_initialized:
            state = self.kalman.statePost
            prediction = (int(state[0][0] + state[2][0]), int(state[1][0] + state[3][0]))
            if self.missing_frames > self.max_missing_predict_frames:
                prediction = None

        fgmask = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.uint8)
        candidate_mask = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH), dtype=np.uint8)
        color_mask = None

        measurement, measurement_bbox = self._detect_ball_yolo(frame, prediction)
        detection_source = "yolo" if measurement is not None else "none"
        use_opencv_fallback = not self.yolo_available or self.allow_opencv_fallback
        if measurement is None and use_opencv_fallback:
            fgmask = self.fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel_small)
            candidate_mask, color_mask = self._build_ball_candidate_mask(frame, fgmask, prediction)
            measurement, measurement_bbox = self._select_ball_candidate(candidate_mask, color_mask, prediction)
            if measurement is not None:
                detection_source = "opencv"
        tracked_point, has_measurement = self._update_tracker(measurement)
        if tracked_point is not None and has_measurement:
            self._update_count(tracked_point)

        side = self._draw_debug(frame, fgmask, candidate_mask, tracked_point, measurement_bbox, detection_source)
        side["detection_source"] = detection_source
        side["yolo_available"] = self.yolo_available
        side["yolo_status"] = self.yolo_status
        return frame, side
