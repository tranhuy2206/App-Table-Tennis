import cv2
import numpy as np
import os
import time
from processor_base import ProcessorBase

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKER_VERSION = "ball-tracker-v5-lightweight-crossing-validation"


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
        count_cooldown_frames=2,
        max_missing_predict_frames=None,
        use_yolo=True,
        yolo_model_path=None,
        yolo_conf=None,
        yolo_every_n_frames=None,
        yolo_imgsz=None,
        yolo_device=None,
        detection_only=None,
        debug_diagnostics=None,
    ):
        self.FRAME_WIDTH = frame_w
        self.FRAME_HEIGHT = frame_h
        self.NET_BUFFER = net_buffer
        self.size = (self.FRAME_WIDTH, self.FRAME_HEIGHT)
        # Ball crossing is frame-sensitive. The desktop controller must not
        # discard source frames with cap.grab(), otherwise a whole crossing
        # can occur between two processed frames.
        self.preserve_all_frames = True

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

        # These checks are intentionally lightweight. Velocity rejection is
        # opt-in because a genuine fast smash can otherwise be mistaken for a
        # detector jump and disappear before Kalman correction.
        self.enable_velocity_filter = _env_bool("BALL_ENABLE_VELOCITY_FILTER", False)
        self.max_measurement_speed_ratio = max(
            0.01,
            _env_float("BALL_MAX_MEASUREMENT_SPEED_RATIO", 0.75),
        )
        self.min_crossing_observations = max(
            1,
            _env_int("BALL_MIN_CROSSING_OBSERVATIONS", 1),
        )
        self.max_crossing_frames = max(
            self.min_crossing_observations,
            _env_int("BALL_MAX_CROSSING_FRAMES", 15),
        )
        self.count_predict_frames = max(
            0,
            _env_int("BALL_COUNT_PREDICT_FRAMES", 3),
        )
        self.enable_crossing_validation = _env_bool("BALL_ENABLE_CROSSING_VALIDATION", True)
        self.pending_crossing = None
        self.last_measurement = None
        self.last_measurement_frame = None
        self.last_measurement_speed = None
        self.last_count_reason = "not-counted"
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
        self.count_original_size = _env_bool("BALL_COUNT_ORIGINAL_SIZE", False)
        self.detection_only_original_size = _env_bool("BALL_DETECTION_ONLY_ORIGINAL_SIZE", True)
        self.yolo_conf = _env_float("BALL_YOLO_CONF", 0.20 if yolo_conf is None else yolo_conf)
        self.yolo_probe_conf = _env_float("BALL_YOLO_PROBE_CONF", -1.0)
        self.yolo_acquire_conf = _env_float("BALL_YOLO_ACQUIRE_CONF", self.yolo_conf)
        self.allow_opencv_fallback = _env_bool("BALL_ALLOW_OPENCV_FALLBACK", False)
        self.yolo_every_n_frames = max(
            1,
            yolo_every_n_frames
            if yolo_every_n_frames is not None
            else _env_int("BALL_YOLO_EVERY_N_FRAMES", 2),
        )
        self.yolo_imgsz = max(128, yolo_imgsz if yolo_imgsz is not None else _env_int("BALL_YOLO_IMGSZ", 640))
        self.yolo_use_table_crop = _env_bool("BALL_YOLO_USE_TABLE_CROP", True)
        self.yolo_net_probe_px = max(
            32,
            _env_int("BALL_YOLO_NET_PROBE_PX", 140),
        )
        self.yolo_min_area_ratio = _env_float("BALL_YOLO_MIN_AREA_RATIO", 0.000003)
        self.yolo_max_area_ratio = _env_float("BALL_YOLO_MAX_AREA_RATIO", 0.005)
        self.yolo_max_side_ratio = _env_float("BALL_YOLO_MAX_SIDE_RATIO", 0.12)
        self.yolo_device = yolo_device or os.getenv("BALL_YOLO_DEVICE")
        default_yolo_model_path = _first_existing_path(
            os.path.join(PROJECT_ROOT, "backend", "models", "yolo11m.pt"),
            os.path.join(PROJECT_ROOT, "backend", "models", "ball_yolo.pt"),
        )
        self.yolo_model_path = yolo_model_path or os.getenv(
            "BALL_YOLO_MODEL_PATH",
            default_yolo_model_path,
        )
        self.yolo_model = None
        self.yolo_model_names = None
        self.yolo_available = False
        self.yolo_status = "off"
        self.last_yolo_bbox = None
        self.debug_diagnostics = (
            debug_diagnostics
            if debug_diagnostics is not None
            else _env_bool("BALL_DEBUG_DIAGNOSTICS", False)
        )
        self.debug_overlay = _env_bool(
            "BALL_DEBUG_OVERLAY",
            self.debug_diagnostics,
        )
        self.debug_log_interval = max(
            0,
            _env_int("BALL_DEBUG_LOG_INTERVAL", 100),
        )
        self.debug_probe_conf = max(
            0.001,
            _env_float("BALL_DEBUG_PROBE_CONF", 0.01),
        )
        self.debug_history_size = max(
            0,
            _env_int("BALL_DEBUG_HISTORY_SIZE", 200),
        )
        self._reset_diagnostics()
        self._load_yolo_model()
        if self.debug_diagnostics:
            print(
                f"[BALL-DIAG] model={self.yolo_model_path} "
                f"status={self.yolo_status} classes={self.yolo_model_names} "
                f"conf={self.yolo_conf} probe_conf={self.debug_probe_conf} "
                f"imgsz={self.yolo_imgsz}"
            )

    def _reset_tracking_state(self):
        self.prev_ball_pos = None
        self.ball_direction = 0
        self.last_ball_side = None
        self.last_stable_ball_side = None
        self.missing_frames = 0
        self.pending_crossing = None
        self.last_measurement = None
        self.last_measurement_frame = None
        self.last_measurement_speed = None
        self.last_count_reason = "not-counted"
        self.prev_gray = None
        self.table_bbox = None
        self.last_yolo_bbox = None
        self.kalman = self._create_kalman_filter()
        self.kalman_initialized = False

    def _reset_diagnostics(self):
        """Reset diagnostic counters for a new video or on first init."""
        self._diag = {
            "frames_processed": 0,
            "yolo_calls": 0,
            "yolo_skips": 0,
            "yolo_raw_boxes": 0,
            "yolo_rejected_conf": 0,
            "yolo_rejected_area": 0,
            "yolo_rejected_side": 0,
            "yolo_rejected_tracking_gate": 0,
            "yolo_rejected_aspect": 0,
            "yolo_accepted": 0,
            "yolo_probe_calls": 0,
            "yolo_probe_found": 0,
            "yolo_probe_boxes": 0,
            "yolo_probe_inference_ms": 0.0,
            "yolo_inference_ms": 0.0,
            "yolo_errors": 0,
            "opencv_accepted": 0,
            "tracker_predict_only": 0,
            "tracker_corrected": 0,
            "tracker_reinitialized": 0,
            "side_transitions": 0,
            "crossings_committed": 0,
            "crossings_cooldown_suppressed": 0,
            "crossings_jitter_cancelled": 0,
            "crossings_expired": 0,
            "velocity_rejected": 0,
            "gate_bypass_opposite": 0,
            "table_detected": False,
            "table_detection_attempts": 0,
            "table_detection_successes": 0,
            "process_ms": 0.0,
            "last_diag_print_frame": 0,
            "diag_start_time": time.perf_counter(),
        }
        self._diag_raw_classes = {}
        self._diag_probe_classes = {}
        self._diag_last_yolo = {
            "ran": False,
            "skip_reason": "not-run",
            "raw_boxes": 0,
            "probe_boxes": 0,
            "accepted": False,
            "inference_ms": 0.0,
            "inference_region": "full-frame",
            "boxes": [],
        }
        self._diag_frame_detail = []

    @staticmethod
    def _diag_increment(mapping, key):
        key = str(key)
        mapping[key] = mapping.get(key, 0) + 1

    def _diagnostic_model_name(self, result, class_id):
        names = getattr(result, "names", None)
        if isinstance(names, dict):
            return names.get(class_id, class_id)
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return names[class_id]
        return class_id

    def reset(self):
        """Reset all per-video state without reloading the YOLO model."""
        self.ball_count = 0
        self.frame_index = 0
        self.last_count_frame = -self.count_cooldown_frames
        self.net_x_position = self.FRAME_WIDTH // 2
        self.table_mask = np.ones(
            (self.FRAME_HEIGHT, self.FRAME_WIDTH),
            dtype=np.uint8,
        ) * 255
        self.last_strict_mask = np.zeros(
            (self.FRAME_HEIGHT, self.FRAME_WIDTH),
            dtype=np.uint8,
        )
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=180,
            varThreshold=35,
            detectShadows=False,
        )
        self._reset_tracking_state()
        self._reset_diagnostics()

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
            self.yolo_model_names = getattr(self.yolo_model, "names", None)
            self.yolo_available = True
            self.yolo_status = "on"
        except Exception as exc:
            print(f"YOLO ball detector is unavailable, falling back to OpenCV: {exc}")
            self.yolo_model = None
            self.yolo_model_names = None
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
        if self.debug_diagnostics:
            self._diag["table_detection_attempts"] += 1
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
        if self.debug_diagnostics:
            self._diag["table_detected"] = True
            self._diag["table_detection_successes"] += 1

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
        if self.debug_diagnostics:
            self._diag_last_yolo = {
                "ran": False,
                "skip_reason": "not-run",
                "raw_boxes": 0,
                "probe_boxes": 0,
                "accepted": False,
                "inference_ms": 0.0,
                "inference_region": "full-frame",
                "boxes": [],
            }
        if not self.yolo_available or self.yolo_model is None:
            if self.debug_diagnostics:
                self._diag_last_yolo["skip_reason"] = "model-unavailable"
            return None, None
        near_net = (
            predicted_point is not None
            and abs(predicted_point[0] - self.net_x_position) <= self.yolo_net_probe_px
        )
        needs_reacquire = self.missing_frames > 0
        if (
            self.frame_index % self.yolo_every_n_frames != 0
            and self.last_yolo_bbox is not None
            and not near_net
            and not needs_reacquire
        ):
            if self.debug_diagnostics:
                self._diag["yolo_skips"] += 1
                self._diag_last_yolo["skip_reason"] = "schedule"
            return None, None

        inference_frame = frame
        offset_x = 0
        offset_y = 0
        if self.yolo_use_table_crop and self.table_bbox is not None:
            x, y, w, h = self.table_bbox
            frame_area = max(1, self.FRAME_WIDTH * self.FRAME_HEIGHT)
            crop_area_ratio = (w * h) / float(frame_area)
            if 0.20 <= crop_area_ratio <= 0.95:
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(self.FRAME_WIDTH, x + w)
                y2 = min(self.FRAME_HEIGHT, y + h)
                if x2 > x1 and y2 > y1:
                    inference_frame = frame[y1:y2, x1:x2]
                    offset_x = x1
                    offset_y = y1
                    if self.debug_diagnostics:
                        self._diag_last_yolo["inference_region"] = [x1, y1, x2, y2]

        try:
            predict_kwargs = {
                "conf": min(self.yolo_conf, self.debug_probe_conf)
                if self.debug_diagnostics
                else self.yolo_conf,
                "imgsz": self.yolo_imgsz,
                "max_det": 3,
                "verbose": False,
            }
            if self.yolo_device:
                predict_kwargs["device"] = self.yolo_device
            inference_started = time.perf_counter()
            results = self.yolo_model.predict(inference_frame, **predict_kwargs)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            if self.debug_diagnostics:
                self._diag["yolo_calls"] += 1
                self._diag["yolo_inference_ms"] += inference_ms
                self._diag["yolo_probe_calls"] += 1
                self._diag["yolo_probe_inference_ms"] += inference_ms
                self._diag_last_yolo["ran"] = True
                self._diag_last_yolo["skip_reason"] = None
                self._diag_last_yolo["inference_ms"] = round(inference_ms, 2)
        except Exception as exc:
            if self.debug_diagnostics:
                self._diag["yolo_errors"] += 1
                self._diag_last_yolo["skip_reason"] = "prediction-error"
            print(f"YOLO prediction failed, falling back to OpenCV for this frame: {exc}")
            return None, None

        best = None
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
                class_id = int(box.cls[0].detach().cpu().item()) if getattr(box, "cls", None) is not None else -1
                class_name = self._diagnostic_model_name(result, class_id)
                box_diag = None
                if self.debug_diagnostics:
                    self._diag["yolo_probe_boxes"] += 1
                    self._diag_last_yolo["probe_boxes"] += 1
                    self._diag_increment(self._diag_probe_classes, class_name)
                    box_diag = {
                        "class_id": class_id,
                        "class_name": str(class_name),
                        "confidence": round(conf, 4),
                        "bbox": None,
                        "reject_reason": None,
                    }
                    self._diag_last_yolo["boxes"].append(box_diag)
                    if conf < self.yolo_conf:
                        self._diag["yolo_rejected_conf"] += 1
                        box_diag["reject_reason"] = "confidence"
                        continue
                    self._diag["yolo_raw_boxes"] += 1
                    self._diag_last_yolo["raw_boxes"] += 1
                    self._diag_increment(self._diag_raw_classes, class_name)
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
                if box_diag is not None:
                    box_diag["bbox"] = [x1, y1, x2, y2]
                if w <= 0 or h <= 0:
                    if self.debug_diagnostics:
                        self._diag["yolo_rejected_area"] += 1
                        box_diag["reject_reason"] = "invalid-area"
                    continue

                area = w * h
                frame_area = max(1, self.FRAME_WIDTH * self.FRAME_HEIGHT)
                min_area = max(3, int(frame_area * self.yolo_min_area_ratio))
                max_area = max(900, int(frame_area * self.yolo_max_area_ratio))
                max_side = max(24, int(min(self.FRAME_WIDTH, self.FRAME_HEIGHT) * self.yolo_max_side_ratio))
                if area < min_area or area > max_area:
                    if self.debug_diagnostics:
                        self._diag["yolo_rejected_area"] += 1
                        box_diag["reject_reason"] = "area"
                    continue
                if max(w, h) > max_side:
                    if self.debug_diagnostics:
                        self._diag["yolo_rejected_side"] += 1
                        box_diag["reject_reason"] = "side-length"
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if self.table_mask[center[1], center[0]] == 0:
                    if self.debug_diagnostics:
                        self._diag["yolo_rejected_area"] += 1
                        box_diag["reject_reason"] = "table-mask"
                    continue

                if (
                    (not self.kalman_initialized or self.missing_frames > self.max_missing_predict_frames)
                    and conf < self.yolo_acquire_conf
                ):
                    if self.debug_diagnostics:
                        self._diag["yolo_rejected_conf"] += 1
                        box_diag["reject_reason"] = "acquire-confidence"
                    continue
                aspect = w / float(h)
                aspect_penalty = abs(1.0 - aspect)
                score = conf * 100 - aspect_penalty * 10
                if predicted_point is not None and self.missing_frames <= self.max_missing_predict_frames:
                    opposite_crossing = (
                        self.kalman_initialized
                        and self.missing_frames > 5
                        and self.last_stable_ball_side is not None
                        and self.last_stable_ball_side * self._side_for_x(center[0]) == -1
                    )
                    dist = np.linalg.norm(np.array(center) - np.array(predicted_point))
                    gate = self.reacquire_gate_px if self.missing_frames > 3 else self.tracking_gate_px
                    if self.kalman_initialized and dist > gate and not opposite_crossing:
                        if self.debug_diagnostics:
                            self._diag["yolo_rejected_tracking_gate"] += 1
                            if box_diag is not None:
                                box_diag["reject_reason"] = "tracking-gate"
                        continue
                    elif self.kalman_initialized and dist > gate and opposite_crossing:
                        if self.debug_diagnostics:
                            self._diag["gate_bypass_opposite"] += 1
                    score += max(0, gate - dist) * 0.25

                if best is None or score > best[0]:
                    best = (score, center, (x1, y1, w, h), (x1, y1, x2, y2))
                    if box_diag is not None:
                        box_diag["reject_reason"] = "selected-candidate"

        if self.debug_diagnostics and self._diag_last_yolo["probe_boxes"] > 0:
            self._diag["yolo_probe_found"] += 1

        if best is None:
            return None, None

        _, center, bbox, xyxy = best
        self.last_yolo_bbox = xyxy
        if self.debug_diagnostics:
            self._diag["yolo_accepted"] += 1
            self._diag_last_yolo["accepted"] = True
        return center, bbox

    def _update_tracker(self, measurement):
        prediction = self.kalman.predict()
        predicted_point = (int(prediction[0][0]), int(prediction[1][0]))

        if measurement is None:
            self.missing_frames += 1
            if self.kalman_initialized and self.missing_frames <= self.max_missing_predict_frames:
                if self.debug_diagnostics:
                    self._diag["tracker_predict_only"] += 1
                return predicted_point, False
            return None, False

        mx, my = measurement
        measured = np.array([[np.float32(mx)], [np.float32(my)]])
        should_reinitialize = (
            not self.kalman_initialized
            or self.missing_frames > self.max_missing_predict_frames
        )
        if should_reinitialize:
            initial_state = np.array(
                [[np.float32(mx)], [np.float32(my)], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.kalman.statePost = initial_state.copy()
            self.kalman.statePre = initial_state.copy()
            self.kalman_initialized = True
            self.missing_frames = 0
            if self.debug_diagnostics:
                self._diag["tracker_reinitialized"] += 1
            # Keep the last confirmed side across reacquisition. Losing the
            # detector must not turn the next crossing into a new first sight.
            self.pending_crossing = None
            return (int(mx), int(my)), True
        corrected = self.kalman.correct(measured)
        self.missing_frames = 0
        if self.debug_diagnostics:
            self._diag["tracker_corrected"] += 1
        return (int(corrected[0][0]), int(corrected[1][0])), True

    def _validate_measurement_velocity(self, measurement):
        """Optionally reject only physically extreme detector jumps."""
        self.last_measurement_speed = None
        if measurement is None or self.last_measurement is None:
            return True

        frame_delta = max(1, self.frame_index - self.last_measurement_frame)
        dx = measurement[0] - self.last_measurement[0]
        dy = measurement[1] - self.last_measurement[1]
        speed = ((dx * dx + dy * dy) ** 0.5) / frame_delta
        self.last_measurement_speed = speed
        max_speed = self.max_measurement_speed_ratio * max(
            1,
            min(self.FRAME_WIDTH, self.FRAME_HEIGHT),
        )
        if speed > max_speed:
            self.last_count_reason = "measurement-speed-rejected"
            if self.debug_diagnostics:
                self._diag["velocity_rejected"] += 1
            return False
        return True

    def _record_measurement(self, measurement):
        if measurement is None:
            return

        self.last_measurement = (int(measurement[0]), int(measurement[1]))
        self.last_measurement_frame = self.frame_index

    def _side_for_x(self, x):
        side_margin = max(
            self.NET_BUFFER,
            self.count_side_margin_px,
            int(self.FRAME_WIDTH * 0.015),
        )
        if x < self.net_x_position - side_margin:
            return -1
        if x > self.net_x_position + side_margin:
            return 1
        return 0

    def _commit_crossing(self, to_side, can_count):
        if can_count:
            self.ball_count += 1
            self.last_count_frame = self.frame_index
            self.last_count_reason = "crossing-counted"
            if self.debug_diagnostics:
                self._diag["crossings_committed"] += 1
        else:
            self.last_count_reason = "crossing-suppressed-cooldown"
            if self.debug_diagnostics:
                self._diag["crossings_cooldown_suppressed"] += 1
        self.ball_direction = to_side
        self.last_stable_ball_side = to_side
        self.last_ball_side = to_side
        self.pending_crossing = None

    def _update_count(self, point):
        if point is None:
            return

        point = (int(point[0]), int(point[1]))
        curr_side = self._side_for_x(point[0])
        can_count = (self.frame_index - self.last_count_frame) >= self.count_cooldown_frames

        if (
            self.pending_crossing is not None
            and self.frame_index - self.pending_crossing["start_frame"] > self.max_crossing_frames
        ):
            self.pending_crossing = None
            self.last_count_reason = "crossing-expired"
            if self.debug_diagnostics:
                self._diag["crossings_expired"] += 1

        if curr_side == 0:
            self.prev_ball_pos = point
            return

        # Preserve the original tracker's recall: the first valid observation
        # establishes a side immediately. Only a transition needs confirmation.
        if self.last_stable_ball_side is None:
            self.last_stable_ball_side = curr_side
            self.last_ball_side = curr_side
            self.ball_direction = curr_side
            self.prev_ball_pos = point
            return

        if self.pending_crossing is not None:
            pending = self.pending_crossing
            if curr_side == pending["from_side"]:
                self.pending_crossing = None
                self.last_count_reason = "crossing-cancelled-jitter"
                if self.debug_diagnostics:
                    self._diag["crossings_jitter_cancelled"] += 1
            elif curr_side == pending["to_side"]:
                pending["observations"] += 1
                if (
                    not self.enable_crossing_validation
                    or pending["observations"] >= self.min_crossing_observations
                ):
                    self._commit_crossing(pending["to_side"], can_count)
            self.prev_ball_pos = point
            return

        if curr_side != self.last_stable_ball_side:
            if self.debug_diagnostics:
                self._diag["side_transitions"] += 1
            if (
                not self.enable_crossing_validation
                or self.min_crossing_observations <= 1
            ):
                self._commit_crossing(curr_side, can_count)
                self.prev_ball_pos = point
                return
            self.pending_crossing = {
                "from_side": self.last_stable_ball_side,
                "to_side": curr_side,
                "start_frame": self.frame_index,
                "observations": 1,
            }
            self.last_ball_side = curr_side
        else:
            self.ball_direction = curr_side
            self.last_ball_side = curr_side

        self.prev_ball_pos = point

    def _draw_debug(self, frame, fgmask, candidate_mask, point, measurement_bbox, detection_source):
        if measurement_bbox is not None:
            bx, by, bw, bh = measurement_bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        cv2.putText(frame, f"Count: {self.ball_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)
        cv2.line(
            frame,
            (self.net_x_position, max(0, self.FRAME_HEIGHT // 4)),
            (self.net_x_position, min(self.FRAME_HEIGHT - 1, self.FRAME_HEIGHT * 3 // 4)),
            (255, 0, 0),
            2,
        )

        if self.debug_diagnostics and self.debug_overlay:
            overlay_lines = [
                f"raw={self._diag['yolo_raw_boxes']} accept={self._diag['yolo_accepted']}",
                f"low_conf={self._diag['yolo_rejected_conf']} gate={self._diag['yolo_rejected_tracking_gate']}",
                f"gate_bypass={self._diag.get('gate_bypass_opposite', 0)}",
                f"area={self._diag['yolo_rejected_area']} side={self._diag['yolo_rejected_side']}",
                f"skip={self._diag['yolo_skips']} err={self._diag['yolo_errors']}",
                f"miss={self.missing_frames} stable={self.last_stable_ball_side}",
                f"reason={self.last_count_reason}",
                f"classes={self._diag_raw_classes}",
            ]
            for index, text in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    text,
                    (10, 80 + index * 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    1,
                )

        side = {
            "mask": fgmask,
            "candidate_mask": candidate_mask,
            "count": self.ball_count,
            "table_bbox": self.table_bbox,
            "net_x_position": self.net_x_position,
            "tracked_ball": point,
        }
        if self.debug_diagnostics:
            detail = self._diag_last_yolo.copy()
            detail["frame"] = self.frame_index
            detail["point"] = point
            detail["source"] = detection_source
            detail["stable_side"] = self.last_stable_ball_side
            detail["count"] = self.ball_count
            detail["count_reason"] = self.last_count_reason
            if len(self._diag_frame_detail) < self.debug_history_size:
                self._diag_frame_detail.append(detail)
        return side

    def process(self, frame_bgr):
        process_start = time.perf_counter()
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
                if self.debug_diagnostics:
                    self._diag["opencv_accepted"] += 1

        if self.enable_velocity_filter:
            if not self._validate_measurement_velocity(measurement):
                measurement = None
                measurement_bbox = None
                detection_source = f"{detection_source}-rejected"
            else:
                self._record_measurement(measurement)

        tracked_point, has_measurement = self._update_tracker(measurement)
        can_count_prediction = (
            tracked_point is not None
            and self.kalman_initialized
            and self.missing_frames <= self.count_predict_frames
        )
        if tracked_point is not None and (has_measurement or can_count_prediction):
            self._update_count(tracked_point)

        side = self._draw_debug(frame, fgmask, candidate_mask, tracked_point, measurement_bbox, detection_source)
        side["detection_source"] = detection_source
        side["has_measurement"] = has_measurement
        side["missing_frames"] = self.missing_frames
        side["stable_ball_side"] = self.last_stable_ball_side
        side["pending_crossing"] = self.pending_crossing
        side["yolo_available"] = self.yolo_available
        side["yolo_status"] = self.yolo_status
        if self.debug_diagnostics:
            self._diag["frames_processed"] = self.frame_index
            self._diag["process_ms"] += (time.perf_counter() - process_start) * 1000.0
            if (
                self.debug_log_interval > 0
                and self.frame_index % self.debug_log_interval == 0
                and self.frame_index != self._diag["last_diag_print_frame"]
            ):
                self._diag["last_diag_print_frame"] = self.frame_index
                print(self._format_diagnostics_summary())
            side["diagnostics"] = self.get_diagnostics()
        return frame, side

    # -----------------------------------------------------------------------
    # Diagnostic reporting
    # -----------------------------------------------------------------------

    def get_diagnostics(self):
        """Return a snapshot of the current diagnostic counters."""
        diag = self._diag.copy()
        diag["table_detected"] = self.table_bbox is not None
        diag["table_bbox"] = self.table_bbox
        diag["net_x_position"] = self.net_x_position
        diag["stable_ball_side"] = self.last_stable_ball_side
        diag["pending_crossing"] = self.pending_crossing
        diag["missing_frames"] = self.missing_frames
        diag["yolo_raw_classes"] = self._diag_raw_classes.copy()
        diag["yolo_probe_classes"] = self._diag_probe_classes.copy()
        diag["yolo_last_frame"] = self._diag_last_yolo.copy()
        diag["recent_frames"] = self._diag_frame_detail[-20:]
        diag["model_path"] = self.yolo_model_path
        diag["model_status"] = self.yolo_status
        diag["model_names"] = self.yolo_model_names
        diag["probe_boxes_summary"] = {
            "total": self._diag["yolo_probe_boxes"],
            "above_threshold": self._diag["yolo_raw_boxes"],
            "accepted": self._diag["yolo_accepted"],
            "frames_with_any_probe_box": self._diag["yolo_probe_found"],
            "by_class": self._diag_probe_classes.copy(),
        }
        diag["config"] = {
            "yolo_conf": self.yolo_conf,
            "yolo_acquire_conf": self.yolo_acquire_conf,
            "yolo_probe_conf": self.debug_probe_conf,
            "yolo_imgsz": self.yolo_imgsz,
            "yolo_every_n_frames": self.yolo_every_n_frames,
            "yolo_use_table_crop": self.yolo_use_table_crop,
            "count_cooldown_frames": self.count_cooldown_frames,
            "min_crossing_observations": self.min_crossing_observations,
            "max_missing_predict_frames": self.max_missing_predict_frames,
            "count_predict_frames": self.count_predict_frames,
            "net_buffer": self.NET_BUFFER,
            "count_side_margin_px": self.count_side_margin_px,
            "yolo_max_area_ratio": self.yolo_max_area_ratio,
            "yolo_max_side_ratio": self.yolo_max_side_ratio,
            "reacquire_gate_px": self.reacquire_gate_px,
            "tracking_gate_px": self.tracking_gate_px,
        }
        diag["frame_detail_count"] = len(self._diag_frame_detail)
        frames = max(1, diag["frames_processed"])
        diag["yolo_avg_inference_ms"] = round(diag["yolo_inference_ms"] / max(1, diag["yolo_calls"]), 2)
        diag["process_avg_ms"] = round(diag["process_ms"] / frames, 2)
        diag["process_fps"] = round(1000.0 / max(0.001, diag["process_avg_ms"]), 2)
        diag["yolo_run_ratio"] = round(diag["yolo_calls"] / frames, 4)
        diag["accept_ratio"] = round(diag["yolo_accepted"] / max(1, diag["yolo_raw_boxes"]), 4)
        diag["probe_conf"] = self.debug_probe_conf
        diag["gate_bypass_opposite"] = self._diag.get("gate_bypass_opposite", 0)
        return diag

    def _format_diagnostics_summary(self):
        d = self.get_diagnostics()
        avg_yolo = d.get("yolo_avg_inference_ms", 0)
        avg_proc = d.get("process_avg_ms", 0)
        fps = d.get("process_fps", 0)
        return (
            f"[BALL-DIAG] f={d['frames_processed']} "
            f"count={self.ball_count} "
            f"yolo_calls={d['yolo_calls']} "
            f"raw={d['yolo_raw_boxes']} "
            f"probe={d['yolo_probe_boxes']} "
            f"probe_found={d['yolo_probe_found']} "
            f"accepted={d['yolo_accepted']} "
            f"skip={d['yolo_skips']} "
            f"err={d['yolo_errors']} "
            f"gate_bypass={d.get('gate_bypass_opposite', 0)} "
            f"avg_yolo={avg_yolo:.1f}ms "
            f"avg_proc={avg_proc:.1f}ms "
            f"fps={fps:.1f} "
            f"side={d.get('stable_ball_side')} "
            f"pending={d.get('pending_crossing')} "
            f"reason={self.last_count_reason} "
            f"classes={d.get('yolo_raw_classes', {})}"
        )
