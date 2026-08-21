#   Vai trái=11, vai phải=12, Khuỷu trái=13, Khuỷu phải=14,
#   Cổ tay trái=15, Cổ tay phải=16, Hông trái=23, Hông phải=24,
#   Gối trái=25, Gối phải=26, Cổ chân trái=27, Cổ chân phải=28

import cv2 as cv
import PoseModule as pm
import math
import numpy as np


FEATURE_KEYS = (
    "right_elbow_angle",
    "forearm_direction",
    "shoulder_line_angle",
    "hip_line_angle",
    "torso_twist_abs",
    "right_knee_angle",
    "left_knee_angle",
    "stance_width_norm",
)

CIRCULAR_FEATURES = {
    "forearm_direction",
    "shoulder_line_angle",
    "hip_line_angle",
}

REQUIRED_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
FEATURE_METADATA_KEYS = {
    "valid_frames",
    "total_frames",
    "valid_ratio",
    "use_3d",
    "fps",
    "duration_seconds",
    "timestamps",
    "frame_indices",
    "frame_quality",
}

FEATURE_LABELS_EN = {
    "right_elbow_angle":    "Right elbow",
    "forearm_direction":    "Forearm/wrist direction",
    "shoulder_line_angle":  "Shoulder line",
    "hip_line_angle":       "Hip line",
    "torso_twist_abs":      "Torso twist",
    "right_knee_angle":     "Right knee",
    "left_knee_angle":      "Left knee",
    "stance_width_norm":    "Stance width",
}

FEATURE_LABELS_VI = {
    "right_elbow_angle":    "Khuỷu tay phải",
    "forearm_direction":    "Hướng cẳng tay/cổ tay",
    "shoulder_line_angle":  "Đường vai",
    "hip_line_angle":       "Đường hông",
    "torso_twist_abs":      "Xoay thân",
    "right_knee_angle":     "Gối phải",
    "left_knee_angle":      "Gối trái",
    "stance_width_norm":    "Độ rộng thế đứng",
}

# Preset weights cho các động tác bóng bàn
PRESET_WEIGHTS = {
    "default": {
        "right_elbow_angle": 0.25,
        "forearm_direction": 0.15,
        "shoulder_line_angle": 0.15,
        "hip_line_angle": 0.10,
        "torso_twist_abs": 0.15,
        "right_knee_angle": 0.08,
        "left_knee_angle": 0.07,
        "stance_width_norm": 0.05,
    },
    "table_tennis_forehand": {
        "right_elbow_angle": 0.28,
        "forearm_direction": 0.22,
        "shoulder_line_angle": 0.20,
        "hip_line_angle": 0.10,
        "torso_twist_abs": 0.12,
        "right_knee_angle": 0.04,
        "left_knee_angle": 0.03,
        "stance_width_norm": 0.01,
    },
    "table_tennis_backhand": {
        "right_elbow_angle": 0.25,
        "forearm_direction": 0.20,
        "shoulder_line_angle": 0.18,
        "hip_line_angle": 0.12,
        "torso_twist_abs": 0.15,
        "right_knee_angle": 0.05,
        "left_knee_angle": 0.04,
        "stance_width_norm": 0.01,
    },
    "table_tennis_serve": {
        "right_elbow_angle": 0.30,
        "forearm_direction": 0.25,
        "shoulder_line_angle": 0.20,
        "hip_line_angle": 0.08,
        "torso_twist_abs": 0.10,
        "right_knee_angle": 0.04,
        "left_knee_angle": 0.02,
        "stance_width_norm": 0.01,
    },
    "table_tennis_smash": {
        "right_elbow_angle": 0.32,
        "forearm_direction": 0.28,
        "shoulder_line_angle": 0.22,
        "hip_line_angle": 0.05,
        "torso_twist_abs": 0.08,
        "right_knee_angle": 0.03,
        "left_knee_angle": 0.01,
        "stance_width_norm": 0.01,
    },
    "table_tennis_loop": {
        "right_elbow_angle": 0.26,
        "forearm_direction": 0.24,
        "shoulder_line_angle": 0.22,
        "hip_line_angle": 0.10,
        "torso_twist_abs": 0.12,
        "right_knee_angle": 0.03,
        "left_knee_angle": 0.02,
        "stance_width_norm": 0.01,
    },
    "table_tennis_push": {
        "right_elbow_angle": 0.22,
        "forearm_direction": 0.20,
        "shoulder_line_angle": 0.18,
        "hip_line_angle": 0.12,
        "torso_twist_abs": 0.15,
        "right_knee_angle": 0.06,
        "left_knee_angle": 0.05,
        "stance_width_norm": 0.02,
    },
    "table_tennis_block": {
        "right_elbow_angle": 0.20,
        "forearm_direction": 0.18,
        "shoulder_line_angle": 0.16,
        "hip_line_angle": 0.14,
        "torso_twist_abs": 0.18,
        "right_knee_angle": 0.07,
        "left_knee_angle": 0.06,
        "stance_width_norm": 0.01,
    },
}

def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def _dist_3d(p, q):
    """Tính khoảng cách 3D giữa 2 điểm"""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    dz = p[2] - q[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def _angle_of_vector_deg(v):
    return math.degrees(math.atan2(v[1], v[0]))

def _joint_angle_deg(a, b, c):
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    ang1 = math.atan2(v1[1], v1[0])
    ang2 = math.atan2(v2[1], v2[0])
    ang = abs(math.degrees(ang2 - ang1))
    if ang > 180:
        ang = 360 - ang
    return ang

def _joint_angle_3d_deg(a, b, c):
    """Tính góc 3D giữa 3 điểm (góc tại điểm b)"""
    v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    
    # Tính góc bằng dot product
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def _wrap_angle_diff_deg(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)

def _extract_xy_from_lmList(lmList):
    pts = {}
    for item in lmList:
        if len(item) >= 3:
            idx, x, y = int(item[0]), float(item[1]), float(item[2])
            pts[idx] = (x, y)
    return pts

def _extract_xyz_from_lmList(lmList):
    """
    Trích xuất tọa độ 3D từ lmList.
    
    Format mới từ pose_world_landmarks: [id, cx, cy, wx, wy, wz]
    - cx, cy: tọa độ pixel trên image (để vẽ)
    - wx, wy, wz: tọa độ 3D trong world space (metric, meters)
    
    Returns:
        Dictionary với key là landmark index, value là tuple (wx, wy, wz) - world coordinates
    """
    pts = {}
    for item in lmList:
        if len(item) >= 6:
            # Format mới: [id, cx, cy, wx, wy, wz]
            idx = int(item[0])
            wx, wy, wz = float(item[3]), float(item[4]), float(item[5])
            pts[idx] = (wx, wy, wz)
        elif len(item) >= 4:
            # Format cũ (fallback): [id, cx, cy, cz] - dùng z từ pose_landmarks
            idx = int(item[0])
            # Nếu là format cũ, dùng z đã scale
            x, y, z = float(item[1]), float(item[2]), float(item[3])
            # Giả sử đây là world coordinates đã scale
            pts[idx] = (x, y, z)
        elif len(item) >= 3:
            # Fallback: nếu không có z, dùng z=0
            idx, x, y = int(item[0]), float(item[1]), float(item[2])
            pts[idx] = (x, y, 0.0)
    return pts


def _extract_visibility_from_lmList(lmList, use_3d=False):
    """Return per-landmark confidence while accepting legacy formats."""
    visibility = {}
    for item in lmList:
        if len(item) < 3:
            continue
        idx = int(item[0])
        if use_3d and len(item) >= 8:
            vis, presence = float(item[6]), float(item[7])
        elif not use_3d and len(item) >= 5:
            vis, presence = float(item[3]), float(item[4])
        else:
            # Legacy callers did not expose confidence; preserve their behavior.
            vis, presence = 1.0, 1.0
        # Legacy MediaPipe Solutions may leave `presence` at protobuf default
        # zero even when visibility is populated. Only combine it when present.
        visibility[idx] = min(vis, presence) if presence > 0.0 else vis
    return visibility

def _normalize_landmarks(pts):
    must_have = [11, 12, 23, 24]
    if not all(k in pts for k in must_have):
        return None
    mid_hip = ((pts[23][0] + pts[24][0]) / 2.0, (pts[23][1] + pts[24][1]) / 2.0)
    trans = {}
    for k in pts:
        x = pts[k][0] - mid_hip[0]
        y = pts[k][1] - mid_hip[1]
        trans[k] = (x, y)
    
    shoulder_w = _dist(trans[11], trans[12])
    if shoulder_w < 1e-6:
        return None
    scaled = {}
    for k in trans:
        x = trans[k][0] / shoulder_w
        y = trans[k][1] / shoulder_w
        scaled[k] = (x,y)
    return scaled

def _normalize_landmarks_3d(pts, reference_shoulder_dir=None):
    """
    Normalize a 3D pose for body-position and body-size differences.
    
    Args:
        pts: Dictionary với key là landmark index, value là tuple (x, y, z)
        reference_shoulder_dir: Deprecated compatibility argument. Per-frame
            shoulder alignment is intentionally not applied because it removes
            the shoulder rotation that the comparison is meant to assess.
    
    Returns:
        Landmarks translated to mid-hip and scaled by shoulder width.
    """
    must_have = [11, 12, 23, 24]
    if not all(k in pts for k in must_have):
        return None
    
    # 1. Translate về mid_hip
    mid_hip = (
        (pts[23][0] + pts[24][0]) / 2.0,
        (pts[23][1] + pts[24][1]) / 2.0,
        (pts[23][2] + pts[24][2]) / 2.0
    )
    trans = {}
    for k in pts:
        x = pts[k][0] - mid_hip[0]
        y = pts[k][1] - mid_hip[1]
        z = pts[k][2] - mid_hip[2]
        trans[k] = np.array([x, y, z])
    
    # 2. Scale theo shoulder width
    shoulder_vec = trans[12] - trans[11]
    shoulder_w = np.linalg.norm(shoulder_vec)
    if shoulder_w < 1e-6:
        return None
    
    scaled = {}
    for k in trans:
        scaled[k] = trans[k] / shoulder_w
    
    # Do not rotate every frame to match the reference shoulders. That would
    # make shoulder_line_angle nearly constant and hide real torso rotation.
    
    # Chuyển về tuple để tương thích với code cũ
    result = {}
    for k in scaled:
        result[k] = tuple(scaled[k])
    
    return result

def _compute_frame_features(pts_norm):
    idx_map = {
        "L_SH": 11, "R_SH": 12,
        "L_EL": 13, "R_EL": 14,
        "L_WR": 15, "R_WR": 16,
        "L_HP": 23, "R_HP": 24,
        "L_KN": 25, "R_KN": 26,
        "L_AN": 27, "R_AN": 28,
    }
    need = set(idx_map.values())
    if not all(i in pts_norm for i in need):
        return None
    SH_L, SH_R = pts_norm[idx_map["L_SH"]], pts_norm[idx_map["R_SH"]]
    EL_R, WR_R = pts_norm[idx_map["R_EL"]], pts_norm[idx_map["R_WR"]]
    HP_L, HP_R = pts_norm[idx_map["L_HP"]], pts_norm[idx_map["R_HP"]]
    KN_L, KN_R = pts_norm[idx_map["L_KN"]], pts_norm[idx_map["R_KN"]]
    AN_L, AN_R = pts_norm[idx_map["L_AN"]], pts_norm[idx_map["R_AN"]]
    right_elbow_angle = _joint_angle_deg(SH_R, EL_R, WR_R)
    forearm_vec = (WR_R[0] - EL_R[0], WR_R[1] - EL_R[1])
    forearm_dir = _angle_of_vector_deg(forearm_vec)
    if forearm_dir < 0:
        forearm_dir += 360
    shoulder_dir = _angle_of_vector_deg((SH_R[0] - SH_L[0], SH_R[1] - SH_L[1]))
    if shoulder_dir < 0:
        shoulder_dir += 360
    hip_dir = _angle_of_vector_deg((HP_R[0] - HP_L[0], HP_R[1] - HP_L[1]))
    if hip_dir < 0:
        hip_dir += 360
    torso_twist = abs(_wrap_angle_diff_deg(shoulder_dir, hip_dir))
    right_knee_angle = _joint_angle_deg(HP_R, KN_R, AN_R)
    left_knee_angle = _joint_angle_deg(HP_L, KN_L, AN_L)
    stance_width_norm = _dist(AN_L, AN_R) #tỉ lệ
    return {
        "right_elbow_angle": right_elbow_angle,
        "forearm_direction": forearm_dir,
        "shoulder_line_angle": shoulder_dir,
        "hip_line_angle": hip_dir,
        "torso_twist_abs": torso_twist,
        "right_knee_angle": right_knee_angle,
        "left_knee_angle": left_knee_angle,
        "stance_width_norm": stance_width_norm,
    }

def _angle_of_vector_3d_deg(v):
    """Tính góc của vector 3D trong mặt phẳng XY (projection)"""
    return math.degrees(math.atan2(v[1], v[0]))

def _compute_frame_features_3d(pts_norm):
    """
    Tính features từ pose 3D đã normalize.
    Sử dụng góc 3D thực tế thay vì góc 2D để chính xác hơn.
    """
    idx_map = {
        "L_SH": 11, "R_SH": 12,
        "L_EL": 13, "R_EL": 14,
        "L_WR": 15, "R_WR": 16,
        "L_HP": 23, "R_HP": 24,
        "L_KN": 25, "R_KN": 26,
        "L_AN": 27, "R_AN": 28,
    }
    need = set(idx_map.values())
    if not all(i in pts_norm for i in need):
        return None
    
    SH_L, SH_R = pts_norm[idx_map["L_SH"]], pts_norm[idx_map["R_SH"]]
    EL_R, WR_R = pts_norm[idx_map["R_EL"]], pts_norm[idx_map["R_WR"]]
    HP_L, HP_R = pts_norm[idx_map["L_HP"]], pts_norm[idx_map["R_HP"]]
    KN_L, KN_R = pts_norm[idx_map["L_KN"]], pts_norm[idx_map["R_KN"]]
    AN_L, AN_R = pts_norm[idx_map["L_AN"]], pts_norm[idx_map["R_AN"]]
    
    # Tính góc 3D thực tế (chính xác hơn 2D)
    right_elbow_angle = _joint_angle_3d_deg(SH_R, EL_R, WR_R)
    
    # Forearm direction: projection lên mặt phẳng XY
    forearm_vec = (WR_R[0] - EL_R[0], WR_R[1] - EL_R[1], WR_R[2] - EL_R[2])
    forearm_dir = _angle_of_vector_3d_deg(forearm_vec)
    if forearm_dir < 0:
        forearm_dir += 360
    
    # Shoulder line direction: projection lên mặt phẳng XY
    shoulder_vec = (SH_R[0] - SH_L[0], SH_R[1] - SH_L[1], SH_R[2] - SH_L[2])
    shoulder_dir = _angle_of_vector_3d_deg(shoulder_vec)
    if shoulder_dir < 0:
        shoulder_dir += 360
    
    # Hip line direction: projection lên mặt phẳng XY
    hip_vec = (HP_R[0] - HP_L[0], HP_R[1] - HP_L[1], HP_R[2] - HP_L[2])
    hip_dir = _angle_of_vector_3d_deg(hip_vec)
    if hip_dir < 0:
        hip_dir += 360
    
    # Torso twist: góc giữa 2 vector trong không gian 3D
    shoulder_vec_norm = np.array(shoulder_vec) / (np.linalg.norm(shoulder_vec) + 1e-6)
    hip_vec_norm = np.array(hip_vec) / (np.linalg.norm(hip_vec) + 1e-6)
    dot = np.clip(np.dot(shoulder_vec_norm, hip_vec_norm), -1.0, 1.0)
    torso_twist_3d = math.degrees(math.acos(abs(dot)))
    # Cũng tính torso_twist theo cách cũ (2D projection) để tương thích
    torso_twist = abs(_wrap_angle_diff_deg(shoulder_dir, hip_dir))
    
    # Góc gối 3D
    right_knee_angle = _joint_angle_3d_deg(HP_R, KN_R, AN_R)
    left_knee_angle = _joint_angle_3d_deg(HP_L, KN_L, AN_L)
    
    # Stance width: khoảng cách 3D
    stance_width_norm = _dist_3d(AN_L, AN_R)
    
    return {
        "right_elbow_angle": right_elbow_angle,
        "forearm_direction": forearm_dir,
        "shoulder_line_angle": shoulder_dir,
        "hip_line_angle": hip_dir,
        "torso_twist_abs": torso_twist,  # Giữ 2D để tương thích
        "torso_twist_3d": torso_twist_3d,  # Thêm 3D version
        "right_knee_angle": right_knee_angle,
        "left_knee_angle": left_knee_angle,
        "stance_width_norm": stance_width_norm,
    }

def _smooth_feature_sequence(values, feature_name, window=5):
    if len(values) < 3 or window <= 1:
        return list(values)
    window = min(int(window), len(values))
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    if feature_name in CIRCULAR_FEATURES:
        radians = np.unwrap(np.deg2rad(np.asarray(values, dtype=float)))
        padded = np.pad(radians, (pad_left, pad_right), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return list(np.mod(np.rad2deg(smoothed), 360.0))
    padded = np.pad(np.asarray(values, dtype=float), (pad_left, pad_right), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid"))


def extract_features(
    video_path,
    draw=False,
    smooth=False,
    use_3d=False,
    reference_shoulder_dir=None,
    min_visibility=0.5,
):
    """
    Trích xuất features từ video.
    
    Args:
        video_path: Đường dẫn video
        draw: Có vẽ skeleton không
        smooth: Làm mượt chuỗi đặc trưng sau khi trích xuất.
        use_3d: Sử dụng tọa độ 3D từ MediaPipe (tốt hơn khi góc quay khác nhau)
        reference_shoulder_dir: Tham số tương thích cũ; không còn xoay từng frame
                               theo vai vì thao tác đó làm mất chuyển động thật.
    
    Returns:
        Dictionary chứa features
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    detector = pm.poseDetector()
    feats = {key: [] for key in FEATURE_KEYS}
    if use_3d:
        feats["torso_twist_3d"] = []

    fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = 0
    valid_frames = 0
    timestamps = []
    frame_indices = []
    frame_quality = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index = total_frames
            total_frames += 1
            frame_proc = detector.findPose(frame, draw=draw)
            lmList = detector.findPosition(frame_proc, draw=False, use_3d=use_3d)
            if not lmList or len(lmList) < 29:
                continue

            visibility = _extract_visibility_from_lmList(lmList, use_3d=use_3d)
            if not all(idx in visibility for idx in REQUIRED_LANDMARKS):
                continue
            quality = min(visibility[idx] for idx in REQUIRED_LANDMARKS)
            if quality < float(min_visibility):
                continue

            if use_3d:
                pts = _extract_xyz_from_lmList(lmList)
                pts_norm = _normalize_landmarks_3d(pts, reference_shoulder_dir)
                if pts_norm is None:
                    continue
                f = _compute_frame_features_3d(pts_norm)
            else:
                pts = _extract_xy_from_lmList(lmList)
                pts_norm = _normalize_landmarks(pts)
                if pts_norm is None:
                    continue
                f = _compute_frame_features(pts_norm)

            if f is None:
                continue

            for key in feats:
                if key in f:
                    feats[key].append(float(f[key]))
            timestamps.append(frame_index / fps)
            frame_indices.append(frame_index)
            frame_quality.append(float(quality))
            valid_frames += 1
    finally:
        cap.release()
        close_detector = getattr(detector, "close", None)
        if callable(close_detector):
            close_detector()

    if smooth:
        for key, values in list(feats.items()):
            feats[key] = _smooth_feature_sequence(values, key)

    feats["valid_frames"] = valid_frames
    feats["total_frames"] = total_frames
    feats["valid_ratio"] = valid_frames / max(1, total_frames)
    feats["use_3d"] = use_3d
    feats["fps"] = fps
    feats["duration_seconds"] = total_frames / fps
    feats["timestamps"] = timestamps
    feats["frame_indices"] = frame_indices
    feats["frame_quality"] = frame_quality
    return feats


def _resample_numeric(values, x_old, x_new, circular=False):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return [float(arr[0])] * len(x_new)
    if circular:
        radians = np.unwrap(np.deg2rad(arr))
        values_new = np.mod(np.rad2deg(np.interp(x_new, x_old, radians)), 360.0)
    else:
        values_new = np.interp(x_new, x_old, arr)
    return list(map(float, values_new))


def resample_features(feats, n=100):
    """Resample features on the real video timeline without changing body scale."""
    n = max(2, int(n))
    feature_lengths = [
        len(v) for k, v in feats.items()
        if k not in FEATURE_METADATA_KEYS and isinstance(v, (list, tuple)) and len(v) > 0
    ]
    if not feature_lengths:
        return {}
    source_len = min(feature_lengths)
    timestamps = feats.get("timestamps", [])
    if isinstance(timestamps, (list, tuple)) and len(timestamps) >= source_len:
        x_old = np.asarray(timestamps[:source_len], dtype=float)
        if len(x_old) > 1 and x_old[-1] > x_old[0]:
            x_old = (x_old - x_old[0]) / (x_old[-1] - x_old[0])
        else:
            x_old = np.linspace(0.0, 1.0, num=source_len)
    else:
        x_old = np.linspace(0.0, 1.0, num=source_len)
    x_new = np.linspace(0.0, 1.0, num=n)

    out = {}
    for key, values in feats.items():
        if key in FEATURE_METADATA_KEYS:
            continue
        if not isinstance(values, (list, tuple)) or len(values) < source_len:
            continue
        out[key] = _resample_numeric(
            values[:source_len], x_old, x_new, circular=key in CIRCULAR_FEATURES
        )

    start_time = float(timestamps[0]) if timestamps else 0.0
    end_time = float(timestamps[source_len - 1]) if len(timestamps) >= source_len else float(feats.get("duration_seconds", 0.0))
    out["timestamps"] = list(map(float, np.linspace(start_time, end_time, num=n)))
    out["fps"] = float(feats.get("fps", 30.0))
    out["duration_seconds"] = float(feats.get("duration_seconds", max(0.0, end_time - start_time)))
    out["valid_frames"] = int(feats.get("valid_frames", source_len))
    out["total_frames"] = int(feats.get("total_frames", source_len))
    out["valid_ratio"] = float(feats.get("valid_ratio", 1.0))
    out["use_3d"] = bool(feats.get("use_3d", False))
    quality = feats.get("frame_quality", [])
    if isinstance(quality, (list, tuple)) and len(quality) >= source_len:
        out["frame_quality"] = _resample_numeric(quality[:source_len], x_old, x_new)
    return out


def get_effective_fps(feats, default=30.0):
    """Return samples/second for a resampled sequence using its actual duration."""
    timestamps = feats.get("timestamps", []) if isinstance(feats, dict) else []
    if isinstance(timestamps, (list, tuple)) and len(timestamps) > 1:
        duration = float(timestamps[-1]) - float(timestamps[0])
        if duration > 1e-6:
            return (len(timestamps) - 1) / duration
    return float(default)

def _run_dtw(n, m, local_cost, window=None, free_start=True, free_end=True):
    """Run DTW on one or many features and return average cost plus one path."""
    if n <= 0 or m <= 0:
        return float("inf"), []
    window = max(n, m) if window is None else max(1, int(window))
    costs = np.full((n + 1, m + 1), np.inf, dtype=float)

    if free_start:
        start_region = min(10, max(1, int(0.1 * min(n, m))))
        costs[0, : min(m, start_region) + 1] = 0.0
        costs[: min(n, start_region) + 1, 0] = 0.0
    else:
        costs[0, 0] = 0.0

    # Center the window on a length-scaled diagonal. This keeps unequal-length
    # sequences reachable, which is essential when teacher/student move at
    # different speeds.
    for i in range(1, n + 1):
        expected_j = int(round(i * m / float(n)))
        j_start = max(1, expected_j - window)
        j_end = min(m, expected_j + window)
        for j in range(j_start, j_end + 1):
            predecessor = min(costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1])
            if np.isfinite(predecessor):
                costs[i, j] = float(local_cost(i - 1, j - 1)) + predecessor

    if free_end:
        end_region = min(10, max(1, int(0.1 * min(n, m))))
        endpoints = [(n, j) for j in range(max(1, m - end_region), m + 1)]
        endpoints += [(i, m) for i in range(max(1, n - end_region), n + 1)]
        best_i, best_j = min(endpoints, key=lambda ij: costs[ij[0], ij[1]])
    else:
        best_i, best_j = n, m
    best_cost = costs[best_i, best_j]
    if not np.isfinite(best_cost):
        return float("inf"), []

    i, j = best_i, best_j
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        predecessors = [
            (i - 1, j - 1, costs[i - 1, j - 1]),
            (i - 1, j, costs[i - 1, j]),
            (i, j - 1, costs[i, j - 1]),
        ]
        finite = [item for item in predecessors if np.isfinite(item[2])]
        if not finite:
            break
        i, j, _ = min(finite, key=lambda item: item[2])
    path.reverse()
    if not path:
        return float("inf"), []
    return float(best_cost / len(path)), path


def _dtw_distance(a, b, is_circular=False, window=None, free_start=True, free_end=True):
    """Backward-compatible single-feature DTW using corrected path tracing."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if is_circular:
        local_cost = lambda i, j: _wrap_angle_diff_deg(a[i], b[j])
    else:
        local_cost = lambda i, j: abs(a[i] - b[j])
    return _run_dtw(
        len(a), len(b), local_cost, window=window,
        free_start=free_start, free_end=free_end,
    )


def _multivariate_dtw_distance(A, B, keys, weights, window=None, free_start=True, free_end=True):
    """Align complete poses with one shared path across all weighted features."""
    n = min(len(A[key]) for key in keys)
    m = min(len(B[key]) for key in keys)
    total_weight = sum(max(0.0, float(weights.get(key, 0.0))) for key in keys)
    if total_weight <= 0:
        return float("inf"), []

    def local_cost(i, j):
        total = 0.0
        for key in keys:
            if key in CIRCULAR_FEATURES:
                diff = _wrap_angle_diff_deg(A[key][i], B[key][j])
            else:
                diff = A[key][i] - B[key][j]
            total += float(weights.get(key, 0.0)) * _feature_norm_error(diff, key)
        return total / total_weight

    return _run_dtw(
        n, m, local_cost, window=window,
        free_start=free_start, free_end=free_end,
    )


def _normalize_cost_to_similarity(cost, feature_name):
    if feature_name in ("stance_width_norm",):
        # stance width: 0.2 ~ lệch đáng kể
        norm = min(cost / 0.2, 1.0)
    else:
        # góc: 30 độ ~ lệch đáng kể
        norm = min(cost / 30.0, 1.0)
    return 1.0 - norm

def get_weights_for_action(action_name=None):
    """
    Lấy trọng số cho một động tác bóng bàn cụ thể.
    
    Args:
        action_name: Tên động tác (str) hoặc None. Nếu None, trả về weights mặc định.
                    Các giá trị hợp lệ: "default", "table_tennis_forehand", 
                    "table_tennis_backhand", "table_tennis_serve", "table_tennis_smash",
                    "table_tennis_loop", "table_tennis_push", "table_tennis_block"
    
    Returns:
        dict: Dictionary chứa trọng số cho các feature
    """
    if action_name is None:
        action_name = "default"
    
    action_name = action_name.lower().strip()
    if action_name not in PRESET_WEIGHTS:
        # Nếu không tìm thấy, dùng default và cảnh báo
        print(f"Warning: Action '{action_name}' not found. Using 'default' weights.")
        action_name = "default"
    
    # Trả về bản copy để tránh thay đổi preset gốc
    return PRESET_WEIGHTS[action_name].copy()

def list_available_actions():
    """
    Liệt kê tất cả các preset động tác có sẵn.
    
    Returns:
        list: Danh sách tên các động tác có sẵn
    """
    return list(PRESET_WEIGHTS.keys())

def get_action_display_name(action_name):
    """
    Lấy tên hiển thị cho động tác bóng bàn (tiếng Việt).
    
    Args:
        action_name: Tên động tác (key trong PRESET_WEIGHTS)
    
    Returns:
        str: Tên hiển thị
    """
    display_names = {
        "default": "Mặc định",
        "table_tennis_forehand": "Forehand (Thuận tay)",
        "table_tennis_backhand": "Backhand (Trái tay)",
        "table_tennis_serve": "Giao bóng (Serve)",
        "table_tennis_smash": "Đập bóng (Smash)",
        "table_tennis_loop": "Vòng cung (Loop)",
        "table_tennis_push": "Đẩy bóng (Push)",
        "table_tennis_block": "Chặn bóng (Block)",
    }
    return display_names.get(action_name, action_name)

def compare_features_DTW(A, B, weights=None, action_name=None, window_ratio=0.1):
    """
    So sánh features giữa hai video sử dụng DTW.
    
    Args:
        A: Dictionary chứa features của video thứ nhất
        B: Dictionary chứa features của video thứ hai
        weights: Dictionary trọng số tùy chỉnh (nếu None, sẽ dùng action_name hoặc default)
        action_name: Tên động tác để tự động chọn preset weights (ưu tiên thấp hơn weights)
        window_ratio: Tỷ lệ window cho DTW (0.0-1.0)
    
    Returns:
        dict: Kết quả so sánh chứa per_feature_cost, per_feature_similarity, 
              weighted_score, paths, missing_features
    """
    # Xác định weights: ưu tiên weights trực tiếp, sau đó action_name, cuối cùng là default
    if weights is None:
        if action_name is not None:
            weights = get_weights_for_action(action_name)
        else:
            weights = get_weights_for_action("default")

    keys = []
    missing = []
    for k in weights.keys():
        if k in A and k in B and len(A[k]) > 0 and len(B[k]) > 0:
            keys.append(k)
        else:
            missing.append(k)

    if not keys:
        return {
            "per_feature_cost": {},
            "per_feature_similarity": {},
            "weighted_score": 0.0,
            "normalized_distance": 1.0,
            "alignment_path": [],
            "paths": {},
            "missing_features": missing,
            "algorithm_version": "multivariate-dtw-v2",
        }

    n = min(len(A[key]) for key in keys)
    m = min(len(B[key]) for key in keys)
    window = max(1, int(float(window_ratio) * max(n, m)))
    normalized_distance, alignment_path = _multivariate_dtw_distance(
        A, B, keys, weights, window=window, free_start=True, free_end=True
    )

    per_feature_cost = {}
    per_feature_sim = {}
    for key in keys:
        raw_diffs = []
        norm_diffs = []
        for i, j in alignment_path:
            if key in CIRCULAR_FEATURES:
                diff = _wrap_angle_diff_deg(A[key][i], B[key][j])
            else:
                diff = abs(A[key][i] - B[key][j])
            raw_diffs.append(float(diff))
            norm_diffs.append(_feature_norm_error(diff, key))
        per_feature_cost[key] = float(np.mean(raw_diffs)) if raw_diffs else float("inf")
        per_feature_sim[key] = float(1.0 - np.mean(norm_diffs)) if norm_diffs else 0.0

    weighted_score = 0.0
    if np.isfinite(normalized_distance):
        weighted_score = float(100.0 * max(0.0, 1.0 - normalized_distance))
    paths = {key: list(alignment_path) for key in keys}

    return {
        "per_feature_cost": per_feature_cost,
        "per_feature_similarity": per_feature_sim,
        "weighted_score": weighted_score,
        "normalized_distance": float(normalized_distance),
        "alignment_path": alignment_path,
        "paths": paths,
        "missing_features": missing,
        "algorithm_version": "multivariate-dtw-v2",
    }

def _feature_is_circular(name):
    return name in {"forearm_direction", "shoulder_line_angle", "hip_line_angle"}

def _feature_norm_error(diff, feature_name):
    if feature_name == "stance_width_norm":
        return min(abs(diff) / 0.2, 1.0)
    else:
        return min(abs(diff) / 30.0, 1.0)

def _group_runs(indices):
    if not indices: return []
    runs, s, prev = [], indices[0], indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((s, prev))
            s = prev = x
    runs.append((s, prev))
    return runs

def compute_per_frame_errors(A_feats, B_feats, paths, weights):
    # Xác định chiều dài timeline theo j
    max_j = 0   
    for p in paths.values():
        if p:
            max_j = max(max_j, max(j for _, j in p))
    T = max_j + 1
    if T <= 0:
        return {}, 0

    per_feature_err = {}
    for feat, w in weights.items():
        if feat not in A_feats or feat not in B_feats or feat not in paths:
            continue
        a_seq, b_seq, path = A_feats[feat], B_feats[feat], paths[feat]
        if not a_seq or not b_seq or not path:
            continue

        circ = _feature_is_circular(feat)
        # khởi tạo error theo j
        e_j_sum = [0.0] * T
        e_j_cnt = [0]   * T

        for (i, j) in path:
            if 0 <= i < len(a_seq) and 0 <= j < len(b_seq):
                ai, bj = a_seq[i], b_seq[j]
                if ai is None or bj is None:
                    continue
                diff = _wrap_angle_diff_deg(ai, bj) if circ else (ai - bj)
                e_norm = _feature_norm_error(diff, feat)  # 0..1
                e_j_sum[j] += e_norm
                e_j_cnt[j] += 1

        errs = []
        for j in range(T):
            errs.append(e_j_sum[j] / e_j_cnt[j] if e_j_cnt[j] > 0 else 0.0)
        per_feature_err[feat] = errs

    return per_feature_err, T

def aggregate_error(per_feature_err, weights):
    # tìm T chung
    T = 0
    for e in per_feature_err.values():
        T = max(T, len(e))
    if T == 0:
        return []

    agg = [0.0] * T
    wsum = [0.0] * T
    for feat, errs in per_feature_err.items():
        w = float(weights.get(feat, 0.0))
        if w <= 0: 
            continue
        for j in range(min(T, len(errs))):
            agg[j]  += errs[j] * w
            wsum[j] += w

    for j in range(T):
        agg[j] = (agg[j] / wsum[j]) if wsum[j] > 0 else 0.0
    return agg

def find_error_segments(err_seq, fps, threshold, min_duration_sec):
    violate_idx = [j for j, e in enumerate(err_seq) if e > threshold]
    runs = _group_runs(violate_idx)
    min_len = max(1, int(round(min_duration_sec * fps)))
    segments = []
    for s, t in runs:
        dur = t - s + 1
        if dur >= min_len:
            seg_max = max(err_seq[s:t+1]) if t+1 <= len(err_seq) else max(err_seq[s:])
            segments.append((s, t, dur, seg_max))
    return segments

def summarize_top_features(per_feature_err, fps, *,
                           error_threshold=0.35,
                           min_duration_sec=0.30,
                           top_k=3):
    results = []
    for feat, seq in per_feature_err.items():
        segs = find_error_segments(seq, fps, error_threshold, min_duration_sec)
        if not segs:
            continue
        coverage = sum(dur for _, _, dur, _ in segs)
        max_err = max(mx for *_, mx in segs)
        results.append((feat, coverage, max_err, segs))
    # xếp hạng: ưu tiên coverage rồi max_err
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return results[:top_k]

def build_feedback_messages(overall_agg_err, per_feature_err, weights, fps,
                            *,
                            overall_threshold=0.35,
                            overall_min_duration=0.50,
                            overall_min_coverage_ratio=0.10,
                            per_feature_threshold=0.35,
                            per_feature_min_duration=0.30,
                            lang="en"):
    
    T = len(overall_agg_err)
    if T == 0:
        return {
            "should_warn": False,
            "overall_segments": [],
            "coverage_ratio": 0.0,
            "messages": [],
            "feedback_messages": {},
        }

    # Tổng thể
    overall_segs = find_error_segments(overall_agg_err, fps, overall_threshold, overall_min_duration)
    total_violate = sum(d for *_, d, _ in overall_segs)
    coverage_ratio = total_violate / max(1, T)
    should_warn = bool(overall_segs) or (coverage_ratio >= overall_min_coverage_ratio)

    # Chi tiết top feature
    top_feats = summarize_top_features(
        per_feature_err, fps,
        error_threshold=per_feature_threshold,
        min_duration_sec=per_feature_min_duration,
        top_k=3
    )

    # Sinh message
    labels = FEATURE_LABELS_EN if lang == "en" else FEATURE_LABELS_VI
    msgs = []
    feedback_messages = {}
    if should_warn:
        if lang == "en":
            msgs.append("Pose alignment off – please check your form.")
        else:
            msgs.append("Tư thế chưa đúng – vui lòng chỉnh lại động tác.")

        if top_feats:
            if lang == "en":
                msgs.append("Main issues:")
            else:
                msgs.append("Các lỗi chính:")

            for feat, coverage, max_err, segs in top_feats:
                name = labels.get(feat, feat)
                name_en = FEATURE_LABELS_EN.get(feat, feat)
                name_vi = FEATURE_LABELS_VI.get(feat, feat)
                if feat == "stance_width_norm":
                    # stance là tỉ lệ → chuyển về %
                    max_pct = min(100.0, max_err * 100.0)
                    line = f"• {name}: deviation up to {max_pct:.0f}%"
                    if lang != "en":
                        line = f"• {name}: lệch tối đa khoảng {max_pct:.0f}%"
                    message_en = f"{name_en}: deviation up to {max_pct:.0f}%"
                    message_vi = f"{name_vi}: lệch tối đa khoảng {max_pct:.0f}%"
                else:
                    # góc/tuyến tính chuẩn hoá theo 30°
                    deg = min(30.0, max_err * 30.0)
                    line = f"• {name}: deviation up to ~{deg:.0f}°"
                    if lang != "en":
                        line = f"• {name}: lệch tối đa khoảng ~{deg:.0f}°"
                    message_en = f"{name_en}: deviation up to approximately {deg:.0f}°"
                    message_vi = f"{name_vi}: lệch tối đa khoảng {deg:.0f}°"
                # thêm thông tin độ dài segment dài nhất
                longest = max(segs, key=lambda x: x[2])
                dur_sec = longest[2] / max(1, fps)
                line += f" ({dur_sec:.2f}s)"
                msgs.append(line)
                feedback_messages[feat] = {
                    "message_vi": f"{message_vi} ({dur_sec:.2f}s)",
                    "message_en": f"{message_en} ({dur_sec:.2f}s)",
                    "max_error": float(max_err),
                    "duration_seconds": float(dur_sec),
                }

    return {
        "should_warn": should_warn,
        "overall_segments": overall_segs,
        "coverage_ratio": coverage_ratio,
        "messages": msgs,
        "feedback_messages": feedback_messages,
    }


def compare_videos(
    reference_video,
    student_video,
    *,
    n_points=100,
    weights=None,
    action_name=None,
    use_3d=True,
    lang="en",
    min_visibility=0.5,
    progress_callback=None,
):
    """Canonical comparison pipeline shared by desktop and API callers."""
    notify = progress_callback if callable(progress_callback) else (lambda _message: None)
    notify("Extracting reference pose...")
    feats_reference = extract_features(
        reference_video, draw=False, use_3d=use_3d,
        min_visibility=min_visibility,
    )
    notify("Extracting student pose...")
    feats_student = extract_features(
        student_video, draw=False, use_3d=use_3d,
        min_visibility=min_visibility,
    )

    for label, feats in (("reference", feats_reference), ("student", feats_student)):
        if int(feats.get("valid_frames", 0)) < 10:
            raise ValueError(
                f"Too few reliable pose frames in {label} video: "
                f"{int(feats.get('valid_frames', 0))}"
            )

    notify("Synchronizing pose sequences...")
    reference_resampled = resample_features(feats_reference, n=n_points)
    student_resampled = resample_features(feats_student, n=n_points)
    actual_weights = weights if weights is not None else get_weights_for_action(action_name)
    result = compare_features_DTW(
        reference_resampled,
        student_resampled,
        weights=actual_weights,
        action_name=action_name,
        window_ratio=0.1,
    )
    if not result.get("alignment_path"):
        raise ValueError("Pose sequences could not be synchronized")

    per_feature_err, _ = compute_per_frame_errors(
        reference_resampled,
        student_resampled,
        result["paths"],
        actual_weights,
    )
    aggregate_err = aggregate_error(per_feature_err, actual_weights)
    effective_fps = get_effective_fps(student_resampled)
    feedback = build_feedback_messages(
        aggregate_err,
        per_feature_err,
        actual_weights,
        fps=effective_fps,
        overall_threshold=0.35,
        overall_min_duration=0.50,
        overall_min_coverage_ratio=0.10,
        per_feature_threshold=0.35,
        per_feature_min_duration=0.30,
        lang=lang,
    )
    per_feature_error_mean = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in per_feature_err.items()
    }
    quality_warnings = []
    for label, feats in (("reference", feats_reference), ("student", feats_student)):
        if float(feats.get("valid_ratio", 0.0)) < 0.5:
            quality_warnings.append(
                f"Low pose visibility in {label} video "
                f"({float(feats.get('valid_ratio', 0.0)):.0%} valid frames)"
            )

    result.update({
        "overall_score": float(result["weighted_score"]),
        "dtw_distance": float(result["normalized_distance"]),
        "per_feature_errors": per_feature_err,
        "per_feature_error_mean": per_feature_error_mean,
        "aggregate_error": aggregate_err,
        "feedback": feedback,
        "error_messages": feedback["messages"] if feedback.get("should_warn") else [],
        "effective_fps": float(effective_fps),
        "student_timeline_seconds": student_resampled.get("timestamps", []),
        "quality": {
            "reference_valid_frames": int(feats_reference.get("valid_frames", 0)),
            "reference_total_frames": int(feats_reference.get("total_frames", 0)),
            "reference_valid_ratio": float(feats_reference.get("valid_ratio", 0.0)),
            "student_valid_frames": int(feats_student.get("valid_frames", 0)),
            "student_total_frames": int(feats_student.get("total_frames", 0)),
            "student_valid_ratio": float(feats_student.get("valid_ratio", 0.0)),
            "warnings": quality_warnings,
        },
    })
    notify("Pose comparison complete.")
    return result



