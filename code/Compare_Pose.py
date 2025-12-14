#   Vai trái=11, vai phải=12, Khuỷu trái=13, Khuỷu phải=14,
#   Cổ tay trái=15, Cổ tay phải=16, Hông trái=23, Hông phải=24,
#   Gối trái=25, Gối phải=26, Cổ chân trái=27, Cổ chân phải=28

import cv2 as cv
import PoseModule as pm
import math
import numpy as np

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

def extract_features(video_path, draw=False, smooth=False):
    cap = cv.VideoCapture(video_path)
    detector = pm.poseDetector()
    feats = {
        "right_elbow_angle": [],
        "forearm_direction": [],
        "shoulder_line_angle": [],
        "hip_line_angle": [],
        "torso_twist_abs": [],
        "right_knee_angle": [],
        "left_knee_angle": [],
        "stance_width_norm": [],
    }
    total_frames = 0
    valid_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        frame_proc = detector.findPose(frame, draw=draw)
        lmList = detector.findPosition(frame_proc, draw=False)
        if not lmList or len(lmList) < 29:
            continue
        pts = _extract_xy_from_lmList(lmList)
        pts_norm = _normalize_landmarks(pts)
        if pts_norm is None:
            continue
        f = _compute_frame_features(pts_norm)
        if f is None:
            continue
        for k in feats.keys():
            feats[k].append(float(f[k]))
        valid_frames += 1
    cap.release()
    feats["valid_frames"] = valid_frames
    feats["total_frames"] = total_frames
    return feats

    
def resample_features(feats, n=100):
    out = {}
    for k, v in feats.items():
        if k in ("valid_frames", "total_frames"):
            continue
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            continue
        x_old = np.linspace(0, 1, num=len(v))
        x_new = np.linspace(0, 1, num=n)
        y_new = np.interp(x_new, x_old, np.array(v, dtype=float))
        out[k] = list(map(float, y_new))
    return out

def _dtw_distance(a, b, is_circular=False, window=None, free_start=True, free_end=True):
    """
    Tính khoảng cách DTW giữa hai chuỗi với hỗ trợ free start/end.
    
    Args:
        a, b: Các chuỗi cần so sánh
        is_circular: True nếu feature là góc tuần hoàn
        window: Kích thước window (None = không giới hạn)
        free_start: True để cho phép bắt đầu từ bất kỳ điểm nào (xử lý video không cùng thời điểm xuất phát)
        free_end: True để tìm điểm kết thúc tốt nhất (xử lý video có độ dài khác nhau)
    
    Returns:
        (avg_cost, path): Chi phí trung bình và đường đi alignment
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), []

    if window is None:
        window = max(n, m)  # không ràng buộc
    window = int(window)

    # Ma trận chi phí tích luỹ
    C = np.full((n + 1, m + 1), np.inf, dtype=float)
    
    # Khởi tạo điểm bắt đầu
    if free_start:
        # Cho phép bắt đầu từ bất kỳ điểm nào trên hàng đầu hoặc cột đầu
        # Giới hạn trong một vùng hợp lý để tránh quá tốn kém
        start_region = min(10, max(1, int(0.1 * min(n, m))))  # 10% hoặc tối đa 10 điểm
        for i in range(min(start_region, n + 1)):
            C[i, 0] = 0.0
        for j in range(min(start_region, m + 1)):
            C[0, j] = 0.0
    else:
        # Bắt đầu cố định từ (0, 0)
        C[0, 0] = 0.0

    # Hàm khoảng cách phần tử
    if is_circular:
        def dfun(x, y): return _wrap_angle_diff_deg(x, y)
    else:
        def dfun(x, y): return abs(x - y)

    # Tính toán ma trận chi phí
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end   = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = dfun(a[i - 1], b[j - 1])
            # Chỉ tính nếu có điểm trước đó hợp lệ (không phải inf)
            prev_costs = []
            if C[i - 1, j] < np.inf:
                prev_costs.append(C[i - 1, j])      # bước dọc (1,0)
            if C[i, j - 1] < np.inf:
                prev_costs.append(C[i, j - 1])      # bước ngang (0,1)
            if C[i - 1, j - 1] < np.inf:
                prev_costs.append(C[i - 1, j - 1])  # bước chéo (1,1)
            
            if prev_costs:
                C[i, j] = cost + min(prev_costs)
            # Nếu không có điểm trước hợp lệ và không phải free_start, giữ nguyên inf

    # Tìm điểm kết thúc tốt nhất
    if free_end:
        # Tìm điểm có chi phí thấp nhất trong vùng kết thúc
        end_region = min(10, max(1, int(0.1 * min(n, m))))  # 10% hoặc tối đa 10 điểm
        best_cost = np.inf
        best_i, best_j = n, m
        
        # Tìm trong hàng cuối
        for j in range(max(1, m - end_region), m + 1):
            if C[n, j] < best_cost:
                best_cost = C[n, j]
                best_i, best_j = n, j
        
        # Tìm trong cột cuối
        for i in range(max(1, n - end_region), n + 1):
            if C[i, m] < best_cost:
                best_cost = C[i, m]
                best_i, best_j = i, m
        
        if best_cost == np.inf:
            # Fallback về điểm cuối nếu không tìm thấy
            best_i, best_j = n, m
            best_cost = C[n, m]
    else:
        # Kết thúc cố định ở (n, m)
        best_i, best_j = n, m
        best_cost = C[n, m]

    # Truy vết đường đi từ điểm kết thúc tốt nhất
    i, j = best_i, best_j
    path = []
    
    # Truy vết ngược lại cho đến khi gặp điểm bắt đầu (cost = 0)
    while True:
        # Thêm điểm hiện tại vào path (chuyển từ index ma trận sang index mảng)
        if i > 0 and j > 0:
            path.append((i - 1, j - 1))
        elif i > 0:
            path.append((i - 1, 0))
        elif j > 0:
            path.append((0, j - 1))
        
        # Dừng nếu đã đến điểm bắt đầu (cost = 0)
        if C[i, j] == 0.0:
            break
        
        # Chọn bước đã tạo giá trị nhỏ nhất
        prevs = []
        if i > 0 and j > 0 and C[i - 1, j - 1] < np.inf:
            prevs.append((i - 1, j - 1, C[i - 1, j - 1]))  # bước chéo (1,1) - ưu tiên
        if i > 0 and C[i - 1, j] < np.inf:
            prevs.append((i - 1, j, C[i - 1, j]))      # bước dọc (1,0)
        if j > 0 and C[i, j - 1] < np.inf:
            prevs.append((i, j - 1, C[i, j - 1]))      # bước ngang (0,1)
        
        if not prevs:
            break
        
        i2, j2, _ = min(prevs, key=lambda x: x[2])
        i, j = i2, j2
    
    path.reverse()

    # Chuẩn hoá theo độ dài đường đi để so sánh công bằng các cặp có n≠m
    avg_cost = float(best_cost / max(1, len(path))) if len(path) > 0 else float("inf")
    return avg_cost, path


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

    # Xác định feature là góc tuần hoàn
    circular_feats = {"forearm_direction", "shoulder_line_angle", "hip_line_angle"}

    keys = []
    missing = []
    for k in weights.keys():
        if k in A and k in B and len(A[k]) > 0 and len(B[k]) > 0:
            keys.append(k)
        else:
            missing.append(k)

    per_feature_cost = {}
    per_feature_sim  = {}
    paths = {}
    total_w = 0.0
    total_score = 0.0

    for k in keys:
        a, b = A[k], B[k]
        L = max(len(a), len(b))
        window = max(1, int(window_ratio * L))

        is_circ = k in circular_feats
        cost, path = _dtw_distance(a, b, is_circular=is_circ, window=window)
        sim = _normalize_cost_to_similarity(cost, k)

        per_feature_cost[k] = float(cost)
        per_feature_sim[k]  = float(sim)
        paths[k] = path

        w = float(weights[k])
        total_w += w
        total_score += sim * w

    weighted_score = float(100.0 * total_score / total_w) if total_w > 0 else 0.0

    return {
        "per_feature_cost": per_feature_cost,
        "per_feature_similarity": per_feature_sim,
        "weighted_score": weighted_score,
        "paths": paths,
        "missing_features": missing
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
        return {"should_warn": False, "overall_segments": [], "coverage_ratio": 0.0, "messages": []}

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
                if feat == "stance_width_norm":
                    # stance là tỉ lệ → chuyển về %
                    max_pct = min(100.0, max_err * 100.0)
                    line = f"• {name}: deviation up to {max_pct:.0f}%"
                    if lang != "en":
                        line = f"• {name}: lệch tối đa khoảng {max_pct:.0f}%"
                else:
                    # góc/tuyến tính chuẩn hoá theo 30°
                    deg = min(30.0, max_err * 30.0)
                    line = f"• {name}: deviation up to ~{deg:.0f}°"
                    if lang != "en":
                        line = f"• {name}: lệch tối đa khoảng ~{deg:.0f}°"
                # thêm thông tin độ dài segment dài nhất
                longest = max(segs, key=lambda x: x[2])
                dur_sec = longest[2] / max(1, fps)
                line += f" ({dur_sec:.2f}s)"
                msgs.append(line)

    return {
        "should_warn": should_warn,
        "overall_segments": overall_segs,
        "coverage_ratio": coverage_ratio,
        "messages": msgs
    }



