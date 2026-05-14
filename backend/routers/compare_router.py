"""
Pose Comparison Router - So sánh poses bằng DTW
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys
from datetime import datetime
from enum import Enum

router = APIRouter()

class ActionType(str, Enum):
    """Các loại động tác bóng bàn"""
    DEFAULT = "default"
    FOREHAND = "table_tennis_forehand"
    BACKHAND = "table_tennis_backhand"
    SERVE = "table_tennis_serve"
    SMASH = "table_tennis_smash"


class FeatureError(BaseModel):
    """Lỗi của từng feature"""
    feature_name: str
    error_value: float
    should_correct: bool
    feedback_vi: str
    feedback_en: str


class ComparisonResponse(BaseModel):
    """Response model cho pose comparison"""
    success: bool
    message: str
    overall_score: Optional[float] = None  # 0-100
    similarity_percentage: Optional[float] = None
    dtw_distance: Optional[float] = None
    per_feature_errors: Optional[List[FeatureError]] = None
    recommendations: Optional[List[str]] = None


@router.post("/compare", response_model=ComparisonResponse)
async def compare_poses(
    reference_video: UploadFile = File(...),
    student_video: UploadFile = File(...),
    action: ActionType = ActionType.DEFAULT,
    use_3d: bool = True,
    n_points: int = 100
):
    """
    So sánh 2 video tư thế bằng DTW algorithm
    
    Args:
        reference_video: Video của người hướng dẫn (tư thế đúng)
        student_video: Video của học viên (tư thế cần assessment)
        action: Loại động tác (default, forehand, backhand, serve, smash)
        use_3d: Sử dụng 3D landmarks
        n_points: Số điểm để resampling (càng nhiều càng chi tiết)
    
    Returns:
        JSON với điểm số DTW, lỗi chi tiết, và khuyến nghị
    """
    
    try:
        upload_dir = "backend/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Lưu 2 files
        timestamp = datetime.now().timestamp()
        ref_temp = f"{upload_dir}/ref_{timestamp}.mp4"
        stu_temp = f"{upload_dir}/stu_{timestamp}.mp4"
        
        # Lưu reference video
        with open(ref_temp, "wb") as f:
            f.write(await reference_video.read())
        
        # Lưu student video
        with open(stu_temp, "wb") as f:
            f.write(await student_video.read())
        
        # Import modules
        CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code"))
        if CODE_DIR not in sys.path:
            sys.path.insert(0, CODE_DIR)
        import Compare_Pose as CP
        
        # Extract features từ cả 2 video
        featsA = CP.extract_features(ref_temp, draw=False, use_3d=use_3d)
        featsB = CP.extract_features(stu_temp, draw=False, use_3d=use_3d)
        
        # Resample features
        A_rs = CP.resample_features(featsA, n=n_points)
        B_rs = CP.resample_features(featsB, n=n_points)
        
        # So sánh bằng DTW
        result = CP.compare_features_DTW(
            A_rs, B_rs, 
            weights=None, 
            action_name=action.value,
            window_ratio=0.1
        )
        
        # Tính lỗi chi tiết per feature
        actual_weights = CP.get_weights_for_action(action.value)
        per_feature_err, T = CP.compute_per_frame_errors(
            A_rs, B_rs, 
            result["paths"], 
            actual_weights
        )
        
        # Tổng hợp lỗi
        agg_err = CP.aggregate_error(per_feature_err, actual_weights)
        
        # Tạo feedback messages
        feedback = CP.build_feedback_messages(
            agg_err, per_feature_err, actual_weights, fps=30,
            overall_threshold=0.35,
            overall_min_duration=0.50,
            overall_min_coverage_ratio=0.10,
            per_feature_threshold=0.35,
            per_feature_min_duration=0.30,
            lang="vi"
        )
        
        # Chuyển đổi sang response model
        feature_errors = []
        for feature_name, error_val in per_feature_err.items():
            if isinstance(error_val, list):
                if len(error_val) > 0:
                    average_error = sum(error_val) / len(error_val)
                else:
                    average_error = 0.0
            else:
                average_error = error_val

            error_float = float(average_error)
            feature_errors.append(FeatureError(
                feature_name=feature_name,
                error_value=error_float,
                should_correct=error_float > 0.35,
                feedback_vi=feedback.get("feedback_messages", {}).get(feature_name, {}).get("message_vi", ""),
                feedback_en=feedback.get("feedback_messages", {}).get(feature_name, {}).get("message_en", "")
            ))
        
        # Tính similarity percentage từ weighted_score
        # weighted_score từ compare_features_DTW() đã là 0-100
        similarity_percentage = float(result.get("weighted_score", 0))
        
        # Tính DTW distance từ per_feature_cost (weighted average)
        per_feature_cost = result.get("per_feature_cost", {})
        if per_feature_cost:
            total_cost = sum(float(v) for v in per_feature_cost.values())
            dtw_distance = total_cost / len(per_feature_cost) if len(per_feature_cost) > 0 else 0.0
        else:
            dtw_distance = 0.0
        
        # overall_score dựa trên average error từ all features
        if feature_errors:
            avg_error = sum(fe.error_value for fe in feature_errors) / len(feature_errors)
            overall_score = max(0, 100 - (avg_error * 100))  # Error to score
        else:
            overall_score = similarity_percentage
        
        os.remove(ref_temp)
        os.remove(stu_temp)
        
        return ComparisonResponse(
            success=True,
            message="So sánh thành công",
            overall_score=overall_score,
            similarity_percentage=similarity_percentage,
            dtw_distance=dtw_distance,
            per_feature_errors=feature_errors,
            recommendations=feedback.get("messages", [])
        )
        
    except Exception as e:
        # Cleanup
        if 'ref_temp' in locals() and os.path.exists(ref_temp):
            os.remove(ref_temp)
        if 'stu_temp' in locals() and os.path.exists(stu_temp):
            os.remove(stu_temp)
        
        return ComparisonResponse(
            success=False,
            message=f"Lỗi: {str(e)}"
        )


@router.get("/actions")
async def get_available_actions():
    """Lấy danh sách các động tác có sẵn"""
    return {
        "actions": [
            {
                "id": "default",
                "name": "Mặc định",
                "description": "Assessment chuẩn"
            },
            {
                "id": "table_tennis_forehand",
                "name": "Cơ bóng thuận",
                "description": "Đánh thuận tay"
            },
            {
                "id": "table_tennis_backhand",
                "name": "Cơ bóng nghịch",
                "description": "Đánh nghịch tay"
            },
            {
                "id": "table_tennis_serve",
                "name": "Phát bóng",
                "description": "Phát bóng mở đầu"
            },
            {
                "id": "table_tennis_smash",
                "name": "Tấn công",
                "description": "Đánh tấn công mạnh"
            }
        ]
    }
