"""
Pose Comparison Router - So sánh poses bằng DTW
"""

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
from enum import Enum
from uuid import uuid4

router = APIRouter()

class ActionType(str, Enum):
    """Các loại động tác bóng bàn"""
    DEFAULT = "default"
    FOREHAND = "table_tennis_forehand"
    BACKHAND = "table_tennis_backhand"
    SERVE = "table_tennis_serve"
    SMASH = "table_tennis_smash"
    LOOP = "table_tennis_loop"
    PUSH = "table_tennis_push"
    BLOCK = "table_tennis_block"


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
        request_id = uuid4().hex
        ref_temp = f"{upload_dir}/ref_{request_id}.mp4"
        stu_temp = f"{upload_dir}/stu_{request_id}.mp4"
        
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
        
        result = CP.compare_videos(
            ref_temp,
            stu_temp,
            n_points=n_points,
            action_name=action.value,
            use_3d=use_3d,
            lang="vi",
        )

        feedback = result.get("feedback", {})
        feedback_by_feature = feedback.get("feedback_messages", {})
        feature_errors = []
        for feature_name, error_value in result.get("per_feature_error_mean", {}).items():
            error_float = float(error_value)
            feature_feedback = feedback_by_feature.get(feature_name, {})
            feature_errors.append(FeatureError(
                feature_name=feature_name,
                error_value=error_float,
                should_correct=feature_name in feedback_by_feature or error_float > 0.35,
                feedback_vi=feature_feedback.get("message_vi", ""),
                feedback_en=feature_feedback.get("message_en", ""),
            ))

        overall_score = float(result.get("overall_score", 0.0))
        quality_warnings = result.get("quality", {}).get("warnings", [])
        recommendations = list(feedback.get("messages", [])) + list(quality_warnings)
        
        os.remove(ref_temp)
        os.remove(stu_temp)
        
        return ComparisonResponse(
            success=True,
            message="So sánh thành công",
            overall_score=overall_score,
            similarity_percentage=overall_score,
            dtw_distance=float(result.get("dtw_distance", 1.0)),
            per_feature_errors=feature_errors,
            recommendations=recommendations,
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
            },
            {
                "id": "table_tennis_loop",
                "name": "Loop",
                "description": "Đánh bóng xoáy lên theo vòng cung"
            },
            {
                "id": "table_tennis_push",
                "name": "Push",
                "description": "Đẩy bóng ngắn"
            },
            {
                "id": "table_tennis_block",
                "name": "Block",
                "description": "Chặn bóng"
            }
        ]
    }
