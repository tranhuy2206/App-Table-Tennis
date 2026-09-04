"""
Ball Tracking Router - Đếm quả bóng bàn
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os
import cv2
import sys
from datetime import datetime

router = APIRouter()

class BallTrackingResponse(BaseModel):
    """Response model cho ball tracking"""
    success: bool
    message: str
    ball_count: int
    frame_count: int
    video_duration_seconds: Optional[float] = None
    diagnostics: Optional[Dict[str, Any]] = None


@router.post("/count", response_model=BallTrackingResponse)
async def count_balls(
    file: UploadFile = File(...),
    frame_w: int = 640,
    frame_h: int = 480,
    debug: bool = False,
):
    """
    Đếm quả bóng bàn qua lưới trong video
    
    Args:
        file: Video file
        frame_w: Chiều rộng frame để xử lý (càng nhỏ càng nhanh)
        frame_h: Chiều cao frame
    
    Returns:
        JSON với số lần bóng qua lưới
    """
    
    try:
        # Lưu file tạm
        upload_dir = "backend/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        temp_filename = f"{upload_dir}/temp_ball_{datetime.now().timestamp()}.mp4"
        
        with open(temp_filename, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Mở video
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            os.remove(temp_filename)
            raise HTTPException(status_code=400, detail="Không thể mở video")
        
        # Import BallProcessor
        CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code"))
        if CODE_DIR not in sys.path:
            sys.path.insert(0, CODE_DIR)
        from processor_ball import BallProcessor
        
        processor = BallProcessor(
            frame_w=frame_w,
            frame_h=frame_h,
            debug_diagnostics=debug,
        )
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Xử lý frame
            processor.process(frame)
            frame_count += 1
        
        # Lấy số đếm cuối cùng
        ball_count = processor.ball_count
        diagnostics = processor.get_diagnostics() if debug else None
        if debug:
            print(processor._format_diagnostics_summary())
        
        # Lấy FPS để tính duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        os.remove(temp_filename)
        
        return BallTrackingResponse(
            success=True,
            message=f"Đếm thành công bóng qua lưới",
            ball_count=ball_count,
            frame_count=frame_count,
            video_duration_seconds=duration,
            diagnostics=diagnostics,
        )
        
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return BallTrackingResponse(
            success=False,
            message=f"Lỗi: {str(e)}",
            ball_count=0,
            frame_count=0,
            diagnostics=None,
        )


@router.get("/info")
async def get_ball_tracking_info():
    """Thông tin về ball tracking"""
    return {
        "algorithm": "Automatic table ROI + MOG2/color filtering + Kalman tracking",
        "features": [
            "Đếm quả bóng qua lưới",
            "Hỗ trợ nhiều loại video"
        ],
        "parameters": {
            "frame_w": "Chiều rộng frame để xử lý",
            "frame_h": "Chiều cao frame"
        }
    }
