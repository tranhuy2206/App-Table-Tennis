"""
Pose Detection Router - Nhận diện tư thế từ video
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import cv2
import numpy as np
import sys
from datetime import datetime
import json

router = APIRouter()

class PoseLandmark(BaseModel):
    """Model cho một landmark"""
    id: int
    x: float
    y: float
    z: Optional[float] = None
    visibility: Optional[float] = None
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    world_z: Optional[float] = None

class PoseDetectionResponse(BaseModel):
    """Response model cho pose detection"""
    success: bool
    message: str
    frame_count: int
    total_frame_count: Optional[int] = None
    processed_frame_count: Optional[int] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    processed_frame_indices: Optional[List[int]] = None
    processed_timestamps_ms: Optional[List[float]] = None
    pose_connections: Optional[List[List[int]]] = None
    landmarks_per_frame: Optional[List[List[PoseLandmark]]] = None
    first_frame_landmarks: Optional[List[PoseLandmark]] = None


@router.post("/detect", response_model=PoseDetectionResponse)
async def detect_pose(
    file: UploadFile = File(...),
    use_3d: bool = True,
    sample_rate: int = 1  # Lấy 1 frame từ mỗi N frame (tiết kiệm processing)
):
    """
    Nhận diện tư thế từ video
    
    Args:
        file: Video file (mp4, avi, mkv, mov)
        use_3d: Sử dụng 3D landmarks (world coordinates)
        sample_rate: Lấy mẫu mỗi N frame (1 = tất cả frame, 5 = cứ 5 frame lấy 1)
    
    Returns:
        JSON với landmarks của từng frame
    """
    
    try:
        sample_rate = max(1, sample_rate)

        # Lưu file tạm thời
        upload_dir = "backend/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Tên file tạm
        temp_filename = f"{upload_dir}/temp_{datetime.now().timestamp()}.mp4"
        
        # Lưu file
        with open(temp_filename, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Mở video
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            os.remove(temp_filename)
            raise HTTPException(status_code=400, detail="Không thể mở video")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = (
            float(total_frame_count / fps)
            if fps > 0 and total_frame_count > 0
            else None
        )
        
        # Import PoseModule từ code folder
        CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code"))
        if CODE_DIR not in sys.path:
            sys.path.insert(0, CODE_DIR)
        import PoseModule as pm
        
        detector = pm.poseDetector()
        pose_connections = [
            [int(start), int(end)]
            for start, end in sorted(
                detector.mpPose.POSE_CONNECTIONS,
                key=lambda conn: (int(conn[0]), int(conn[1]))
            )
        ]
        frame_count = 0
        first_frame_landmarks = None
        all_landmarks = []
        processed_frame_indices = []
        processed_timestamps_ms = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Lấy mẫu cứ mỗi sample_rate frame
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
            
            # Nhận diện pose
            detector.findPose(frame, draw=False)
            lm_list = detector.findPosition(frame, draw=False, use_3d=use_3d)

            image_landmarks = (
                detector.results.pose_landmarks.landmark
                if getattr(detector, "results", None) and detector.results.pose_landmarks
                else []
            )
            
            # Chuyển đổi sang model
            landmarks = []
            for lm in lm_list:
                lm_id = int(lm[0])
                image_lm = image_landmarks[lm_id] if lm_id < len(image_landmarks) else None
                z = float(image_lm.z) if image_lm is not None else None
                visibility = float(image_lm.visibility) if image_lm is not None else None

                if use_3d and len(lm) >= 6:  # [id, x, y, wx, wy, wz]
                    landmarks.append(PoseLandmark(
                        id=lm_id,
                        x=float(lm[1]),
                        y=float(lm[2]),
                        z=z,
                        visibility=visibility,
                        world_x=float(lm[3]),
                        world_y=float(lm[4]),
                        world_z=float(lm[5])
                    ))
                else:  # 2D [id, x, y]
                    landmarks.append(PoseLandmark(
                        id=lm_id,
                        x=float(lm[1]),
                        y=float(lm[2]),
                        z=z,
                        visibility=visibility
                    ))
            
            all_landmarks.append(landmarks)
            processed_frame_indices.append(frame_count)
            processed_timestamps_ms.append(
                float((frame_count / fps) * 1000.0) if fps > 0 else 0.0
            )
            
            if first_frame_landmarks is None:
                first_frame_landmarks = landmarks
            
            frame_count += 1
        
        cap.release()
        os.remove(temp_filename)
        
        return PoseDetectionResponse(
            success=True,
            message=f"Nhận diện thành công {frame_count} frame",
            frame_count=frame_count,
            total_frame_count=total_frame_count or frame_count,
            processed_frame_count=len(all_landmarks),
            video_width=video_width,
            video_height=video_height,
            fps=fps,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            processed_frame_indices=processed_frame_indices,
            processed_timestamps_ms=processed_timestamps_ms,
            pose_connections=pose_connections,
            landmarks_per_frame=all_landmarks,
            first_frame_landmarks=first_frame_landmarks
        )
        
    except Exception as e:
        return PoseDetectionResponse(
            success=False,
            message=f"Lỗi: {str(e)}",
            frame_count=0
        )


@router.get("/models")
async def get_available_models():
    """Lấy danh sách model có sẵn"""
    return {
        "models": [
            {
                "name": "MediaPipe Pose",
                "complexity": "standard",
                "description": "Phát hiện 33 landmarks trên cơ thể"
            }
        ]
    }
