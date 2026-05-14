"""
Video Stream Router - API endpoints cho streaming video hướng dẫn
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Dict, Any
from services.video_service import get_video_service

router = APIRouter()

@router.get("/stream/{video_id}")
async def stream_video(video_id: str):
    """Stream video file"""
    try:
        import os
        video_service = get_video_service()
        video = video_service.get_video_by_id(video_id)

        if not video:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Kiểm tra file tồn tại - xử lý cả đường dẫn tương đối và tuyệt đối
        file_path = video.file_path
        if not os.path.isabs(file_path):
            # Nếu là đường dẫn tương đối, chuyển thành tuyệt đối từ project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            file_path = os.path.join(project_root, file_path)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File video không tồn tại: {file_path}")

        # Trả về file với content-type phù hợp
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=f"{video.title}.mp4"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi stream video: {str(e)}")

@router.get("/info/{video_id}")
async def get_video_info(video_id: str) -> Dict[str, Any]:
    """Lấy thông tin video"""
    try:
        video_service = get_video_service()
        video = video_service.get_video_by_id(video_id)

        if not video:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Kiểm tra file tồn tại
        import os
        file_exists = os.path.exists(video.file_path)

        return {
            "id": video.id,
            "title": video.title,
            "description": video.description,
            "technique": video.technique,
            "difficulty": video.difficulty,
            "duration": video.duration,
            "tags": video.tags,
            "file_exists": file_exists,
            "stream_url": f"/api/video/stream/{video.id}",
            "created_at": video.created_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thông tin video: {str(e)}")

@router.get("/search")
async def search_videos(query: str = "", technique: str = None, difficulty: str = None, limit: int = 10):
    """Tìm kiếm video"""
    try:
        video_service = get_video_service()
        videos = video_service.search_videos(
            query=query,
            technique=technique,
            difficulty=difficulty,
            limit=limit
        )

        results = []
        for video in videos:
            import os
            file_exists = os.path.exists(video.file_path)

            results.append({
                "id": video.id,
                "title": video.title,
                "description": video.description,
                "technique": video.technique,
                "difficulty": video.difficulty,
                "tags": video.tags,
                "file_exists": file_exists,
                "stream_url": f"/api/video/stream/{video.id}"
            })

        return {
            "query": query,
            "total_results": len(results),
            "videos": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm video: {str(e)}")