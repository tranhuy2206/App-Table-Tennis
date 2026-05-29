"""
Video Stream Router - API endpoints cho streaming video hướng dẫn
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, Any
from services.video_service import get_video_service
import os

router = APIRouter()

@router.get("/stream/{video_id}")
async def stream_video(video_id: str, request: Request):
    """Stream video file với hỗ trợ Range Request (cho Android VideoView)"""
    try:
        video_service = get_video_service()
        video = video_service.get_video_by_id(video_id)

        if not video:
            raise HTTPException(status_code=404, detail="Video không tồn tại")

        # Kiểm tra file tồn tại - xử lý cả đường dẫn tương đối và tuyệt đối
        file_path = video.file_path
        if not os.path.isabs(file_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            file_path = os.path.join(project_root, file_path)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File video không tồn tại: {file_path}")

        file_size = os.path.getsize(file_path)
        
        # Xử lý Range Request (Android VideoView yêu cầu)
        range_header = request.headers.get("range")
        
        if range_header:
            # Parse range: "bytes=0-1023" hoặc "bytes=0-"
            range_str = range_header.replace("bytes=", "")
            parts = range_str.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            
            # Giới hạn end không vượt quá file size
            end = min(end, file_size - 1)
            content_length = end - start + 1
            
            def iter_file():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(8192, remaining)
                        data = f.read(chunk_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
            
            return StreamingResponse(
                iter_file(),
                status_code=206,
                media_type="video/mp4",
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                },
            )
        else:
            # Không có Range header → trả về toàn bộ file + Accept-Ranges
            return FileResponse(
                path=file_path,
                media_type="video/mp4",
                filename=f"{video.title}.mp4",
                headers={"Accept-Ranges": "bytes"},
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