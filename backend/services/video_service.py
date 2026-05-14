"""
Video Service - Quản lý video hướng dẫn động tác (chỉ cho developer)
"""

import os
import json
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

class VideoMetadata(BaseModel):
    """Metadata cho một video hướng dẫn"""
    id: str
    title: str
    description: str
    technique: str  # Tên động tác (forehand, backhand, serve, etc.)
    difficulty: str  # beginner, intermediate, advanced
    duration: Optional[int] = None  # Thời lượng (giây)
    file_path: str
    tags: List[str] = []  # Tags để tìm kiếm
    created_at: datetime

class VideoService:
    """Service quản lý video hướng dẫn"""

    def __init__(self, video_dir: str = "video/", metadata_file: str = "video/metadata.json"):
        self.video_dir = video_dir
        self.metadata_file = metadata_file
        self.videos: Dict[str, VideoMetadata] = {}

        # Tạo thư mục nếu chưa có
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata từ file JSON"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for video_data in data.get('videos', []):
                        video = VideoMetadata(**video_data)
                        self.videos[video.id] = video
            except Exception as e:
                print(f"Lỗi load metadata: {e}")

    def _save_metadata(self):
        """Lưu metadata vào file JSON"""
        data = {
            'videos': [video.model_dump() for video in self.videos.values()],
            'last_updated': datetime.now().isoformat()
        }
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def add_video_manual(self, video_id: str, title: str, description: str,
                        technique: str, difficulty: str, file_path: str, tags: List[str] = None) -> VideoMetadata:
        """Thêm video thủ công (cho developer)"""

        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File video không tồn tại: {file_path}")

        # Tạo metadata
        video = VideoMetadata(
            id=video_id,
            title=title,
            description=description,
            technique=technique,
            difficulty=difficulty,
            file_path=file_path,
            tags=tags or [],
            created_at=datetime.now()
        )

        self.videos[video_id] = video
        self._save_metadata()

        return video

    def search_videos(self, query: str, technique: str = None,
                     difficulty: str = None, limit: int = 5) -> List[VideoMetadata]:
        """Tìm kiếm video dựa trên query"""

        results = []
        query_lower = query.lower()

        for video in self.videos.values():
            # Tìm trong title, description, technique, tags
            searchable_text = f"{video.title} {video.description} {video.technique} {' '.join(video.tags)}".lower()

            # Kiểm tra technique filter
            if technique and video.technique.lower() != technique.lower():
                continue

            # Kiểm tra difficulty filter
            if difficulty and video.difficulty.lower() != difficulty.lower():
                continue

            # Tìm kiếm text
            if query_lower in searchable_text:
                results.append(video)

        # Sắp xếp theo relevance (ưu tiên technique match)
        results.sort(key=lambda x: x.technique.lower() == query_lower, reverse=True)

        return results[:limit]

    def get_video_by_id(self, video_id: str) -> Optional[VideoMetadata]:
        """Lấy video theo ID"""
        return self.videos.get(video_id)

    def get_all_videos(self) -> List[VideoMetadata]:
        """Lấy tất cả video"""
        return list(self.videos.values())

    def get_videos_by_technique(self, technique: str) -> List[VideoMetadata]:
        """Lấy video theo động tác"""
        return [v for v in self.videos.values() if v.technique.lower() == technique.lower()]

    def delete_video(self, video_id: str) -> bool:
        """Xóa video"""
        if video_id in self.videos:
            video = self.videos[video_id]

            # Xóa file vật lý nếu muốn
            # if os.path.exists(video.file_path):
            #     os.remove(video.file_path)

            # Xóa metadata
            del self.videos[video_id]
            self._save_metadata()

            return True
        return False

# Global instance
_video_service = None

def get_video_service() -> VideoService:
    """Lấy video service instance"""
    global _video_service
    if _video_service is None:
        # Dùng đường dẫn tuyệt đối để tránh working directory issue
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        video_dir = os.path.join(project_root, "video")
        metadata_file = os.path.join(video_dir, "metadata.json")
        _video_service = VideoService(video_dir=video_dir, metadata_file=metadata_file)
    return _video_service