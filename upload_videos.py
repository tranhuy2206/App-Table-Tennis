#!/usr/bin/env python3
"""
Upload Video Script - Upload video hướng dẫn vào database
"""

import os
import sys

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.services.video_service import get_video_service

def upload_video(video_id: str, title: str, description: str, technique: str,
                difficulty: str, video_path: str, tags: list = None):
    """Upload một video vào database"""

    try:
        video_service = get_video_service()

        video = video_service.add_video_manual(
            video_id=video_id,
            title=title,
            description=description,
            technique=technique,
            difficulty=difficulty,
            file_path=video_path,
            tags=tags or []
        )

        print("✅ Upload video thành công!")
        print(f"   ID: {video.id}")
        print(f"   Title: {video.title}")
        print(f"   Technique: {video.technique}")
        print(f"   Difficulty: {video.difficulty}")
        print(f"   File: {video.file_path}")

        return True

    except Exception as e:
        print(f"❌ Lỗi upload video {video_id}: {e}")
        return False

if __name__ == "__main__":
    print("🎬 Video Upload Script")
    print("=" * 50)

    # Upload video mẫu để test
    videos_to_upload = [
        {
            "id": "smash_basic",
            "title": "Smash Basic",
            "description": "Hướng dẫn cú đánh smash cơ bản trong bóng bàn",
            "technique": "smash",
            "difficulty": "beginner",
            "file_path": "video/smash_basic.mp4",
            "tags": ["smash", "basic", "cơ bản"]
        }
    ]

    uploaded_count = 0
    total_count = len(videos_to_upload)

    for video_config in videos_to_upload:
        video_path = video_config["file_path"]

        if os.path.exists(video_path):
            success = upload_video(
                video_id=video_config["id"],
                title=video_config["title"],
                description=video_config["description"],
                technique=video_config["technique"],
                difficulty=video_config["difficulty"],
                video_path=video_config["file_path"],
                tags=video_config["tags"]
            )
            if success:
                uploaded_count += 1
        else:
            print(f"⚠️  File không tồn tại: {video_path}")
            print("   Bạn cần đặt file video vào đúng thư mục trước khi chạy script này.")

    print(f"\n🎉 Hoàn thành! Đã upload {uploaded_count}/{total_count} video")