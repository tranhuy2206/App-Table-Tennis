"""
Chatbot Router - RAG chatbot về giáo trình bóng bàn
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import sys
import os

router = APIRouter()

class ChatMessage(BaseModel):
    """Model cho một tin nhắn"""
    role: str  # "user" hoặc "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Request model cho chat"""
    message: str
    session_id: Optional[str] = None  # Thread ID cho LangGraph


class ChatResponse(BaseModel):
    """Response model cho chat"""
    success: bool
    message: str
    response: Optional[str] = None
    sources: Optional[List[str]] = None  # PDF sources được trích dẫn
    video_ids: Optional[List[str]] = None  # Video IDs để Android app parse


# Global chatbot instance (khởi tạo lần đầu)
_chatbot_instance = None
_chatbot_ready = False
_chatbot_error = None


def get_chatbot():
    """Lấy chatbot instance"""
    global _chatbot_instance, _chatbot_ready, _chatbot_error
    
    if _chatbot_instance is None:
        try:
            CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code"))
            if CODE_DIR not in sys.path:
                sys.path.insert(0, CODE_DIR)
            from chatbot import build_chatbot
            
            print("⏳ Khởi tạo chatbot...")
            # Data directory (đường dẫn tuyệt đối) để tránh WinError 3 khi chạy từ backend/
            DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
            if not os.path.exists(DATA_DIR):
                raise FileNotFoundError(f"Data directory không tồn tại: {DATA_DIR}")
            _chatbot_instance = build_chatbot(DATA_DIR)
            _chatbot_ready = True
            _chatbot_error = None
            print("✅ Chatbot sẵn sàng")
        except Exception as e:
            _chatbot_instance = None
            _chatbot_ready = False
            _chatbot_error = str(e)
            print(f"❌ Lỗi khởi tạo chatbot: {_chatbot_error}")
    
    return _chatbot_instance


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Gửi tin nhắn đến chatbot
    
    Args:
        message: Nội dung câu hỏi
        session_id: ID session cho context dài hạn 
    
    Returns:
        JSON với response từ chatbot
    """
    
    try:
        chatbot = get_chatbot()
        
        if not _chatbot_ready or chatbot is None:
            msg = "Chatbot chưa sẵn sàng."
            if _chatbot_error:
                msg += f" Lỗi khởi tạo: {_chatbot_error}"
            return ChatResponse(
                success=False,
                message=msg,
                response=None
            )
        
        # Tạo session nếu chưa có
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        config = {"configurable": {"thread_id": session_id}}
        
        # Gọi chatbot
        from langchain_core.messages import HumanMessage
        
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        if not response or not isinstance(response, dict) or "messages" not in response:
            raise ValueError("Không nhận được messages từ chatbot")

        final_msg = response.get('messages')[-1]
        if final_msg is None or not hasattr(final_msg, 'content'):
            raise ValueError("Không nhận được nội dung từ chatbot")

        if isinstance(final_msg.content, list):
            clean_text = "".join([
                item.get('text') if isinstance(item, dict) and 'text' in item else str(item)
                for item in final_msg.content
            ])
            response_text = clean_text
        else:
            response_text = str(final_msg.content)

        if not response_text:
            raise ValueError("Chatbot trả về nội dung rỗng")

        # Extract VIDEO_IDs từ response text
        import re
        video_ids = re.findall(r'\[VIDEO_ID:([^\]]+)\]', response_text)

        return ChatResponse(
            success=True,
            message="Lấy response thành công",
            response=response_text,
            video_ids=video_ids if video_ids else None
        )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            message=f"Lỗi: {str(e)}",
            response=None
        )


@router.get("/init")
async def init_chatbot():
    """Khởi tạo chatbot sẵn sàng"""
    try:
        chatbot = get_chatbot()
        
        if _chatbot_ready:
            return {
                "status": "ready",
                "message": "Chatbot sẵn sàng"
            }
        else:
            return {
                "status": "initializing",
                "message": "Chatbot đang khởi tạo..."
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Lỗi khởi tạo: {str(e)}"
        }


@router.get("/status")
async def chatbot_status():
    """Kiểm tra trạng thái chatbot"""
    return {
        "ready": _chatbot_ready,
        "instance_id": id(_chatbot_instance),
        "message": "Chatbot sẵn sàng" if _chatbot_ready else "Chatbot chưa khởi tạo"
    }


@router.get("/info")
async def chatbot_info():
    """Thông tin về chatbot"""
    return {
        "name": "TTATool Chatbot",
        "description": "RAG chatbot trợ lý học tập môn Bóng bàn",
        "capabilities": [
            "Trả lời câu hỏi về nội dung giáo trình",
            "Cung cấp lịch học chi tiết",
            "Giải thích kỹ thuật bóng bàn",
            "Tư vấn điều kiện thi"
        ],
        "llm": "Google Gemini 2.5 Flash",
        "vector_db": "ChromaDB",
        "language": "Vietnamese"
    }
