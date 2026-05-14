"""
FastAPI Backend Server - Table Tennis Assessment Toolkit
API endpoints for mobile/desktop clients
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import routers
from routers import pose_router, ball_router, compare_router, chatbot_router, video_stream_router

# Initialize FastAPI app with lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("Backend server starting...")
    yield
    print("Backend server shutting down...")

app = FastAPI(
    title="TTATool Backend API",
    description="Table Tennis Assessment Toolkit - Mobile/Desktop Backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for mobile/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, đặt domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pose_router.router, prefix="/api/pose", tags=["Pose Detection"])
app.include_router(ball_router.router, prefix="/api/ball", tags=["Ball Tracking"])
app.include_router(compare_router.router, prefix="/api/compare", tags=["Pose Comparison"])
app.include_router(chatbot_router.router, prefix="/api/chatbot", tags=["Chatbot"])
app.include_router(video_stream_router.router, prefix="/api/video", tags=["Video Stream"])

# Root endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "TTATool Backend API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"  # Swagger UI
    }

@app.get("/health")
async def health_check():
    """Health check for mobile clients"""
    return {
        "status": "healthy",
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Chạy server
    # Windows: python backend/main.py
    # Linux/Mac: python -m uvicorn backend.main:app --reload
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Accessible từ mọi IP
        port=8000,
        reload=True  # Auto-reload khi code thay đổi
    )
