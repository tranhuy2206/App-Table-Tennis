"""
Configuration file for backend
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Server
    API_TITLE: str = "TTATool Backend API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Backend API for Table Tennis Assessment Toolkit"
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Paths
    UPLOAD_DIR: str = "backend/uploads"
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "./chroma_db_official"
    
    # Processing parameters
    DEFAULT_FRAME_WIDTH: int = 640
    DEFAULT_FRAME_HEIGHT: int = 480
    DEFAULT_SAMPLE_RATE: int = 1
    DEFAULT_N_POINTS: int = 100
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
