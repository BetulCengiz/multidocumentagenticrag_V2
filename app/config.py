"""
Agentic RAG - Konfigürasyon Modülü
Tüm proje ayarları merkezi olarak burada yönetilir.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Proje Ayarları"""
    
    # Ollama Ayarları
    ollama_host: str = Field(default="http://ollama:11434", env="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Embedding Ayarları (HuggingFace - GPU)
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        env="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cuda", env="EMBEDDING_DEVICE")
    
    # ChromaDB Ayarları
    chroma_persist_dir: str = Field(default="/app/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="turkish_contracts",
        env="CHROMA_COLLECTION_NAME"
    )
    
    # Chunking Ayarları
    chunk_size: int = Field(default=1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, env="CHUNK_OVERLAP")
    
    # Veri Dizini
    data_dir: str = Field(default="/app/data", env="DATA_DIR")
    
    # RAG Ayarları
    similarity_top_k: int = Field(default=5, env="SIMILARITY_TOP_K")
    
    # Gradio Ayarları
    gradio_server_name: str = Field(default="0.0.0.0", env="GRADIO_SERVER_NAME")
    gradio_server_port: int = Field(default=7860, env="GRADIO_SERVER_PORT")
    
    # Çalışma Modu
    run_mode: str = Field(default="ui", env="RUN_MODE")
    
    # Log Ayarları
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # HuggingFace Cache
    hf_home: str = Field(default="/app/.cache/huggingface", env="HF_HOME")
    
    # CUDA
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
settings = Settings()


def get_data_path() -> Path:
    """Veri dizini yolunu döndürür"""
    return Path(settings.data_dir)


def get_chroma_path() -> Path:
    """ChromaDB dizini yolunu döndürür"""
    return Path(settings.chroma_persist_dir)