"""
Agentic RAG - Embedding Modülü
HuggingFace multilingual-e5-large modeli ile GPU üzerinde embedding üretimi.
"""

import logging
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """HuggingFace Embedding Yöneticisi (GPU Destekli)"""
    
    _instance = None
    _embed_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._embed_model is None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Embedding modelini GPU üzerinde başlatır"""
        logger.info(f"Embedding modeli yükleniyor: {settings.embedding_model}")
        logger.info(f"Cihaz: {settings.embedding_device}")
        if settings.embedding_device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA bulunamadı, CPU kullanılacak!")
                device = "cpu"
            else:
                device = "cuda"
                logger.info(f"GPU bulundu: {torch.cuda.get_device_name(0)}")
        else:
            device = settings.embedding_device
        
        self._embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            device=device,
            embed_batch_size=32,
            trust_remote_code=True,
            query_instruction="query: ",
            text_instruction="passage: ",
        )
        
        logger.info("Embedding modeli başarıyla yüklendi!")
    
    @property
    def model(self) -> HuggingFaceEmbedding:
        """Embedding modelini döndürür"""
        return self._embed_model
    
    def get_embedding(self, text: str) -> list:
        """Tek bir metin için embedding üretir"""
        return self._embed_model.get_text_embedding(text)
    
    def get_embeddings(self, texts: list) -> list:
        """Birden fazla metin için embedding üretir"""
        return self._embed_model.get_text_embedding_batch(texts)


def get_embedding_model() -> HuggingFaceEmbedding:
    """Embedding modelini döndüren yardımcı fonksiyon"""
    manager = EmbeddingManager()
    return manager.model