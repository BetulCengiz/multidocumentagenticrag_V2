"""
Agentic RAG - LLM Modülü
Ollama üzerinden llama3.1:8b modeli ile inference.
"""

import logging
from llama_index.llms.ollama import Ollama
from app.config import settings
from app.core.prompts import TURKISH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMManager:
    """Ollama LLM Yöneticisi"""
    
    _instance = None
    _llm = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._llm is None:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Ollama LLM'i başlatır"""
        logger.info(f"Ollama LLM başlatılıyor: {settings.ollama_model}")
        logger.info(f"Ollama Host: {settings.ollama_host}")
        
        self._llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_host,
            request_timeout=600.0,
            temperature=0.1,
            context_window=8192,
            system_prompt=TURKISH_SYSTEM_PROMPT,
            additional_kwargs={
                "num_predict": 2048,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "mirostat": 0,
                "tfs_z": 1.0,
            }
        )
        
        logger.info("Ollama LLM başarıyla başlatıldı!")
    
    @property
    def model(self) -> Ollama:
        """LLM modelini döndürür"""
        return self._llm
    
    def generate(self, prompt: str) -> str:
        """Basit metin üretimi"""
        response = self._llm.complete(prompt)
        return response.text


def get_llm() -> Ollama:
    """LLM'i döndüren yardımcı fonksiyon"""
    manager = LLMManager()
    return manager.model