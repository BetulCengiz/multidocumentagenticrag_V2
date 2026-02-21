"""
Agentic RAG - ChromaDB Vektör Deposu
Kalıcı vektör depolama ve sorgulama.
"""

import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from app.config import settings

logger = logging.getLogger(__name__)


class ChromaManager:
    """ChromaDB Vektör Deposu Yöneticisi"""
    
    _instance = None
    _client = None
    _collection = None
    _vector_store = None
    _storage_context = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._initialize_chroma()
    
    def _initialize_chroma(self):
        """ChromaDB'yi başlatır"""
        persist_path = Path(settings.chroma_persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChromaDB başlatılıyor: {persist_path}")
        
        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Koleksiyon hazır: {settings.chroma_collection_name}")
        logger.info(f"Mevcut chunk sayısı: {self._collection.count()}")
        
        self._vector_store = ChromaVectorStore(
            chroma_collection=self._collection
        )
        
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
    
    @property
    def client(self) -> chromadb.PersistentClient:
        return self._client
    
    @property
    def collection(self):
        return self._collection
    
    @property
    def vector_store(self) -> ChromaVectorStore:
        return self._vector_store
    
    @property
    def storage_context(self) -> StorageContext:
        return self._storage_context
    
    def document_exists(self, doc_name: str) -> bool:
        """Dökümanın zaten indekslenip indekslenmediğini kontrol eder"""
        try:
            results = self._collection.get(
                where={"doc_name": doc_name},
                limit=1
            )
            exists = len(results['ids']) > 0
            if exists:
                logger.info(f"'{doc_name}' zaten indekslenmiş, atlanıyor.")
            return exists
        except Exception as e:
            logger.warning(f"Döküman kontrolü başarısız: {e}")
            return False
    
    def get_indexed_documents(self) -> list:
        """İndekslenmiş döküman listesini döndürür"""
        try:
            results = self._collection.get()
            doc_names = set()
            for metadata in results.get('metadatas', []):
                if metadata and 'doc_name' in metadata:
                    doc_names.add(metadata['doc_name'])
            return list(doc_names)
        except Exception as e:
            logger.warning(f"Döküman listesi alınamadı: {e}")
            return []
    
    def delete_document(self, doc_name: str):
        """Belirli bir dökümanı siler"""
        try:
            self._collection.delete(
                where={"doc_name": doc_name}
            )
            logger.info(f"'{doc_name}' silindi.")
        except Exception as e:
            logger.error(f"Döküman silinemedi: {e}")
    
    def reset_collection(self):
        """Koleksiyonu sıfırlar"""
        logger.warning(f"Koleksiyon sıfırlanıyor: {settings.chroma_collection_name}")
        self._client.delete_collection(settings.chroma_collection_name)
        self._initialize_chroma()
    
    def get_stats(self) -> dict:
        """Koleksiyon istatistiklerini döndürür"""
        indexed_docs = self.get_indexed_documents()
        return {
            "collection_name": settings.chroma_collection_name,
            "document_count": self._collection.count(),
            "indexed_documents": indexed_docs,
            "persist_directory": settings.chroma_persist_dir,
        }
    
    def get_nodes_by_document(self, doc_name: str) -> list:
        """Belirli bir dökümana ait node'ları ChromaDB'den çeker"""
        try:
            results = self._collection.get(
                where={"doc_name": doc_name}
            )
            
            from llama_index.core.schema import TextNode
            import json
            
            nodes = []
            for i in range(len(results['ids'])):
                metadata = results['metadatas'][i]
                if metadata and '_node_content' in metadata:
                    node_dict = json.loads(metadata['_node_content'])
                    nodes.append(TextNode.from_dict(node_dict))
                else:
                    content = results['documents'][i]
                    nodes.append(TextNode(text=content, metadata=metadata))
            
            logger.info(f"'{doc_name}' için {len(nodes)} adet node veritabanından çekildi.")
            return nodes
        except Exception as e:
            logger.error(f"Node'lar alınamadı ({doc_name}): {e}")
            return []


def get_chroma_manager() -> ChromaManager:
    """ChromaManager instance döndürür"""
    return ChromaManager()


def get_storage_context() -> StorageContext:
    """StorageContext döndüren yardımcı fonksiyon"""
    return ChromaManager().storage_context