"""
Agentic RAG - Döküman İşleme Modülü
"""

import logging
import re
from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core import Settings as LlamaSettings
from app.config import settings
from app.core.embeddings import get_embedding_model
from app.core.llm import get_llm
from app.vectorstore.chroma_store import get_chroma_manager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Döküman İşleme ve İndeksleme"""
    
    def __init__(self):
        LlamaSettings.embed_model = get_embedding_model()
        LlamaSettings.llm = get_llm()
        LlamaSettings.chunk_size = settings.chunk_size
        LlamaSettings.chunk_overlap = settings.chunk_overlap
        
        self.chroma_manager = get_chroma_manager()
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            paragraph_separator="\n\n",
        )
        
        logger.info(f"DocumentProcessor başlatıldı (chunk_size={settings.chunk_size})")
    
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """Dökümanları yükler"""
        if file_paths:
            reader = SimpleDirectoryReader(input_files=file_paths)
        else:
            data_path = Path(settings.data_dir)
            reader = SimpleDirectoryReader(
                input_dir=str(data_path),
                recursive=True,
                required_exts=[".pdf", ".txt", ".docx", ".md"],
            )
        
        documents = reader.load_data()
        for doc in documents:
            doc.text = self.clean_text(doc.text)
            
        logger.info(f"{len(documents)} döküman yüklendi ve temizlendi")
        return documents
    
    def clean_text(self, text: str) -> str:
        """Metindeki fazla boşlukları ve PDF karakter hatalarını temizler"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def process_documents(
        self, 
        documents: List[Document],
        doc_name: str = None
    ) -> List[BaseNode]:
        nodes = self.splitter.get_nodes_from_documents(documents)
        for i, node in enumerate(nodes):
            if doc_name:
                node.metadata["doc_name"] = doc_name
            node.metadata["chunk_index"] = i
            madde_match = re.search(r'(?:M\s*A\s*D\s*D\s*E|Madde|MADDE)\s*[:.-]?\s*(\d+)', node.text)
            if madde_match:
                node.metadata["madde_no"] = madde_match.group(1)
            if i < 5 and ("taraf" in node.text.lower() or "arasında" in node.text.lower()):
                node.metadata["is_header"] = True
        
        logger.info(f"{len(nodes)} node oluşturuldu")
        return nodes
    
    def create_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """Vektör indeksi oluşturur"""
        logger.info("Vektör indeksi oluşturuluyor...")
        
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.chroma_manager.storage_context,
            show_progress=True,
        )
        
        logger.info("Vektör indeksi oluşturuldu")
        return index


def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()