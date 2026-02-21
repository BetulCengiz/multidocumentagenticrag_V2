"""
Agentic RAG - Ana Uygulama
Türkçe sözleşme ve hukuki metinler için Agentic RAG sistemi.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from app.config import settings
from app.core.embeddings import get_embedding_model
from app.core.llm import get_llm
from app.ingestion.document_processor import get_document_processor
from app.tools.query_tools import get_query_tool_factory
from app.agents.rag_agent import MultiDocumentRAGAgent
from app.vectorstore.chroma_store import get_chroma_manager
from app.ui.gradio_app import create_gradio_interface


class AgenticRAGSystem:
    """Ana Agentic RAG Sistemi"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("🚀 AGENTIC RAG SİSTEMİ")
        logger.info("=" * 60)
        
        self.document_processor = get_document_processor()
        self.tool_factory = get_query_tool_factory()
        self.chroma_manager = get_chroma_manager()
        
        self.document_tools: Dict[str, List] = {}
        self.document_nodes: Dict[str, List] = {}
        self.agent: Optional[MultiDocumentRAGAgent] = None
        
        logger.info("Sistem bileşenleri hazır")
    
    def ingest_all_documents(self):
        """Sadece YENİ dökümanları işler"""
        data_path = Path(settings.data_dir)
        
        if not data_path.exists():
            logger.warning(f"Veri dizini bulunamadı, oluşturuluyor: {data_path}")
            data_path.mkdir(parents=True, exist_ok=True)
            return
        
        all_files = []
        for ext in [".pdf", ".txt", ".docx", ".md"]:
            all_files.extend(list(data_path.glob(f"*{ext}")))
        
        if not all_files:
            logger.warning(f"İşlenecek döküman bulunamadı! Veri yolu: {data_path.absolute()}")
            return
        
        logger.info(f"Bulunan dosyalar: {[f.name for f in all_files]}")
        
        indexed_docs = self.chroma_manager.get_indexed_documents()
        logger.info(f"Mevcut indeksli dökümanlar: {indexed_docs}")
        
        for file_path in all_files:
            doc_name = file_path.stem
            
            if doc_name in indexed_docs:
                logger.info(f"'{doc_name}' zaten indeksli, atlanıyor.")
                self._create_tools_for_existing_doc(doc_name)
            else:
                self._process_single_document(file_path)
        
        logger.info(f"İşleme tamamlandı. Toplam döküman: {list(self.document_tools.keys())}")
    
    def _create_tools_for_existing_doc(self, doc_name: str):
        """Mevcut indekslenmiş döküman için araçlar oluşturur"""
        try:
            from llama_index.core import VectorStoreIndex
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.chroma_manager.vector_store,
            )
            nodes = self.chroma_manager.get_nodes_by_document(doc_name)
            if nodes:
                vector_tool, summary_tool = self.tool_factory.create_tools_for_document(
                    index=index,
                    nodes=nodes,
                    doc_name=doc_name,
                )
                self.document_tools[doc_name] = [vector_tool, summary_tool]
            else:
                vector_tool = self.tool_factory.create_vector_tool(index, doc_name)
                self.document_tools[doc_name] = [vector_tool]
                
            logger.info(f"'{doc_name}' için mevcut araçlar oluşturuldu")
            
        except Exception as e:
            logger.error(f"Mevcut döküman araçları oluşturulamadı: {e}")
    
    def _process_single_document(self, file_path: Path):
        """Tek bir dökümanı işler ve indeksler"""
        doc_name = file_path.stem
        
        logger.info(f"İşleniyor: {doc_name}")
        
        try:
            self.chroma_manager.delete_document(doc_name)
            documents = self.document_processor.load_documents([str(file_path)])
            nodes = self.document_processor.process_documents(documents, doc_name)
            for node in nodes:
                node.metadata["doc_name"] = doc_name
            index = self.document_processor.create_index(nodes)
            self.document_nodes[doc_name] = nodes
            vector_tool, summary_tool = self.tool_factory.create_tools_for_document(
                index=index,
                nodes=nodes,
                doc_name=doc_name,
            )
            self.document_tools[doc_name] = [vector_tool, summary_tool]
            logger.info(f"'{doc_name}' başarıyla işlendi")
            
        except Exception as e:
            logger.error(f"'{doc_name}' işlenirken hata: {e}")
            import traceback
            traceback.print_exc()
    
    def initialize_agent(self):
        """Ajanı başlatır"""
        if not self.document_tools:
            logger.warning("İşlenmiş döküman yok")
            self.agent = None
            return
        
        self.agent = MultiDocumentRAGAgent(self.document_tools)
        logger.info("Agent başarıyla oluşturuldu")
    
    def query(self, question: str) -> str:
        """Soru sorar"""
        if self.agent is None:
            return "Sistem hazır değil veya döküman yüklenmemiş."
        
        result = self.agent.query(question)
        return result["response"]
    
    def launch_ui(self):
        """Gradio arayüzünü başlatır"""
        logger.info("=" * 60)
        logger.info("GRADIO WEB ARAYÜZÜ BAŞLATILIYOR")
        logger.info("🌐 URL: http://localhost:7860")
        logger.info("=" * 60)
        
        interface = create_gradio_interface(self)
        interface.launch()


def main():
    """Ana fonksiyon"""
    logger.info("Agentic RAG Sistemi Başlatılıyor...")
    
    system = AgenticRAGSystem()
    
    stats = system.chroma_manager.get_stats()
    logger.info(f"ChromaDB: {stats['document_count']} chunk")
    logger.info(f"İndeksli dökümanlar: {stats['indexed_documents']}")
    system.ingest_all_documents()
    system.initialize_agent()
    system.launch_ui()


if __name__ == "__main__":
    main()