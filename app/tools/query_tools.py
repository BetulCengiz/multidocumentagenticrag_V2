"""
Agentic RAG - Sorgu Araçları
Vektör arama ve özet araçları - Türkçe prompt desteği ile.
"""

import logging
import re
from typing import List, Optional, Tuple
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition, MetadataFilter
from llama_index.core.schema import BaseNode
from llama_index.core.postprocessor import SimilarityPostprocessor
from app.core.llm import get_llm
from app.config import settings
from app.core.prompts import (
    TURKISH_QA_TEMPLATE,
    TURKISH_REFINE_TEMPLATE,
    TURKISH_TREE_SUMMARIZE_TEMPLATE,
)

logger = logging.getLogger(__name__)


class QueryToolFactory:
    """Sorgu araçları üreticisi"""
    
    def __init__(self):
        self.llm = get_llm()
    
    def create_vector_tool(
        self,
        index: VectorStoreIndex,
        name: str,
        description: Optional[str] = None
    ) -> FunctionTool:
        """Vektör arama aracı oluşturur"""
        
        def vector_query(
            query: str,
            page_numbers: Optional[List[str]] = None
        ) -> str:
            """Semantik vektör araması yapar."""
            
            madde_match = re.search(r'[Mm]adde\s*(\d+)', query)
            
            filters = None
            is_party_query = any(kw in query.lower() for kw in ["taraf", "kimler", "hazırlayan", "arasında"])
            
            similarity_postprocessor = SimilarityPostprocessor(
                similarity_cutoff=0.15
            )
            
            current_top_k = 15 if is_party_query else 10
            
            query_engine = index.as_query_engine(
                similarity_top_k=current_top_k,
                llm=self.llm,
                filters=filters,
                node_postprocessors=[similarity_postprocessor],
                text_qa_template=TURKISH_QA_TEMPLATE,
                refine_template=TURKISH_REFINE_TEMPLATE,
                streaming=True,
            )
            
            response = query_engine.query(query)
            return response
        
        tool_description = description or (
            f"'{name}' dökümanında semantik arama yapar. "
            "Belirli konular, maddeler veya detaylar hakkında soru sormak için kullanın."
        )
        
        return FunctionTool.from_defaults(
            name=f"vector_search_{name}",
            fn=vector_query,
            description=tool_description,
        )
    
    def create_summary_tool(
        self,
        nodes: List[BaseNode],
        name: str,
        description: Optional[str] = None
    ) -> QueryEngineTool:
        """Özet aracı oluşturur"""
        
        summary_index = SummaryIndex(nodes)
        
        summary_engine = summary_index.as_query_engine(
            response_mode="compact",
            llm=self.llm,
            summary_template=TURKISH_TREE_SUMMARIZE_TEMPLATE,
            streaming=True,
        )
        
        tool_description = description or (
            f"'{name}' dökümanının genel özetini almak için kullanın. "
            "SADECE dökümanın tamamı hakkında genel bilgi istendiğinde kullanın."
        )
        
        return QueryEngineTool.from_defaults(
            name=f"summary_{name}",
            query_engine=summary_engine,
            description=tool_description,
        )
    
    def create_tools_for_document(
        self,
        index: VectorStoreIndex,
        nodes: List[BaseNode],
        doc_name: str
    ) -> Tuple[FunctionTool, QueryEngineTool]:
        """Bir döküman için tüm araçları oluşturur"""
        
        vector_tool = self.create_vector_tool(index, doc_name)
        summary_tool = self.create_summary_tool(nodes, doc_name)
        
        logger.info(f"'{doc_name}' için araçlar oluşturuldu")
        
        return vector_tool, summary_tool


def get_query_tool_factory() -> QueryToolFactory:
    """QueryToolFactory instance döndürür"""
    return QueryToolFactory()