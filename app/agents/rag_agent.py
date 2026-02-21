"""
Agentic RAG - Agent Modülü
"""

import logging
from typing import List, Dict, Any, Optional
from llama_index.core.tools import BaseTool
from app.core.llm import get_llm

logger = logging.getLogger(__name__)


class RAGAgent:
    """Agentic RAG Sorgulama Ajanı"""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.llm = get_llm()
        
        self.vector_tools = {}
        self.summary_tools = {}
        
        for tool in tools:
            if tool is None:
                continue
            tool_name = tool.metadata.name
            if "vector" in tool_name or "search" in tool_name:
                doc_name = tool_name.replace("vector_search_", "")
                self.vector_tools[doc_name] = tool
            elif "summary" in tool_name:
                doc_name = tool_name.replace("summary_", "")
                self.summary_tools[doc_name] = tool
        
        logger.info(f"RAGAgent başlatıldı - {len(tools)} araç")
    
    def _select_tool(self, query: str, doc_name: Optional[Any] = None) -> Optional[BaseTool]:
        """Sorguya uygun aracı seçer"""
        if isinstance(doc_name, list):
            doc_name = doc_name[0] if doc_name else None
            
        summary_keywords = ["özetle", "özet", "genel", "kısaca", "tamamı"]
        query_lower = query.lower()
        is_summary = any(kw in query_lower for kw in summary_keywords)
        
        if doc_name and doc_name != "Tüm Dökümanlar":
            if is_summary and doc_name in self.summary_tools:
                return self.summary_tools[doc_name]
            elif doc_name in self.vector_tools:
                return self.vector_tools[doc_name]
        
        if is_summary and self.summary_tools:
            return list(self.summary_tools.values())[0]
        elif self.vector_tools:
            return list(self.vector_tools.values())[0]
        
        if self.tools:
            for tool in self.tools:
                if tool is not None:
                    return tool
        return None
    
    def query(self, query: str, doc_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Sorguyu işler.
        
        Args:
            query: Kullanıcı sorusu
            doc_name: Opsiyonel döküman adı
        """
        logger.info(f"Sorgu: {query[:80]}...")
        
        if not self.tools:
            return {
                "response": "Henüz döküman yüklenmemiş.",
                "tools_used": [],
                "success": False,
            }
        
        selected_tool = self._select_tool(query, doc_name)
        
        if selected_tool is None:
            return {
                "response": "Uygun araç bulunamadı.",
                "tools_used": [],
                "success": False,
            }
        
        tool_name = selected_tool.metadata.name
        logger.info(f"Seçilen araç: {tool_name}")
        
        try:
            if hasattr(selected_tool, 'fn'):
                response = selected_tool.fn(query)
            elif hasattr(selected_tool, 'query_engine'):
                response = selected_tool.query_engine.query(query)
            else:
                response = str(selected_tool(query))
            
            return {
                "response": str(response),
                "tools_used": [tool_name],
                "success": True,
            }
        except Exception as e:
            logger.error(f"Araç hatası: {e}")
            return {
                "response": f"Hata: {str(e)}",
                "tools_used": [tool_name],
                "success": False,
            }

    def query_stream(self, query: str, doc_name: Optional[Any] = None):
        """Sorguyu akışlı (streaming) olarak işler."""
        selected_tool = self._select_tool(query, doc_name)
        if selected_tool is None:
            yield "Uygun araç bulunamadı."
            return

        try:
            if hasattr(selected_tool, 'query_engine'):
                response = selected_tool.query_engine.query(query)
                if hasattr(response, "response_gen"):
                    buffer = ""
                    for token in response.response_gen:
                        buffer += token
                        if len(buffer) >= 20 or "\n" in buffer:
                            yield buffer
                            buffer = ""
                    if buffer:
                        yield buffer
                else:
                    yield str(response)
            elif hasattr(selected_tool, 'fn'):
                response = selected_tool.fn(query)
                if hasattr(response, "response_gen"):
                    buffer = ""
                    for token in response.response_gen:
                        buffer += token
                        if len(buffer) >= 20 or "\n" in buffer:
                            yield buffer
                            buffer = ""
                    if buffer:
                        yield buffer
                else:
                    yield str(response)
            else:
                yield str(selected_tool(query))
        except Exception as e:
            yield f"Hata: {str(e)}"
    
    def list_tools(self) -> List[Dict[str, str]]:
        result = []
        for tool in self.tools:
            if tool is not None and hasattr(tool, 'metadata'):
                result.append({
                    "name": tool.metadata.name,
                    "description": tool.metadata.description[:80] + "...",
                })
        return result


class MultiDocumentRAGAgent(RAGAgent):
    """Çoklu döküman destekli RAG Ajanı"""
    
    def __init__(self, document_tools: Dict[str, List[BaseTool]]):
        self.document_tools = document_tools
        
        all_tools = []
        for doc_name, tools in document_tools.items():
            for tool in tools:
                if tool is not None:
                    all_tools.append(tool)
        
        super().__init__(all_tools)
    
    def query_document(self, query: str, document_name: str) -> Dict[str, Any]:
        if document_name not in self.document_tools:
            return {
                "response": f"'{document_name}' bulunamadı.",
                "tools_used": [],
                "success": False,
            }
        return self.query(query, document_name)
    
    def get_document_list(self) -> List[str]:
        return list(self.document_tools.keys())