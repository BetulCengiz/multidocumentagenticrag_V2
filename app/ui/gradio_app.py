"""
Agentic RAG - Gradio Web Arayüzü
Türkçe sözleşme ve hukuki metinler için kullanıcı arayüzü.
"""

import gradio as gr
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import time
import shutil

from app.config import settings


CUSTOM_CSS = """
.container {
    max-width: 1200px !important;
    margin: auto !important;
}
.title {
    text-align: center !important;
    color: #1a365d !important;
    margin-bottom: 0.5rem !important;
}
.subtitle {
    text-align: center !important;
    color: #4a5568 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1.5rem !important;
}
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
.footer {
    text-align: center !important;
    color: #718096 !important;
    font-size: 0.875rem !important;
    margin-top: 1rem !important;
    padding-top: 1rem !important;
    border-top: 1px solid #e2e8f0 !important;
}
"""


class GradioInterface:
    """Gradio Web Arayüzü"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.chat_history: List[Tuple[str, str]] = []
    
    def process_query(
        self,
        message: str,
        history: List[Tuple[str, str]],
        selected_doc: Optional[str] = None
    ):
        """Kullanıcı sorgusunu işler (Streaming)"""
        
        if not message.strip():
            yield "", history
            return
        
        try:
            if self.rag_system.agent is None:
                yield "", history + [(message, "⚠️ Sistem henüz hazır değil.")]
                return
            
            history.append((message, ""))
            if selected_doc and selected_doc != "Tüm Dökümanlar":
                response_gen = self.rag_system.agent.query_stream(message, selected_doc)
            else:
                response_gen = self.rag_system.agent.query_stream(message, None)
            
            for chunk in response_gen:
                partial_response += chunk
                history[-1] = (message, f"{partial_response} ▌")
                yield "", history
            
            elapsed_time = time.time() - start_time
            history[-1] = (message, f"{partial_response}\n\n---\n⏱️ Süre: {elapsed_time:.2f}s")
            yield "", history
        
        except Exception as e:
            logger.error(f"Sorgu hatası: {e}")
            history.append((message, f"❌ Hata: {str(e)}"))
            yield "", history
    
    def get_document_list(self) -> List[str]:
        """Döküman listesini döndürür"""
        docs = ["Tüm Dökümanlar"]
        try:
            if self.rag_system.agent:
                docs.extend(self.rag_system.agent.get_document_list())
            elif self.rag_system.document_tools:
                docs.extend(list(self.rag_system.document_tools.keys()))
        except Exception as e:
            logger.error(f"Döküman listesi alınamadı: {e}")
        return docs
    def get_system_stats(self) -> str:
        """Sistem istatistiklerini döndürür"""
        try:
            stats = self.rag_system.chroma_manager.get_stats()
            
            stats_text = f"""
### 📊 Sistem İstatistikleri

| Metrik | Değer |
|--------|-------|
| 📁 Koleksiyon | {stats['collection_name']} |
| 📄 Chunk Sayısı | {stats['document_count']} |

### ⚙️ Model Bilgileri

| Bileşen | Model |
|---------|-------|
| 🤖 LLM | {settings.ollama_model} |
| 📐 Embedding | {settings.embedding_model} |
| 📏 Chunk Size | {settings.chunk_size} |
| 🔄 Overlap | {settings.chunk_overlap} |
"""
            return stats_text
        except Exception as e:
            return f"❌ İstatistikler alınamadı: {str(e)}"
    
    def get_tool_list(self) -> str:
        """Araç listesini döndürür"""
        try:
            if not self.rag_system.agent:
                return "⚠️ Agent henüz başlatılmadı."
            
            tools = self.rag_system.agent.list_tools()
            
            if not tools:
                return "⚠️ Henüz araç oluşturulmamış."
            
            tool_text = "### 🔧 Mevcut Araçlar\n\n"
            for tool in tools:
                tool_text += f"**{tool['name']}**\n"
                tool_text += f"> {tool['description']}\n\n"
            
            return tool_text
        except Exception as e:
            return f"❌ Araç listesi alınamadı: {str(e)}"
    
    def upload_document(self, files) -> Tuple[str, List[str]]:
        """Yeni döküman yükler"""
        if not files:
            return "⚠️ Dosya seçilmedi.", self.get_document_list()
        
        try:
            uploaded_files = []
            data_dir = Path(settings.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            for file in files:
                file_path = Path(file.name)
                dest_path = data_dir / file_path.name
                
                shutil.copy(file.name, dest_path)
                uploaded_files.append(file_path.name)
                self.rag_system._process_single_document(dest_path)
            self.rag_system.initialize_agent()
            
            return (
                f"✅ Yüklenen dosyalar: {', '.join(uploaded_files)}\n\nDökümanlar başarıyla indekslendi!",
                gr.update(choices=self.get_document_list(), value="Tüm Dökümanlar")
            )
        
        except Exception as e:
            logger.error(f"Yükleme hatası: {e}")
            return f"❌ Yükleme hatası: {str(e)}", self.get_document_list()
    
    def clear_history(self) -> List:
        """Sohbet geçmişini temizler"""
        self.chat_history = []
        return []
    
    def _format_doc_list(self) -> str:
        """Döküman listesini formatlar"""
        try:
            if self.rag_system.agent:
                docs = self.rag_system.agent.get_document_list()
            elif self.rag_system.document_tools:
                docs = list(self.rag_system.document_tools.keys())
            else:
                docs = []
            
            if not docs:
                return "⚠️ Henüz döküman yüklenmemiş."
            
            doc_text = "| # | Döküman Adı |\n|---|-------------|\n"
            for i, doc in enumerate(docs, 1):
                doc_text += f"| {i} | 📄 {doc} |\n"
            
            doc_text += f"\n**Toplam: {len(docs)} döküman**"
            return doc_text
        except Exception as e:
            return f"⚠️ Liste alınamadı: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Gradio arayüzünü oluşturur"""
        
        with gr.Blocks(
            title="Agentic RAG - Türkçe Sözleşme Analizi",
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="purple",
            )
        ) as demo:
            
            gr.HTML("""
                <div class="title">
                    <h1>🔍 Agentic RAG</h1>
                </div>
                <div class="subtitle">
                    Türkçe Sözleşme ve Hukuki Metin Analiz Sistemi
                </div>
            """)
            with gr.Tabs() as main_tabs:
                with gr.TabItem("💬 Soru-Cevap", id="chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Sohbet",
                                height=500,
                                show_label=False,
                            )
                            
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    placeholder="Sözleşme hakkında bir soru sorun...",
                                    lines=2,
                                    scale=4,
                                    show_label=False,
                                )
                                submit_btn = gr.Button(
                                    "Gönder 📤",
                                    variant="primary",
                                    scale=1,
                                )
                            
                            clear_btn = gr.Button("🗑️ Sohbeti Temizle", size="sm")
                        
                        with gr.Column(scale=1):
                            doc_dropdown = gr.Dropdown(
                                label="📁 Döküman Seçin",
                                choices=self.get_document_list(),
                                value="Tüm Dökümanlar",
                                interactive=True,
                            )
                            
                            refresh_docs_btn = gr.Button("🔄 Listeyi Yenile", size="sm")
                            
                            
                            gr.Markdown("### 💡 Örnek Sorular")
                            
                            examples = [
                                "Sözleşmenin tarafları kimlerdir?",
                                "Sözleşme bedeli ne kadar?",
                                "Madde 5'te neler belirtilmiş?",
                                "Fesih koşulları nelerdir?",
                                "Sözleşmeyi özetle",
                            ]
                            
                            for q in examples:
                                gr.Button(q, size="sm").click(
                                    fn=lambda x=q: x,
                                    outputs=msg_input
                                )
                
                
                with gr.TabItem("📄 Döküman Yönetimi", id="docs"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📤 Yeni Döküman Yükle")
                            file_upload = gr.File(
                                label="PDF veya TXT dosyası seçin",
                                file_types=[".pdf", ".txt", ".docx"],
                                file_count="multiple",
                            )
                            upload_btn = gr.Button(
                                "Yükle ve İndeksle 📥",
                                variant="primary",
                            )
                            upload_status = gr.Markdown("")
                        
                        with gr.Column():
                            gr.Markdown("### 📚 Mevcut Dökümanlar")
                            doc_list_display = gr.Markdown(
                                value=self._format_doc_list()
                            )
                            refresh_list_btn = gr.Button("🔄 Yenile", size="sm")
                
                with gr.TabItem("⚙️ Sistem", id="system"):
                    with gr.Row():
                        with gr.Column():
                            stats_display = gr.Markdown(value=self.get_system_stats())
                            refresh_stats_btn = gr.Button("🔄 Yenile", size="sm")
                        
                        with gr.Column():
                            tools_display = gr.Markdown(value=self.get_tool_list())
                            refresh_tools_btn = gr.Button("🔄 Yenile", size="sm")
                
            gr.HTML("""
                <div class="footer">
                    <p>🚀 Agentic RAG v1.0 | GPU Accelerated</p>
                </div>
            """)
            
            
            submit_btn.click(
                fn=self.process_query,
                inputs=[msg_input, chatbot, doc_dropdown],
                outputs=[msg_input, chatbot],
            )
            
            msg_input.submit(
                fn=self.process_query,
                inputs=[msg_input, chatbot, doc_dropdown],
                outputs=[msg_input, chatbot],
            )
            
            clear_btn.click(
                fn=self.clear_history,
                outputs=chatbot,
            )
            
            refresh_docs_btn.click(
                fn=self.get_document_list,
                outputs=doc_dropdown,
            )
            
            upload_btn.click(
                fn=self.upload_document,
                inputs=file_upload,
                outputs=[upload_status, doc_dropdown],
            ).then(
                fn=self._format_doc_list,
                outputs=doc_list_display,
            )


            
            refresh_list_btn.click(
                fn=self._format_doc_list,
                outputs=doc_list_display,
            )
            
            refresh_stats_btn.click(
                fn=self.get_system_stats,
                outputs=stats_display,
            )
            
            refresh_tools_btn.click(
                fn=self.get_tool_list,
                outputs=tools_display,
            )
        
        return demo
    
    def launch(self, **kwargs):
        """Arayüzü başlatır"""
        demo = self.create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            **kwargs
        )


def create_gradio_interface(rag_system) -> GradioInterface:
    """GradioInterface instance oluşturur"""
    return GradioInterface(rag_system)