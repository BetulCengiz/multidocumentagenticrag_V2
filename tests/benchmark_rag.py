"""
Agentic RAG - Benchmark & Test Script
Sistemin doğruluğunu ve hızını ölçmek için kullanılır.
"""

import time
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import nest_asyncio

sys.path.append(os.getcwd())

from app.main import AgenticRAGSystem
from app.config import settings

nest_asyncio.apply()

logging.getLogger("llama_index").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class RAGBenchmark:
    def __init__(self):
        print("\n" + "="*50)
        print("🚀 AGENTIC RAG BENCHMARK BAŞLATILIYOR")
        print("="*50)
        
        self.system = AgenticRAGSystem()
        self.system.ingest_all_documents()
        self.system.initialize_agent()
        self.test_questions = [
            # Spesifik Bilgi (Vector Search)
            "Sözleşmenin tarafları kimlerdir?",
            "Sözleşme bedeli ne kadar?",
            "Ödeme planı nasıl?",
            "Gecikme cezası (günlük) ne kadar?",
            "Garanti süresi kaç ay?",
            "Fesih koşulları nelerdir?",
            "Gizlilik cezası ne kadar?",
            "Banka bilgileri nelerdir?",
            
            # Özetleme (Summary Tool)
            "Sözleşmeyi 3 ana madde halinde özetle",
            "Bu dökümanın ana amacı nedir?"
        ]

    def run_tests(self, questions: List[str] = None):
        if questions:
            self.test_questions = questions
            
        results = []
        
        print(f"\n📊 Toplam {len(self.test_questions)} soru test edilecek...\n")
        
        for i, q in enumerate(self.test_questions, 1):
            print(f"[{i}/{len(self.test_questions)}] Sorgulanıyor: {q}")
            
            start_time = time.time()
            try:
                response = self.system.query(q)
                end_time = time.time()
                
                duration = end_time - start_time
                
                results.append({
                    "ID": i,
                    "Soru": q,
                    "Yanıt": response[:150] + "..." if len(response) > 150 else response,
                    "Süre (sn)": round(duration, 2),
                    "Durum": "✅ BAŞARILI" if len(response) > 20 else "⚠️ KISA/EKSIK"
                })
            except Exception as e:
                print(f"❌ Hata: {str(e)}")
                results.append({
                    "ID": i,
                    "Soru": q,
                    "Yanıt": f"HATA: {str(e)}",
                    "Süre (sn)": 0,
                    "Durum": "❌ HATALI"
                })
        
        return results

    def print_results(self, results):
        df = pd.DataFrame(results)
        
        print("\n" + "="*100)
        print("🏆 TEST SONUÇLARI ÖZETİ")
        print("="*100)
        
        # Tabloyu göster
        print(df[["ID", "Soru", "Süre (sn)", "Durum"]].to_string(index=False))
        
        # İstatistikler
        avg_time = df["Süre (sn)"].mean()
        total_success = len(df[df["Durum"] == "✅ BAŞARILI"])
        accuracy_rate = (total_success / len(df)) * 100
        
        print("\n" + "-"*30)
        print(f"📈 Ortalama Yanıt Süresi: {avg_time:.2f} saniye")
        print(f"🎯 Tahmini Başarı Oranı: %{accuracy_rate:.1f}")
        print(f"🏢 Kullanılan Model: {settings.ollama_model}")
        print("-"*30)
        
        print("\n💡 Not: Yanıtların doğruluğunu manuel olarak kontrol etmeniz önerilir.")
        print("====================================================================================================\n")

if __name__ == "__main__":
    benchmark = RAGBenchmark()
    stats = benchmark.system.chroma_manager.get_stats()
    if stats['document_count'] == 0:
        print("\n❌ HATA: ChromaDB'de hiç döküman bulunamadı. Lütfen 'data' klasörüne dosya ekleyiniz.")
    else:
        results = benchmark.run_tests()
        benchmark.print_results(results)
