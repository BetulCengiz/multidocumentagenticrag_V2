"""
Limited Şirket Ana Sözleşmesi - Özel Test Scripti
16 Soruluk Kapsamlı Doğruluk ve Hız Testi
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

DOC_NAME = "limitedsirketanasozlesme"

class LTDBenchmark:
    def __init__(self):
        print("\n" + "="*60)
        print(f"🚀 {DOC_NAME.upper()} TESTİ BAŞLATILIYOR")
        print("="*60)
        
        self.system = AgenticRAGSystem()
        self.system.ingest_all_documents()
        self.system.initialize_agent()
        self.test_set = [
            {"q": "Şirketin toplam sermayesi ne kadardır?", "expected": "5.000,00 TL"},
            {"q": "Şirketin süresi kaç yıldır?", "expected": "99 yıl"},
            {"q": "Madde 6’da hangi hususlar düzenlenmiştir?", "expected": "Esas sermaye, pay dağılımı, ödeme süresi vb."},
            {"q": "Rekabet yasağı hangi maddede yer almaktadır?", "expected": "Madde 9"},
            {"q": "Bir ortak payını devretmek isterse hangi prosedürü izlemelidir?", "expected": "Noter onayı, %75 ortak onayı, tescil"},
            {"q": "Şirket feshedildiğinde tasfiye süreci nasıl yürütülür?", "expected": "Tasfiye memuru ataması, borç ödeme, kalan dağıtımı"},
            {"q": "Bir ortak şirketle rekabet ederse hangi sonuç doğar?", "expected": "Zarar tazmini talebi"},
            {"q": "Müdürlerin münferit temsil yetkisi hangi riski doğurabilir?", "expected": "Kontrol zayıflığı, finansal risk"},
            {"q": "Sözleşmeyi 3–4 cümle ile özetleyiniz.", "expected": "Gıda/inşaat/nakliye alanı, 5 bin TL sermaye, 99 yıl süre..."},
            {"q": "Kar dağıtımı nasıl yapılır?", "expected": "%5 yedek akçe sonrası sermaye payı oranında"},
            {"q": "Müdürlerin görev süresi belirlenmiş midir?", "expected": "Belirtilmemiş"},
            {"q": "Temsil yetkisi müşterek mi yoksa münferit midir?", "expected": "Münferit"},
            {"q": "Sözleşmede tahkim şartı var mıdır?", "expected": "Hayır"},
            {"q": "Şirket sermayesi 50.000 TL midir?", "expected": "Hayır (5.000 TL)"},
            {"q": "Şirket zarar etmişse kar dağıtımı yapılabilir mi?", "expected": "Hayır"},
            {"q": "Müdürlerden biri tek başına kredi çekebilir mi?", "expected": "Evet (Münferit yetki)"}
        ]

    def run_tests(self):
        results = []
        total_start = time.time()
        
        for i, item in enumerate(self.test_set, 1):
            q = item["q"]
            expected = item["expected"]
            print(f"\n[{i}/16] Soru: {q}")
            
            start_time = time.time()
            try:
                response_data = self.system.agent.query(q, DOC_NAME)
                response = response_data["response"]
                end_time = time.time()
                
                duration = end_time - start_time
                print(f"⏱️ Süre: {duration:.2f}s")
                print(f"📝 Yanıt: {response[:200]}...")
                
                results.append({
                    "No": i,
                    "Soru": q,
                    "Beklenen (Özet)": expected,
                    "Sistem Yanıtı": response,
                    "Süre (sn)": round(duration, 2)
                })
            except Exception as e:
                print(f"❌ Hata: {str(e)}")
                results.append({
                    "No": i, "Soru": q, "Beklenen (Özet)": expected,
                    "Sistem Yanıtı": f"HATA: {str(e)}", "Süre (sn)": 0
                })
        
        total_duration = time.time() - total_start
        return results, total_duration

    def report(self, results, total_duration):
        df = pd.DataFrame(results)
        output_file = f"tests/results_{DOC_NAME}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        print("\n" + "="*80)
        print(f"🏆 {DOC_NAME.upper()} TEST SONUÇLARI")
        print("="*80)
        print(df[["No", "Süre (sn)"]].to_string(index=False))
        
        print(f"\n📈 Toplam Test Süresi: {total_duration/60:.2f} dakika")
        print(f"📈 Ortalama Soru Süresi: {df['Süre (sn)'].mean():.2f} saniye")
        print(f"📁 Detaylı rapor kaydedildi: {output_file}")
        print("="*80 + "\n")

if __name__ == "__main__":
    test = LTDBenchmark()
    results, total_duration = test.run_tests()
    test.report(results, total_duration)
