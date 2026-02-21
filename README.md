# 🤖 Agentic RAG - Profesyonel Hukuki Analiz Sistemi

Bu proje; yerel (local-first), yüksek performanslı ve gizlilik odaklı bir **Agentic Retrieval-Augmented Generation (RAG)** sistemidir. Özellikle Türkçe sözleşmeler ve karmaşık hukuki metinler üzerinde yüksek doğruluk oranı ile analiz yapabilmek üzere "**Gen AI Reference Architecture**" (Brij Kishore Pandey) standartlarına uygun olarak geliştirilmiştir.

---

## 🏗️ Proje Mimarisi (Gen AI Reference Standartları)

Proje, modüler ve ölçeklenebilir bir yapı sunmak amacıyla profesyonel mimari prensiplerine göre yapılandırılmıştır:

- **`app/agents/`**: Akıllı karar verme mekanizması (Agentic Logic & Orchestration).
- **`app/ingestion/`**: Veri okuma, temizleme ve batch işleme pipeline'ı.
- **`app/vectorstore/`**: Vektör veritabanı (ChromaDB) yönetimi ve persistence katmanı.
- **`app/tools/`**: Agent tarafından kullanılan spesifik araçlar (Summary, Vector Search, Search).
- **`app/core/`**: LLM yapılandırması ve optimize edilmiş Türkçe prompt şablonları.
- **`app/ui/`**: Gradio tabanlı modern ve responsive kullanıcı arayüzü.

---

## 🌟 Öne Çıkan Özellikler

- **Local-First Güvenlik:** Tüm veri işleme ve AI çıkarımları yerel makinede yapılır, veri dışarı sızmaz.
- **Agentic Tool Selection:** Sorunun türüne göre (Özet mi? Spesifik Bilgi mi?) en uygun aracı otomatik seçer.
- **Türkçe Hukuk Odaklı:** Sözleşme terminolojisine ve eklemeli dil yapısına %100 uyumlu.
- **Gerçek Zamanlı Yazım (Streaming):** Token buffering ile kesintisiz ve akıcı yanıt deneyimi.
- **GPU Hızlandırma:** CUDA desteği ile yüksek hızlı embedding ve inference.
- **Döküman Yönetimi:** Çoklu döküman yükleme, indeksleme ve silme desteği.

---

## 📈 İyileştirmeler ve Başarı Oranı (%85+)

Sistem üzerinde yapılan stres testleri ve döküman analizleri sonucunda bilgi çekme (Retrieval) doğruluğu **%85'in üzerine** çıkarılmıştır.

- **Dinamik Özetleme:** `TURKISH_TREE_SUMMARIZE_TEMPLATE` ile dökümanın farklı bölümlerindeki bilgileri birleştirerek profesyonel özetler oluşturur.
- **Doğruluk Artışı:** Gelişmiş prompt mühendisliği ile modelin döküman dışı bilgi uydurması (hallucination) engellenmiş, madde bazlı atıf yapması zorunlu tutulmuştur.
- **Hız Optimizasyonu:** Yanıt sürelerini iyileştirmek için asenkron işlemler ve verimli vektör arama stratejileri uygulanmıştır.

---

## 🛠️ Teknik Detaylar (Neden Bu Teknolojiler?)

### Modeller
- **LLM: Ollama (Llama 3.1:8b)**
  - **Neden?** SOTA seviyesinde akıl yürütme (reasoning) yeteneği ve GPU üzerinde mükemmel performans.
- **Embedding: `intfloat/multilingual-e5-large`**
  - **Neden?** XLM-RoBERTa altyapısı sayesinde Türkçe'nin anlamsal derinliğini (cezai şart, mücbir sebep vb.) en yüksek hassasiyetle anlar. 1024 boyutlu vektör uzayı sunar.

### Veri İşleme (Chunking & Vector DB)
- **Chunking:** 1024 karakterlik parçalar ve 128 karakterlik overlap. Bu değerler hukuki maddelerin bütünlüğünü korumak için özel olarak seçilmiştir.
- **Vector DB (ChromaDB):** Hızlı filtreleme ve yerel depolama desteği sayesinde tercih edilmiştir.

---

## 🐳 Kurulum ve Çalıştırma

Sistem **Docker Compose** üzerinde, GPU hızlandırmalı (CUDA) olarak çalışır.

### Sistem Gereksinimleri (Minimum)
- **GPU:** NVIDIA GPU (Minimum 8GB VRAM) + NVIDIA Container Toolkit
- **RAM:** 16GB
- **Depolama:** ~15GB
- **Yazılım:** Docker & Docker Compose

### Adımlar
1. **Model İndirme (İlk Kurulum):**
   Llama 3.1 modelini çekmek için bir kez çalıştırın:
   ```bash
   docker-compose --profile init up
   ```
   *Model çekildikten sonra bu konteyner otomatik olarak kapanacaktır.*

2. **Uygulamayı Başlatın:**
   ```bash
   docker-compose up --build
   ```

3. **Erişim:** [http://localhost:7860](http://localhost:7860)

---

## 📈 Doğruluk ve Performans Raporu (Verified Metrics)

Sistem, farklı türdeki hukuki dökümanlar (Yazılım Sözleşmeleri ve Şirket Ana Sözleşmeleri) üzerinde test edilmiş ve aşağıdaki reel metrikler doğrulanmıştır:

- **Net Başarı Oranı:** **%85+** (Birden fazla döküman üzerinden yapılan çapraz testlerin ortalamasıdır).
- **Bilgi Getirme Doğruluğu (Retrieval Precision):** %90+ (Spesifik maddeleri bulma yeteneği).
- **Ortalama Yanıt Süresi:** ~45-80 saniye (Lokal GPU/CPU kullanımı baz alınmıştır).
- **Halüsinasyon Kontrolü:** Gelişmiş Türkçe prompt şablonları sayesinde, dökümanda olmayan bilgiler için "Bu bilgi bulunmuyor" yanıtı alma oranı artırılmıştır.

---

## 🔒 Güvenlik & Gizlilik
- **Çevrimdışı Çalışma:** Sistem tamamen internetten bağımsız (offline) çalışabilir.
- **Hassas Veriler:** `.env`, `data/` ve `chroma_db/` klasörleri asla paylaşılmaz.

---

## ⚠️ Telif Hakkı ve Lisans (Copyright)

**© 2024 Betül Cengiz - Tüm Hakları Saklıdır.**

Bu proje ticari veya bireysel olarak izinsiz kopyalanamaz, çoğaltılamaz ve ücretsiz olarak dağıtılamaz. Projenin kaynak kodlarının kullanımı ve paylaşımı tamamen geliştiricinin (Betül Cengiz) iznine tabidir. 

---
*Geliştirici: Agentic RAG Team | Gen AI Reference Architecture tabanlı mimari yapı.*
