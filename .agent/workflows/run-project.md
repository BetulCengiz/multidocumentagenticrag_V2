---
description: Projeyi yerel ortamda veya Docker üzerinde çalıştırma adımları.
---

# 🚀 Proje Çalıştırma Rehberi

Bu rehber, **Agentic RAG** projesini sorunsuz bir şekilde ayağa kaldırmanız için gereken adımları içerir.

## 1. Hazırlık (Ortak Adım)

Öncelikle gerekli ortam değişkenlerini yapılandırın:

- `.env.example` dosyasının bir kopyasını oluşturun ve adını `.env` yapın.
- `.env` dosyası içindeki ayarları (model isimleri, portlar vb.) ihtiyacınıza göre düzenleyin.

```powershell
cp .env.example .env
```

---

## 2. Seçenek A: Docker ile Çalıştırma (Önerilen)

Docker kullanarak tüm bağımlılıkları (Ollama, ChromaDB ve Uygulama) tek bir komutla ayağa kaldırabilirsiniz.

### İlk Kurulum ve Model İndirme
Sistemi ilk kez çalıştırıyorsanız, gerekli LLM modelini (örn: llama3.1) indirmek için "init" profilini kullanmalısınız:

// turbo
```powershell
docker-compose --profile init up --build
```

### Normal Çalıştırma
Modeller indirildikten sonra, uygulamayı her zaman şu komutla başlatabilirsiniz:

// turbo
```powershell
docker-compose up
```

- **Gradio Arayüzü:** `http://localhost:7860`
- **Ollama API:** `http://localhost:11434`

---

## 3. Seçenek B: Yerel Python Ortamında Çalıştırma

Eğer Docker kullanmak istemiyorsanız, adımlar şöyledir:

### Gereksinimler
- Bilgisayarınızda **Ollama** yüklü ve çalışıyor olmalıdır.
- Python 3.11+ yüklü olmalıdır.

### Bağımlılıkları Yükleme
// turbo
```powershell
pip install -r requirements.txt
```

### Uygulamayı Başlatma
// turbo
```powershell
python app/main.py
```

---

## 🛠️ Sorun Giderme

- **Ollama Bağlantı Hatası:** `.env` dosyasındaki `OLLAMA_HOST` değerinin doğru olduğundan emin olun (Docker için `http://ollama:11434`, yerel için `http://localhost:11434`).
- **GPU Kullanımı:** Docker kurgusu NVIDIA GPU destekli yapılandırılmıştır. GPU'nuz yoksa `docker-compose.yml` dosyasındaki `deploy` bölümlerini yorum satırı yapabilirsiniz.
- **Modellerin Yüklenmesi:** İlk açılışta modellerin (HuggingFace ve Ollama) indirilmesi zaman alabilir. Lütfen internet bağlantınızı kontrol edin.
