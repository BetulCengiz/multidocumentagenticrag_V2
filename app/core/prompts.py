"""
Agentic RAG - Türkçe Prompt Şablonları
Sözleşme ve hukuki metinler için optimize edilmiş promptlar.
"""

from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType


TURKISH_SYSTEM_PROMPT = """Sen Türkçe sözleşme ve hukuki metinleri analiz eden uzman bir yapay zeka asistanısın.

TEMEL KURALLAR:
1. SADECE verilen bağlam metnini kullan
2. Bağlamda olmayan bilgiyi KESİNLİKLE UYDURMA
3. Başka dökümanlardan veya genel bilginden CEVAP VERME
4. Emin değilsen "Bu bilgi verilen metinde bulunmuyor" de

Yanıt verirken:
- Her zaman Türkçe yanıt ver
- Madde numaralarına atıf yap (örn: "Madde 3'te belirtildiği üzere...")
- Alıntı yapıyorsan tırnak içinde göster
- Kısa ve öz ol, gereksiz uzatma
"""


TURKISH_QA_TEMPLATE = PromptTemplate(
    template="""Aşağıda bir sözleşme metninden alınan bölüm verilmiştir:

---------------------
BAĞLAM METNİ:
{context_str}
---------------------

ÖNEMLİ KURALLAR:
1. SADECE yukarıdaki bağlam metnini kullanarak yanıt ver
2. Bağlamda bulunmayan bilgiyi UYDURMA
3. Başka sözleşmelerden veya genel bilgiden yararlanma
4. Yanıtın tamamen bağlama dayalı olmalı

SORU: {query_str}

Eğer soru bağlamdaki bilgiyle yanıtlanamıyorsa, "Bu bilgi verilen metinde bulunmuyor" yaz.
Eğer yanıtlayabiliyorsan, ilgili madde numarasını belirt.

TÜRKÇE YANIT:""",
    prompt_type=PromptType.QUESTION_ANSWER,
)

TURKISH_REFINE_TEMPLATE = PromptTemplate(
    template="""Orijinal soru: {query_str}

Mevcut yanıtımız: {existing_answer}

Aşağıda EK bağlam bilgisi verilmiştir:
---------------------
{context_msg}
---------------------

GÖREV:
- Ek bağlam SADECE mevcut yanıtı destekliyorsa veya tamamlıyorsa kullan
- Ek bağlam farklı bir dökümanla ilgiliyse KULLANMA
- Çelişkili bilgi varsa mevcut yanıtı koru
- Yanıtı Türkçe olarak ver

GELİŞTİRİLMİŞ TÜRKÇE YANIT:""",
    prompt_type=PromptType.REFINE,
)


TURKISH_SUMMARY_TEMPLATE = PromptTemplate(
    template="""Aşağıdaki sözleşme veya hukuki metin bölümünü Türkçe olarak özetle.

METİN:
{context_str}

Özet şunları içermeli (metinde varsa):
- Sözleşmenin/dökümanın türü
- Taraflar
- Ana konu ve amaç
- Önemli maddeler
- Kritik tarihler veya süreler

TÜRKÇE ÖZET:""",
    prompt_type=PromptType.SUMMARY,
)

TURKISH_TREE_SUMMARIZE_TEMPLATE = PromptTemplate(
    template="""Aşağıda bir dökümanın farklı bölümlerinden alınan parçalar verilmiştir:

---------------------
{context_str}
---------------------

GÖREV:
Bu parçaları kullanarak dökümanın profesyonel ve kapsayıcı bir özetini oluştur. 
Özet şunları KESİNLİKLE içermeli:
1. Sözleşmenin tarafları (Tam isimler)
2. Sözleşmenin ana konusu ve amacı
3. Sözleşme bedeli ve ödeme koşulları
4. Kritik süreler, teslimatlar ve fesih koşulları

Yanıtı başlıklar halinde ve profesyonel bir hukuk diliyle ver.

TÜRKÇE ÖZET:""",
    prompt_type=PromptType.SUMMARY,
)