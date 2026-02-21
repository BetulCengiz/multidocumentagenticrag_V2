# Core components for AgenticRAG
from app.core.embeddings import get_embedding_model
from app.core.llm import get_llm
from app.core.prompts import (
    TURKISH_SYSTEM_PROMPT,
    TURKISH_QA_TEMPLATE,
    TURKISH_REFINE_TEMPLATE,
    TURKISH_SUMMARY_TEMPLATE,
)