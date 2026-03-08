from functools import lru_cache

from langchain_deepseek import ChatDeepSeek
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_CHAT_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
)


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.2) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=DEEPSEEK_CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        temperature=temperature,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
