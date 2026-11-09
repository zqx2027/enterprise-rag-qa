from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    persist_dir: Path = Path(
        os.getenv("RAG_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma"))
    ).expanduser()
    collection_name: str = os.getenv("RAG_COLLECTION_NAME", "company_docs")
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "BAAI/bge-large-zh-v1.5",
    )
    embedding_device: str = os.getenv("RAG_DEVICE", "cpu")
    llm_model: str = os.getenv("RAG_MODEL", "deepseek-r1:7b")
    max_vector_distance: float = float(os.getenv("RAG_MAX_DISTANCE", "0.90"))
    vector_top_k: int = int(os.getenv("RAG_VECTOR_TOP_K", "8"))
    bm25_top_k: int = int(os.getenv("RAG_BM25_TOP_K", "8"))
    rrf_rank_constant: int = int(os.getenv("RAG_RRF_RANK_CONSTANT", "60"))
    enable_rerank: bool = env_bool("RAG_ENABLE_RERANK", False)
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL",
        "BAAI/bge-reranker-large",
    )
    reranker_device: str = os.getenv("RAG_RERANKER_DEVICE", "cpu")
    reranker_local_only: bool = env_bool("RAG_RERANKER_LOCAL_ONLY", True)
    rerank_min_score: float = float(os.getenv("RAG_RERANK_MIN_SCORE", "0.45"))
    api_url: str = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


settings = Settings()
