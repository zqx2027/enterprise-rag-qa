from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import jieba
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


def tokenize_chinese(text: str) -> list[str]:
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


@dataclass(slots=True)
class RankedDocument:
    document_id: str
    document: Document
    vector_rank: int | None = None
    vector_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class HybridRetriever:
    def __init__(
        self,
        vector_store: Chroma,
        embedding: Embeddings,
        *,
        max_vector_distance: float = 0.90,
        vector_top_k: int = 8,
        bm25_top_k: int = 8,
        rrf_rank_constant: int = 60,
        rerank_min_score: float = 0.45,
        reranker_model: str = "BAAI/bge-reranker-large",
        reranker_device: str = "cpu",
        reranker_local_only: bool = True,
        enable_rerank: bool = False,
    ) -> None:
        self.vector_store = vector_store
        self.embedding = embedding
        self.max_vector_distance = max_vector_distance
        self.vector_top_k = max(1, vector_top_k)
        self.bm25_top_k = max(1, bm25_top_k)
        self.rrf_rank_constant = max(1, rrf_rank_constant)
        self.rerank_min_score = rerank_min_score
        self.reranker_model_name = reranker_model
        self.reranker_device = reranker_device
        self.reranker_local_only = reranker_local_only
        self.enable_rerank = enable_rerank

        self.documents, self.document_by_id = self._load_documents()
        tokenized_documents = [
            tokenize_chinese(document.page_content) for document in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents)

        self._reranker: CrossEncoder | None = None
        self._reranker_lock = threading.Lock()

    def _load_documents(self) -> tuple[list[Document], dict[str, Document]]:
        result = self.vector_store.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        contents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        documents = []
        document_by_id = {}
        for document_id, content, metadata in zip(ids, contents, metadatas):
            if not content:
                continue
            document = Document(
                id=document_id,
                page_content=content,
                metadata=metadata or {},
            )
            documents.append(document)
            document_by_id[document_id] = document

        if not documents:
            raise RuntimeError("The Chroma collection does not contain any documents.")
        return documents, document_by_id

    def _vector_search(self, question: str) -> list[tuple[str, float]]:
        query_embedding = self.embedding.embed_query(question)
        result = self.vector_store._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.vector_top_k, len(self.documents)),
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches = []
        for document_id, distance in zip(ids, distances):
            distance = float(distance)
            if (
                document_id in self.document_by_id
                and distance <= self.max_vector_distance
            ):
                matches.append((document_id, distance))
        return matches

    def _bm25_search(self, question: str) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize_chinese(question))
        ranked = sorted(
            (
                (self.documents[index].id, float(score))
                for index, score in enumerate(scores)
                if score > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[: self.bm25_top_k]

    def _fuse_rrf(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
    ) -> list[RankedDocument]:
        candidates: dict[str, RankedDocument] = {}

        for rank, (document_id, distance) in enumerate(vector_results, start=1):
            candidate = candidates.setdefault(
                document_id,
                RankedDocument(
                    document_id=document_id,
                    document=self.document_by_id[document_id],
                ),
            )
            candidate.vector_rank = rank
            candidate.vector_distance = distance
            candidate.rrf_score += 1 / (self.rrf_rank_constant + rank)

        for rank, (document_id, score) in enumerate(bm25_results, start=1):
            candidate = candidates.setdefault(
                document_id,
                RankedDocument(
                    document_id=document_id,
                    document=self.document_by_id[document_id],
                ),
            )
            candidate.bm25_rank = rank
            candidate.bm25_score = score
            candidate.rrf_score += 1 / (self.rrf_rank_constant + rank)

        return sorted(candidates.values(), key=lambda item: item.rrf_score, reverse=True)

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    logger.info("Loading reranker model: %s", self.reranker_model_name)
                    self._reranker = CrossEncoder(
                        self.reranker_model_name,
                        device=self.reranker_device,
                        local_files_only=self.reranker_local_only,
                    )
        return self._reranker

    def _rerank(
        self,
        question: str,
        candidates: list[RankedDocument],
    ) -> list[RankedDocument]:
        if not candidates:
            return []

        reranker = self._get_reranker()
        pairs = [
            (question, candidate.document.page_content) for candidate in candidates
        ]
        with self._reranker_lock:
            scores = reranker.predict(pairs, show_progress_bar=False)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        ranked = sorted(
            candidates,
            key=lambda item: item.rerank_score or float("-inf"),
            reverse=True,
        )
        return [
            candidate
            for candidate in ranked
            if candidate.rerank_score is not None
            and candidate.rerank_score >= self.rerank_min_score
        ]

    def retrieve(self, question: str, top_k: int) -> list[RankedDocument]:
        vector_results = self._vector_search(question)
        bm25_results = self._bm25_search(question)
        candidates = self._fuse_rrf(vector_results, bm25_results)

        if self.enable_rerank:
            ranked = self._rerank(question, candidates)
        else:
            ranked = candidates

        return ranked[: max(1, top_k)]


def create_hybrid_retriever(
    vector_store: Chroma,
    embedding: Embeddings,
) -> HybridRetriever:
    return HybridRetriever(
        vector_store=vector_store,
        embedding=embedding,
        max_vector_distance=settings.max_vector_distance,
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_rank_constant=settings.rrf_rank_constant,
        rerank_min_score=settings.rerank_min_score,
        reranker_model=settings.reranker_model,
        reranker_device=settings.reranker_device,
        reranker_local_only=settings.reranker_local_only,
        enable_rerank=settings.enable_rerank,
    )
