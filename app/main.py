from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.retrieval import RankedDocument, create_hybrid_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not settings.persist_dir.exists():
    raise RuntimeError(
        f"Vector database not found: {settings.persist_dir}. "
        "Run `python scripts/build_index.py` first."
    )

embedding = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"device": settings.embedding_device},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name=settings.collection_name,
    embedding_function=embedding,
    persist_directory=str(settings.persist_dir),
)
retriever = create_hybrid_retriever(vector_store, embedding)

llm = ChatOllama(
    model=settings.llm_model,
    temperature=0,
)

PROMPT = ChatPromptTemplate.from_template(
    """
你是企业知识库助手。必须遵守以下规则：
1. 只能使用【参考资料】中的事实回答，不得补充资料外的数字、比例或建议。
2. 引用资料时使用 [资料1]、[资料2] 这样的编号。
3. 如果参考资料不足以回答问题，只回答："知识库中未找到相关信息"。

【参考资料】
{context}

【问题】
{question}
"""
)

chain = PROMPT | llm
app = FastAPI(
    title="Enterprise RAG Knowledge Q&A API",
    description=(
        "Hybrid retrieval, reranking and grounded answers "
        "for enterprise knowledge bases."
    ),
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


def retrieve_documents(question: str, top_k: int) -> list[RankedDocument]:
    return retriever.retrieve(question, top_k)


def build_sources(matches: list[RankedDocument]) -> list[dict[str, Any]]:
    sources = []
    for index, candidate in enumerate(matches, start=1):
        metadata = candidate.document.metadata
        sources.append(
            {
                "id": index,
                "source": metadata.get("source", "未知来源"),
                "page": metadata.get("page"),
                "category": metadata.get("category"),
                "content": candidate.document.page_content,
                "vector_rank": candidate.vector_rank,
                "vector_distance": candidate.vector_distance,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
                "rrf_score": round(candidate.rrf_score, 6),
                "rerank_score": candidate.rerank_score,
            }
        )
    return sources


def build_context(matches: list[RankedDocument]) -> str:
    parts = []
    for index, candidate in enumerate(matches, start=1):
        metadata = candidate.document.metadata
        source = metadata.get("source", "未知来源")
        page = metadata.get("page")
        page_text = f" 第{page}页" if page is not None else ""
        parts.append(
            f"【资料{index}】（出处：{source}{page_text}）\n"
            f"{candidate.document.page_content}"
        )
    return "\n\n".join(parts)


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def get_matches(question: str, top_k: int) -> list[RankedDocument]:
    try:
        return await asyncio.to_thread(retrieve_documents, question, top_k)
    except Exception as exc:
        logger.exception("Vector retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="知识库检索失败，请稍后重试。",
        ) from exc


@app.get("/")
def root():
    return {
        "service": "Enterprise RAG Knowledge Q&A API",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    matches = await get_matches(req.question, req.top_k)
    sources = build_sources(matches)
    if not matches:
        return {"answer": "知识库中未找到相关信息", "sources": []}

    try:
        result = await chain.ainvoke(
            {"context": build_context(matches), "question": req.question}
        )
    except Exception as exc:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=503, detail="模型服务暂时不可用。") from exc

    return {"answer": result.content.strip(), "sources": sources}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    matches = await get_matches(req.question, req.top_k)
    sources = build_sources(matches)

    async def generate():
        yield sse({"type": "sources", "sources": sources})
        if not matches:
            yield sse({"type": "token", "content": "知识库中未找到相关信息"})
            yield sse({"type": "done"})
            return

        try:
            async for chunk in chain.astream(
                {"context": build_context(matches), "question": req.question}
            ):
                if chunk.content:
                    yield sse({"type": "token", "content": chunk.content})
        except Exception:
            logger.exception("Streaming LLM request failed")
            yield sse({"type": "error", "message": "模型服务中断，请稍后重试。"})
        finally:
            yield sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": settings.collection_name,
        "chunks": vector_store._collection.count(),
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "max_distance": settings.max_vector_distance,
        "retrieval": "vector+bm25+rrf+reranker",
        "reranker_model": retriever.reranker_model_name,
        "rerank_enabled": retriever.enable_rerank,
        "rerank_min_score": retriever.rerank_min_score,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("RAG_API_HOST", "127.0.0.1"),
        port=int(os.getenv("RAG_API_PORT", "8000")),
    )
