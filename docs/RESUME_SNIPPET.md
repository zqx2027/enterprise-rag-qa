# 简历项目描述

仓库地址：https://github.com/zqx2027/enterprise-rag-qa

## 中文版本

**企业知识问答系统（RAG）**｜Python、FastAPI、LangChain、Chroma、Streamlit｜[GitHub](https://github.com/zqx2027/enterprise-rag-qa)

- 构建企业制度文档的检索增强问答系统，覆盖 Markdown、TXT、PDF、DOCX 加载、递归分块、向量建库、流式回答与引用溯源。
- 使用 BGE 中文向量模型与 BM25 双路召回，通过 RRF 融合排名，并支持 BGE CrossEncoder 二阶段重排。
- 设计 8 条检索评测集，在虚构示例知识库上达到 `Hit@3 = 1.000`、`MRR@3 = 1.000`。
- 实现 FastAPI SSE 流式接口、检索分数透出、配置化离线模型接入和项目自检脚本。

## English Version

**Enterprise Knowledge Q&A System (RAG)** | Python, FastAPI, LangChain, Chroma, Streamlit | [GitHub](https://github.com/zqx2027/enterprise-rag-qa)

- Built a retrieval-augmented question answering system for enterprise policy documents, covering ingestion, recursive chunking, vector indexing, streaming responses, and source attribution.
- Combined BGE embeddings and BM25 retrieval with reciprocal rank fusion, with optional cross-encoder reranking.
- Created an 8-question retrieval benchmark and measured `Hit@3 = 1.000` and `MRR@3 = 1.000` on a synthetic demonstration corpus.
- Delivered a FastAPI SSE API, retrieval diagnostics, configurable offline model support, and a local validation script.
