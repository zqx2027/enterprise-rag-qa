from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


def load_file(path: Path, source_root: Path) -> list[Document]:
    relative_source = path.relative_to(source_root).as_posix()
    category = path.parent.name if path.parent != source_root else "general"
    base_metadata = {"source": relative_source, "category": category}

    if path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return (
            [Document(page_content=text, metadata=base_metadata)] if text.strip() else []
        )

    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        documents = []
        for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={**base_metadata, "page": page_number},
                    )
                )
        return documents

    if path.suffix.lower() == ".docx":
        from docx import Document as DocxDocument

        word_document = DocxDocument(str(path))
        parts = [paragraph.text for paragraph in word_document.paragraphs]
        for table in word_document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        text = "\n".join(part for part in parts if part.strip())
        return (
            [Document(page_content=text, metadata=base_metadata)] if text.strip() else []
        )

    return []


def load_documents(source_root: Path) -> list[Document]:
    documents = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        loaded = load_file(path, source_root)
        documents.extend(loaded)
        print(f"Loaded {path.relative_to(source_root)}: {len(loaded)} section(s)")
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    ids = []
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index
        identity = (
            f"{chunk.metadata.get('source')}:"
            f"{chunk.metadata.get('page')}:"
            f"{chunk.metadata.get('start_index')}:"
            f"{chunk.page_content}"
        )
        ids.append(hashlib.sha1(identity.encode("utf-8")).hexdigest())
    return chunks, ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Chroma knowledge index.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_docs",
        help="Directory containing Markdown, TXT, PDF or DOCX files.",
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild the vector store from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.docs_dir.expanduser().resolve()
    if not source_root.exists():
        raise SystemExit(f"Document directory not found: {source_root}")

    documents = load_documents(source_root)
    if not documents:
        raise SystemExit(f"No supported documents found in: {source_root}")

    chunks, ids = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.reset and settings.persist_dir.exists():
        print(f"Removing existing vector store: {settings.persist_dir}")
        shutil.rmtree(settings.persist_dir)

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
    vector_store.add_documents(chunks, ids=ids)

    print(
        f"Indexed {len(chunks)} chunks from {len(documents)} sections "
        f"into {settings.persist_dir}"
    )


if __name__ == "__main__":
    main()
