from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.retrieval import create_hybrid_retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "questions.json",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.questions.read_text(encoding="utf-8"))

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

    reciprocal_ranks = []
    hits = 0
    for case in cases:
        expected_sources = set(case["expected_sources"])
        matches = retriever.retrieve(case["question"], args.top_k)
        first_rank = next(
            (
                rank
                for rank, candidate in enumerate(matches, start=1)
                if candidate.document.metadata.get("source") in expected_sources
            ),
            None,
        )
        if first_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

        status = f"rank={first_rank}" if first_rank else "miss"
        print(f"{case['question']} -> {status}")

    hit_rate = hits / len(cases)
    mrr = sum(reciprocal_ranks) / len(cases)
    print(f"\nHit@{args.top_k}: {hit_rate:.3f}")
    print(f"MRR@{args.top_k}: {mrr:.3f}")


if __name__ == "__main__":
    main()
