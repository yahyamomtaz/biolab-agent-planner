"""retrieve_protocol tool  -  RAG over Qdrant collection ``protocols``."""

from __future__ import annotations

import functools
import os

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from biolab_agent.config import load_settings
from biolab_agent.schemas import ProtocolHit
from biolab_agent.tools.rag_hybrid import _expand_query

COLLECTION = "protocols"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@functools.lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@functools.lru_cache(maxsize=1)
def _client() -> QdrantClient:
    settings = load_settings()
    return QdrantClient(url=settings.qdrant_url, timeout=10.0)


@functools.lru_cache(maxsize=1)
def _reranker(model_id: str) -> CrossEncoder:
    settings = load_settings()
    device = "cuda" if settings.biolab_device == "cuda" else "cpu"
    return CrossEncoder(model_id, device=device)


_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _reranker_model() -> str:
    raw = os.environ.get("BIOLAB_RERANKER_MODEL")
    if raw is None:
        return _DEFAULT_RERANKER_MODEL
    return raw.strip()


def _rerank_candidates() -> int:
    raw = os.environ.get("BIOLAB_RERANK_CANDIDATES", "20")
    try:
        return max(1, min(int(raw), 25))
    except ValueError:
        return 20


def _dense_retrieve_protocol(query: str, limit: int) -> list[ProtocolHit]:
    vec = _embedder().encode([_expand_query(query)], normalize_embeddings=True).tolist()[0]

    client = _client()
    resp = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=max(1, min(limit, 25)),
        with_payload=True,
    )
    out: list[ProtocolHit] = []
    for h in resp.points:
        payload = h.payload or {}
        out.append(
            ProtocolHit(
                doc_id=str(payload.get("doc_id", "")),
                chunk_id=str(payload.get("chunk_id", "")),
                title=str(payload.get("title", "")),
                source_url=payload.get("source_url"),
                text=str(payload.get("text", ""))[:1200],
                score=float(h.score),
            )
        )
    return out


def retrieve_protocol(query: str, k: int = 5) -> list[ProtocolHit]:
    """Return the top-``k`` protocol chunks, optionally reranked by a cross-encoder."""
    if not query.strip():
        return []

    model_id = _reranker_model()
    first_stage_k = max(k, _rerank_candidates()) if model_id else k
    hits = _dense_retrieve_protocol(query, first_stage_k)
    if not model_id or len(hits) <= 1:
        return hits[:k]

    pairs = [[query, f"{hit.title}\n{hit.text}"] for hit in hits]
    scores = _reranker(model_id).predict(pairs)
    reranked = sorted(
        zip(hits, scores, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [hit for hit, _score in reranked[:k]]


retrieve_protocol_spec = {
    "type": "function",
    "function": {
        "name": "retrieve_protocol",
        "description": (
            "Search the local protocol library (indexed from OpenTrons) and "
            "return the top-k most relevant protocol chunks with their doc_id, "
            "title, source URL, and a text excerpt. Use this when the user asks "
            "for a procedure, SOP, or reference protocol."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of what to find.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many hits to return (1-10).",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}
