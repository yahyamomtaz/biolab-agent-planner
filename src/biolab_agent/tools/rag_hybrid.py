"""retrieve_protocol tool with dense, BM25 hybrid, and reranking."""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from biolab_agent.config import load_settings
from biolab_agent.schemas import ProtocolHit

COLLECTION = "protocols"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_RRF_K = 60
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "in",
    "is",
    "its",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


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


_LINKS_BLOCK_RE = re.compile(
    r"Links:\s*(?:(?:\*\s*\[[^\]]+\]\([^)]+\)\s*)|(?:<br\s*/?>\s*)|(?:</br>\s*))+",
    re.IGNORECASE,
)
_BR_TAG_RE = re.compile(r"<br\s*/?>|</br>", re.IGNORECASE)


def _clean_doc_text(text: str) -> str:
    """Strip Markdown navigation noise (Links: bullet lists, <br> tags).

    Applied before BM25 tokenisation and before forming the reranker pair so
    sibling-protocol names in a Links: block do not steal relevance signal from
    the actual protocol description.
    """
    if not text:
        return text
    cleaned = _LINKS_BLOCK_RE.sub("", text)
    cleaned = _BR_TAG_RE.sub(" ", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    ]


@functools.lru_cache(maxsize=256)
def _llm_expand_query(query: str) -> tuple[str, ...]:
    """Ask the configured LLM for 2-3 retrieval paraphrases of ``query``.

    Opt-in via ``BIOLAB_QUERY_EXPANSION=llm``. Best-effort: any error or
    empty response yields ``()``, in which case the caller falls back to the
    original query unchanged. The tuple return makes it cache-safe.
    """
    if os.environ.get("BIOLAB_QUERY_EXPANSION", "").strip().lower() != "llm":
        return ()
    try:
        from biolab_agent.llm import get_client

        client = get_client()
        settings = load_settings()
        system = (
            "Rewrite a lab-protocol search query into 2-3 short paraphrases "
            "to improve retrieval. Preserve technical terms; expand common "
            "abbreviations only when unambiguous. Output one paraphrase per "
            "line, no numbering, no commentary."
        )
        resp = client.chat(
            model=settings.biolab_llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.2, "num_predict": 80},
        )
        text = (resp.get("message") or {}).get("content", "") or ""
        lines = [ln.strip(" -*\t").strip() for ln in text.splitlines() if ln.strip()]
        return tuple(ln for ln in lines if ln and ln.lower() != query.lower())[:3]
    except Exception:
        return ()


def _expand_query(query: str) -> str:
    expansions = _llm_expand_query(query)
    if not expansions:
        return query
    return " ".join([query, *expansions])


def _title_overlap_boost(query: str, hit: ProtocolHit) -> float:
    """Small generic boost when query tokens overlap title tokens.

    No synonym lists, no corpus-specific phrases — just a Jaccard-style
    overlap so titles sharing words with the query rank slightly higher.
    """
    title_tokens = set(_tokenize(hit.title))
    query_tokens = set(_tokenize(query))
    if not title_tokens or not query_tokens:
        return 0.0
    return 2.0 * (len(query_tokens & title_tokens) / len(query_tokens))


@functools.lru_cache(maxsize=1)
def _load_protocol_docs() -> list[ProtocolHit]:
    settings = load_settings()
    path = Path(settings.biolab_data_dir) / "protocols" / "opentrons.jsonl"
    docs: list[ProtocolHit] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            doc_id = str(row.get("doc_id", ""))
            if not doc_id:
                continue
            docs.append(
                ProtocolHit(
                    doc_id=doc_id,
                    chunk_id=str(row.get("chunk_id") or f"{doc_id}:0"),
                    title=str(row.get("title", "")),
                    source_url=row.get("source_url"),
                    text=str(row.get("text", ""))[:1200],
                    score=0.0,
                )
            )
    if not docs:
        raise ValueError(f"No protocol documents loaded from {path}")
    return docs


@functools.lru_cache(maxsize=1)
def _bm25_index() -> tuple[list[ProtocolHit], BM25Okapi]:
    docs = _load_protocol_docs()
    # Repeat titles so exact protocol-name matches can beat generic body hits.
    tokenized = [
        _tokenize(f"{d.title} {d.title} {d.title} {_clean_doc_text(d.text)}")
        for d in docs
    ]
    return docs, BM25Okapi(tokenized)


_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _reranker_model() -> str:
    raw = os.environ.get("BIOLAB_RERANKER_MODEL")
    if raw is None:
        return _DEFAULT_RERANKER_MODEL
    return raw.strip()


def _hybrid_enabled() -> bool:
    raw = os.environ.get("BIOLAB_HYBRID_RETRIEVAL", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def _bm25_retrieve_protocol(query: str, limit: int) -> list[ProtocolHit]:
    docs, bm25 = _bm25_index()
    bm25_scores = bm25.get_scores(_tokenize(_expand_query(query)))
    scores = [
        float(score) + _title_overlap_boost(query, doc)
        for doc, score in zip(docs, bm25_scores, strict=True)
    ]
    ranked = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[: max(1, min(limit, 25))]

    hits: list[ProtocolHit] = []
    for idx, score in ranked:
        if float(score) <= 0.0:
            continue
        hits.append(docs[idx].model_copy(update={"score": float(score)}))
    return hits


def _rrf_merge(
    dense_hits: list[ProtocolHit],
    bm25_hits: list[ProtocolHit],
    limit: int,
) -> list[ProtocolHit]:
    scores: dict[tuple[str, str], float] = {}
    hit_by_key: dict[tuple[str, str], ProtocolHit] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        key = (hit.doc_id, hit.chunk_id)
        hit_by_key[key] = hit
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    for rank, hit in enumerate(bm25_hits, start=1):
        key = (hit.doc_id, hit.chunk_id)
        hit_by_key.setdefault(key, hit)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [
        hit_by_key[key].model_copy(update={"score": float(scores[key])})
        for key in ranked_keys[:limit]
    ]


def _cross_encoder_rerank(query: str, hits: list[ProtocolHit], limit: int) -> list[ProtocolHit]:
    model_id = _reranker_model()
    if not model_id or len(hits) <= 1:
        return hits[:limit]

    pairs = [[query, f"{hit.title}\n{_clean_doc_text(hit.text)}"] for hit in hits]
    scores = _reranker(model_id).predict(pairs)
    reranked = sorted(
        zip(hits, scores, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        hit.model_copy(update={"score": float(score)})
        for hit, score in reranked[:limit]
    ]


def retrieve_protocol(query: str, k: int = 5) -> list[ProtocolHit]:
    """Return top protocol chunks using dense search, optional BM25, and reranking."""
    if not query.strip():
        return []

    use_hybrid = _hybrid_enabled()
    first_stage_k = max(k, _rerank_candidates()) if (use_hybrid or _reranker_model()) else k
    dense_hits = _dense_retrieve_protocol(query, first_stage_k)
    if use_hybrid:
        bm25_hits = _bm25_retrieve_protocol(query, first_stage_k)
        hits = _rrf_merge(dense_hits, bm25_hits, first_stage_k)
    else:
        hits = dense_hits

    return _cross_encoder_rerank(query, hits, k)


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
