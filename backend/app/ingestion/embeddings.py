"""Embedding providers for the ingestion pipeline and (later) retrieval.

Kept behind a small Protocol so swapping providers never touches the
pipeline or the vector store — the same "provider abstraction"
principle as the LLM Model Router (app/llm/router.py), applied here to
embeddings instead of chat completions.

IMPORTANT — HashingEmbeddingProvider is NOT semantically meaningful.
It's a fast, dependency-free, fully offline embedding used to make the
ingestion pipeline and retrieval plumbing genuinely testable in any
environment, including one with no network access to an embedding
API/model (which is the situation this pipeline was actually built
under — see docs/PROGRESS_TRACKER.md Phase 2 notes). It lets chunking,
storage, dedup, and query logic all be verified end-to-end. It MUST be
swapped for a real embedding model (OpenAIEmbeddingProvider below,
GeminiEmbeddingProvider, or a local sentence-transformers model)
before retrieval quality means anything — cosine similarity over
hashing-embedding vectors does not reliably find semantically related
legal text.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import threading
import time
from typing import Optional, Protocol, Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    dimension: int

    def embed(
        self, texts: list[str], *, task_type: Optional[str] = None
    ) -> list[list[float]]: ...


# Whether a provider produces semantically meaningful vectors. `False`
# for HashingEmbeddingProvider (explicitly NOT semantic — see its
# docstring). `HybridRetriever.retrieve` reads this to decide which
# candidate-generation strategy is safe: the full-category BM25 path
# (works with garbage vectors) vs the vector-pre-filtered candidate
# pool (only valid when the vector signal is real — see
# app/rag/retrieval.py for the two paths). Defaults to False via
# `getattr` at the call site so the Protocol doesn't have to grow.
IS_SEMANTIC_ATTR = "is_semantic"


class HashingEmbeddingProvider:
    """Deterministic, offline, dependency-free bag-of-words hashing
    embedding. See module docstring — NOT a substitute for a real
    embedding model."""

    is_semantic = False
    dimension = 256

    def embed(
        self, texts: list[str], *, task_type: Optional[str] = None
    ) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float64)
        for token in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class OpenAIEmbeddingProvider:
    """Real embedding provider using OpenAI's embeddings API. Requires
    OPENAI_API_KEY and network access — this is what a real deployment
    should configure. Verify the model name against current OpenAI
    docs (platform.openai.com/docs/guides/embeddings) before relying on
    the default below."""

    is_semantic = True
    dimension = 1536  # matches text-embedding-3-small as of this writing

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — cannot use OpenAIEmbeddingProvider")
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def embed(
        self, texts: list[str], *, task_type: Optional[str] = None
    ) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


def _gemini_embedding_keys_from_env() -> list[str]:
    """All usable keys from GOOGLE_API_KEYS / GOOGLE_API_KEY, honoring
    the same comma-separated multi-key format the LLM Gemini adapter
    accepts. Unlike the LLM adapter (which rotates to retry a request),
    the embedding provider rotates keys UP FRONT per batch so the 100
    req/min per-key free-tier quota is spread across every configured
    key instead of hammering one — see GeminiEmbeddingProvider."""
    for var in ("GOOGLE_API_KEYS", "GOOGLE_API_KEY"):
        raw = os.getenv(var)
        if raw:
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            if keys:
                return keys
    return []


class GeminiEmbeddingProvider:
    """Real embedding provider using Google's Gemini embeddings API.
    Default model `gemini-embedding-001` (confirmed available for
    embedContent on the installed google-genai version 2.18.1 — the
    newer `gemini-embedding-2` also exists; the provider defaults to
    the stable non-preview model). Output dimensionality pinned to 768
    to keep vectors small and make the dimension explicit in one place.

    Uses the same GOOGLE_API_KEY the LLM Model Router already runs on,
    so a deployment that has Gemini for chat gets semantic embeddings
    with zero new credentials — this is the provider this deployment
    actually uses (no OPENAI_API_KEY configured; see
    docs/DEBUGGING_LOOP_LOG.md Iteration 9 for the switch + re-seed).
    Vietnamese quality on this model is solid, which matters here since
    the corpus is Vietnamese legal text.

    `task_type` is honored for the retrieval-oriented models that
    accept it (e.g. "RETRIEVAL_DOCUMENT" for corpus documents,
    "RETRIEVAL_QUERY" for search queries) and ignored otherwise —
    the ingestion pipeline and the test suite never pass it, the
    retrieval path (app/rag/retrieval.py) passes "RETRIEVAL_QUERY"
    for the query vector.

    KEY ROTATION: the free tier caps embedding at 100 req/min PER KEY.
    This provider therefore builds one client per configured key and
    round-robins across them per batch, so an N-key deployment gets
    ~N× the throughput before hitting a 429 (observed live during the
    Iteration 9 reseed — see docs/DEBUGGING_LOOP_LOG.md). On a
    retryable error it rotates to the next key and backs off (still
    honoring the server's retryDelay), so a single exhausted key can't
    stall a run. With one key it degrades to plain backoff."""
    is_semantic = True
    dimension = 768  # gemini-embedding-001 supports output_dimensionality 1-3072

    def __init__(
        self,
        api_key: Optional[Union[str, list[str]]] = None,
        model: Optional[str] = None,
    ):
        if api_key is None:
            keys = _gemini_embedding_keys_from_env()
        elif isinstance(api_key, str):
            keys = [k.strip() for k in api_key.split(",") if k.strip()]
        else:
            keys = [k for k in api_key if k]
        if not keys:
            raise RuntimeError("GOOGLE_API_KEY not set — cannot use GeminiEmbeddingProvider")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover — same import as the LLM adapter
            raise RuntimeError(f"'google-genai' package not installed: {exc}") from exc
        self._clients = [genai.Client(api_key=k) for k in keys]
        self._client = self._clients[0]
        self._idx = 0
        # Rotation state (_idx/_client) is mutable across the per-batch
        # request/retry section — a lock makes a shared instance safe for
        # concurrent callers (e.g. parallel checklist category workers).
        self._lock = threading.Lock()
        self._model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    def _advance(self) -> None:
        """Round-robin to the next key (no-op with a single key)."""
        if len(self._clients) > 1:
            self._idx = (self._idx + 1) % len(self._clients)
            self._client = self._clients[self._idx]

    def embed(
        self, texts: list[str], *, task_type: Optional[str] = None
    ) -> list[list[float]]:
        from google.genai import errors as genai_errors
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            config = types.EmbedContentConfig(output_dimensionality=self.dimension)
            if task_type:
                config.task_type = task_type
            # Proactive rotation: spread batches across keys so the
            # per-key free-tier quota is used in parallel, not serially
            # against one key until it 429s. Held under the lock so
            # concurrent threads never corrupt rotation state.
            with self._lock:
                self._advance()
                last_error: Optional[Exception] = None
                for attempt in range(_EMBED_MAX_ATTEMPTS):
                    try:
                        resp = self._client.models.embed_content(
                            model=self._model,
                            contents=batch,
                            config=config,
                        )
                        out.extend([e.values for e in resp.embeddings])
                        break
                    except genai_errors.ClientError as exc:
                        # The genai client's own tenacity retry stops at 429
                        # (RESOURCE_EXHAUSTED), which the free tier hits at
                        # 100 embed requests/minute — observed live during
                        # the Iteration 9 corpus reseed. Rotate to the next
                        # key and retry instead of failing.
                        if exc.code not in (429, 500, 502, 503, 504):
                            raise
                        last_error = exc
                        self._advance()
                        if len(self._clients) > 1 and attempt < len(self._clients) - 1:
                            # A fresh, untried key gets an immediate retry —
                            # the server's retryDelay applies to the exhausted
                            # key's quota, not the next key's.
                            time.sleep(0.25)
                        else:
                            # Every key exhausted (or single-key deploy):
                            # respect the server retryDelay / exponential
                            # backoff before the next attempt.
                            time.sleep(_retry_delay(exc, attempt))
                else:
                    assert last_error is not None
                    raise last_error
        return out


_EMBED_MAX_ATTEMPTS = 10


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Exponential backoff (cap 60s + jitter), preferring the
    server-provided `retryDelay` from the error's RetryInfo when the
    API includes one."""
    delay = min(2.0**attempt, 60.0)
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        error = details.get("error") or {}
        for detail in error.get("details", []) or []:
            raw = detail.get("retryDelay")
            if raw:
                try:
                    delay = max(delay, float(str(raw).rstrip("s")))
                except ValueError:
                    pass
    return delay + random.uniform(0.0, 1.0)


def get_default_embedding_provider() -> "EmbeddingProvider":
    """Real semantic provider if a key is configured — OpenAI preferred,
    then Gemini (no new credentials in deployments already using
    GOOGLE_API_KEY for the LLM Router) — otherwise falls back to the
    offline hashing provider with a loud warning, since that fallback
    is unsuitable for real retrieval quality: it exists to keep the
    pipeline runnable and testable, not to power a real product.

    `EMBEDDING_PROVIDER` (auto|openai|gemini|local|hashing) overrides
    the automatic ordering. `local` selects
    `LocalSentenceTransformerEmbeddingProvider` — a quota-free,
    in-process model that makes a full-corpus ingest of the ~158K-doc
    vbpl-vn dataset practical (no per-request cost or rate limit; see
    docs/DEBUGGING_LOOP_LOG.md Iteration 11). The default `auto`
    ordering deliberately does NOT auto-select the local model even
    when it's importable: constructing it downloads a model, which
    would make an offline/test environment silently fetch hundreds of
    MB on first use. Deployments that want it set
    `EMBEDDING_PROVIDER=local` explicitly."""
    explicit = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if explicit == "local":
        return LocalSentenceTransformerEmbeddingProvider()
    if explicit == "openai":
        return OpenAIEmbeddingProvider()
    if explicit == "gemini":
        return GeminiEmbeddingProvider()
    if explicit == "hashing":
        return HashingEmbeddingProvider()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddingProvider()
    if _gemini_embedding_keys_from_env():
        return GeminiEmbeddingProvider()
    logger.warning(
        "No OPENAI_API_KEY or GOOGLE_API_KEY set — falling back to "
        "HashingEmbeddingProvider, which is NOT semantically meaningful. "
        "Set OPENAI_API_KEY (or rely on Gemini's gemini-embedding-001 via "
        "GOOGLE_API_KEY) or set EMBEDDING_PROVIDER=local for the offline "
        "sentence-transformers model before trusting retrieval quality "
        "beyond pipeline testing."
    )
    return HashingEmbeddingProvider()


class LocalSentenceTransformerEmbeddingProvider:
    """Quota-free embedding provider running a sentence-transformers
    model in-process. There is no API key, no per-request cost, and no
    rate limit — the only constraint is the container's own CPU/RAM —
    which is what makes ingesting the ENTIRE ~158K-doc `tmquan/vbpl-vn`
    dataset (≈500K+ chunks) practical (see docs/DEBUGGING_LOOP_LOG.md
    Iteration 11). The model is downloaded from Hugging Face once on
    first construction and cached under
    `/root/.cache/huggingface` (persisted via the `hf-cache` volume in
    docker-compose.yml), then runs fully offline.

    Default model `intfloat/multilingual-e5-small` (384-dim) is chosen
    over larger multilingual models (e.g. `BAAI/bge-m3`, 1024-dim) for
    this corpus specifically: it is fast enough on CPU to embed the
    full corpus in tens of minutes rather than hours, small enough to
    fit alongside the rest of the backend in a modest container, and
    still ranks solidly for Vietnamese. Override via
    `LOCAL_EMBEDDING_MODEL` (any SentenceTransformer model works, but
    `dimension` is read from the model at init — a different model
    changes the stored vector dimension, so the Chroma collection must
    be re-seeded to match).

    E5-family models expect a task prefix on every input
    ("query: ..." / "passage: ...") — retrieval quality drops
    noticeably without it. The provider detects e5 models by name and
    applies the prefix from `task_type`: "RETRIEVAL_QUERY" → "query:",
    everything else (document ingestion) → "passage:". Non-e5 models
    get no prefix. This mirrors how the Gemini/OpenAI providers honor
    `task_type` for the retrieval path (app/rag/retrieval.py passes
    "RETRIEVAL_QUERY")."""

    is_semantic = True

    def __init__(self, model: Optional[object] = None, model_name: Optional[str] = None):
        self._model_name = model_name or os.getenv(
            "LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "the 'sentence-transformers' package is required for "
                    "LocalSentenceTransformerEmbeddingProvider (pip install "
                    "sentence-transformers torch)"
                ) from exc
            model = SentenceTransformer(self._model_name)
        self._model = model
        self._use_e5_prefixes = "e5" in self._model_name.lower()
        # sentence-transformers renamed this in 5.x (deprecation warning
        # observed on 5.7.0); fall back for older versions.
        get_dim = getattr(model, "get_embedding_dimension", None)
        self.dimension = (
            get_dim() if get_dim else model.get_sentence_embedding_dimension()
        )
        # model.encode is safe to call concurrently (read-only weights),
        # but a lock keeps lifecycle simple and matches the Gemini
        # provider's threading contract for shared instances.
        self._lock = threading.Lock()

    def embed(
        self, texts: list[str], *, task_type: Optional[str] = None
    ) -> list[list[float]]:
        prompts = self._make_prompts(texts, task_type)
        with self._lock:
            vecs = self._model.encode(
                prompts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vecs]

    def _make_prompts(self, texts: list[str], task_type: Optional[str]) -> list[str]:
        if not self._use_e5_prefixes:
            return texts
        prefix = "query: " if task_type == "RETRIEVAL_QUERY" else "passage: "
        return [prefix + t for t in texts]
