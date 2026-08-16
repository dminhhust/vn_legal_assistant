"""Unit tests for embedding providers (embeddings.py).

Only the offline HashingEmbeddingProvider is tested against real
behavior — OpenAIEmbeddingProvider and GeminiEmbeddingProvider need
network + a real key and are exercised only for their provider-
selection wiring and construction errors, not a live call.
"""
from __future__ import annotations

import math

import pytest

from app.ingestion.embeddings import (
    GeminiEmbeddingProvider,
    HashingEmbeddingProvider,
    LocalSentenceTransformerEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_default_embedding_provider,
)


class _FakeSTModel:
    """Stand-in for a sentence-transformers model — records the prompts
    it was given and returns fixed-dimension vectors, so the local
    provider's prefix logic / dimension plumbing are testable without
    downloading a real model."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.last_texts: list[str] = []
        self.last_kwargs: dict = {}

    def get_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, **kwargs):
        import numpy as np

        self.last_texts = list(texts)
        self.last_kwargs = kwargs
        return np.full((len(texts), self._dim), 0.5, dtype=np.float64)


def test_same_text_produces_identical_vector():
    provider = HashingEmbeddingProvider()
    v1 = provider.embed(["hello world"])[0]
    v2 = provider.embed(["hello world"])[0]
    assert v1 == v2


def test_different_text_usually_produces_different_vectors():
    provider = HashingEmbeddingProvider()
    v1 = provider.embed(["hello world"])[0]
    v2 = provider.embed(["completely different sentence about tax law"])[0]
    assert v1 != v2


def test_vector_dimension_matches_declared_dimension():
    provider = HashingEmbeddingProvider()
    vec = provider.embed(["some text"])[0]
    assert len(vec) == provider.dimension


def test_vector_is_normalized():
    provider = HashingEmbeddingProvider()
    vec = provider.embed(["some text with several words in it"])[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_empty_text_does_not_crash_and_returns_zero_vector():
    provider = HashingEmbeddingProvider()
    vec = provider.embed([""])[0]
    assert len(vec) == provider.dimension
    assert all(x == 0.0 for x in vec)


def test_embed_handles_batch_of_multiple_texts():
    provider = HashingEmbeddingProvider()
    vecs = provider.embed(["first text", "second text", "third text"])
    assert len(vecs) == 3


def test_default_provider_falls_back_to_hashing_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    provider = get_default_embedding_provider()
    assert isinstance(provider, HashingEmbeddingProvider)


def test_default_provider_uses_gemini_when_google_key_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    provider = get_default_embedding_provider()
    assert isinstance(provider, GeminiEmbeddingProvider)


def test_gemini_provider_builds_all_comma_separated_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "key-one, key-two , key-three")
    provider = get_default_embedding_provider()
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.dimension == 768
    assert len(provider._clients) == 3


def test_gemini_provider_rotates_across_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "k1, k2, k3")
    provider = GeminiEmbeddingProvider()
    first = provider._client
    provider._advance()
    assert provider._client is not first
    provider._advance()
    provider._advance()
    assert provider._client is first  # wrapped around


def test_gemini_provider_single_key_advance_is_noop(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "only-key")
    provider = GeminiEmbeddingProvider()
    client = provider._client
    provider._advance()
    assert provider._client is client


def test_gemini_provider_uses_google_api_keys_plural_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEYS", "key-one")
    provider = get_default_embedding_provider()
    assert isinstance(provider, GeminiEmbeddingProvider)


def test_gemini_provider_raises_without_any_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        GeminiEmbeddingProvider()


def test_openai_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider()


class TestLocalSentenceTransformerProvider:
    def test_e5_query_prompt_is_prefixed(self):
        fake = _FakeSTModel()
        provider = LocalSentenceTransformerEmbeddingProvider(
            model=fake, model_name="intfloat/multilingual-e5-small"
        )
        provider.embed(["thuế thu nhập cá nhân"], task_type="RETRIEVAL_QUERY")
        assert fake.last_texts == ["query: thuế thu nhập cá nhân"]

    def test_e5_document_prompt_is_prefixed_as_passage(self):
        fake = _FakeSTModel()
        provider = LocalSentenceTransformerEmbeddingProvider(
            model=fake, model_name="intfloat/multilingual-e5-small"
        )
        provider.embed(["Điều 1. Nội dung điều luật."], task_type="RETRIEVAL_DOCUMENT")
        assert fake.last_texts == ["passage: Điều 1. Nội dung điều luật."]

    def test_e5_default_task_type_is_passage(self):
        fake = _FakeSTModel()
        provider = LocalSentenceTransformerEmbeddingProvider(
            model=fake, model_name="intfloat/multilingual-e5-small"
        )
        provider.embed(["nội dung luật"], task_type=None)
        assert fake.last_texts == ["passage: nội dung luật"]

    def test_non_e5_model_gets_no_prefix(self):
        fake = _FakeSTModel()
        provider = LocalSentenceTransformerEmbeddingProvider(
            model=fake, model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        provider.embed(["no prefix expected"], task_type="RETRIEVAL_QUERY")
        assert fake.last_texts == ["no prefix expected"]

    def test_dimension_read_from_model_and_is_semantic(self):
        fake = _FakeSTModel(dim=512)
        provider = LocalSentenceTransformerEmbeddingProvider(model=fake)
        assert provider.dimension == 512
        assert provider.is_semantic is True
        vecs = provider.embed(["a", "b"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 512


def test_local_provider_requires_sentence_transformers_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("module not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        LocalSentenceTransformerEmbeddingProvider()


class TestEmbeddingProviderEnvOverride:
    def test_env_selects_local(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

        class DummyLocal:
            is_semantic = True

        monkeypatch.setattr(
            "app.ingestion.embeddings.LocalSentenceTransformerEmbeddingProvider", DummyLocal
        )
        provider = get_default_embedding_provider()
        assert isinstance(provider, DummyLocal)

    def test_env_selects_openai_even_with_google_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        provider = get_default_embedding_provider()
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_env_selects_gemini(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        provider = get_default_embedding_provider()
        assert isinstance(provider, GeminiEmbeddingProvider)

    def test_env_selects_hashing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
        monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
        provider = get_default_embedding_provider()
        assert isinstance(provider, HashingEmbeddingProvider)

    def test_unknown_env_value_falls_back_to_auto(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
        monkeypatch.setenv("EMBEDDING_PROVIDER", "something-bogus")
        provider = get_default_embedding_provider()
        assert isinstance(provider, HashingEmbeddingProvider)

    def test_semantic_flag_is_true_only_for_real_providers(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        assert HashingEmbeddingProvider().is_semantic is False
        assert OpenAIEmbeddingProvider(api_key="fake-key").is_semantic is True
        assert GeminiEmbeddingProvider(api_key="fake-key").is_semantic is True
