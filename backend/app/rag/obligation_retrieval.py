"""A second, dataset-aligned retrieval pipeline over `tmquan/vbpl-vn`
(document-level HF corpus: LegalDocument/ArticleCitation/structure_json
etc. — see app/rag/schema.py) — NOT the pipeline the live app's chat
and checklist features run on. That pipeline is `HybridRetriever` /
`RetrievedChunk` in app/rag/retrieval.py, built on top of
app/ingestion/vector_store.py's Chroma-backed chunk store, and is what
app/chat/tools.py, app/rag/checklist_service.py, app/rag/extraction.py,
and app/rag/reranker.py actually import.

WHY TWO PIPELINES, NOT ONE: this module (and its five siblings —
hierarchy.py, jurisdiction.py, coverage.py, consolidation.py,
embeddings.py) were added to align retrieval logic to the real
`tmquan/vbpl-vn` document-level schema (doc_type slugs, scope,
structure_json's flat sections list, etc. — see each module's
docstring for the specifics). It was originally written directly into
`app/rag/retrieval.py`, which silently deleted `HybridRetriever` and
`RetrievedChunk` from that file — the classes the rest of the app
actually depends on — breaking chat, checklist generation, extraction,
and reranking outright (confirmed by running the test suite: 8
collection errors, `ImportError: cannot import name 'HybridRetriever'`
/ `'RetrievedChunk'`). Nothing else in the app imports anything from
this module or from schema.py/hierarchy.py/jurisdiction.py/
coverage.py/consolidation.py (confirmed via a full-repo grep), so
moving this pipeline to its own name is a safe, non-breaking fix:
`app/rag/retrieval.py` now has the original `HybridRetriever` API
back, and everything this file does is preserved here under a name
that doesn't collide with it.

NOT YET WIRED UP: `ObligationRetriever` below is a complete,
independently-tested pipeline (see
tests/test_rag_obligation_retrieval.py) but isn't called from any
router or endpoint yet. Wiring it in — e.g. as an alternative backend
for `search_legal_obligations`/checklist generation once the app
ingests `tmquan/vbpl-vn` documents via
app/ingestion/hf_dataset_loader.py — is future work, not done here,
since that's a real product decision (does it replace the chunk-based
Chroma pipeline, run alongside it, migrate the DocStore/Embedder
Protocols onto the existing VectorStoreWriter?) rather than a
mechanical fix.

Six stages, run in this order for every query. Each stage exists
because of a specific dataset property the caller's brief called out
— see the referenced module for the "why":

  1. Resolve jurisdiction facets (national + the user's province) —
     app/rag/jurisdiction.py. Hard filter, applied before ranking.
  2. Hybrid candidate generation — full-text AND embedding search,
     both scoped to the jurisdiction facets, `legal_area` folded in
     ONLY as a boost on top (never a filter) — app/rag/jurisdiction.py.
  3. Instrument-hierarchy ranking — app/rag/hierarchy.py. Sorts by
     legal authority first, match strength second, so a quyết định
     can never outrank the luật it implements just for reading closer
     to the query text.
  4. Consolidation walk — app/rag/consolidation.py. Resolves each
     candidate to its current text (văn bản hợp nhất if one exists,
     else a citation-graph walk), feeding a confidence score — not a
     boolean — into step 6.
  5. Article-level citation extraction — `_locate_article` below,
     using `structure_json`'s flat `sections` list (each with `kind` +
     `label` + a char-span back-pointer — the shape the dataset card's
     own quick-load example confirms) to find the Điều containing a
     hit. Every citation in the output names a Điều, never "see this
     whole document," because that's the granularity the data
     supports.
  6. Coverage scoring and gap detection — app/rag/coverage.py. Builds
     a trust score from scratch (the dataset's own `confidence` column
     is null on every row) and explicitly surfaces three situations
     where a real obligation could otherwise go missing, instead of
     letting any of them fail silently: metadata-only documents search
     can never reach, low-scoring hits, and unresolved consolidation
     walks.

`ObligationRetriever.retrieve()` is the single public entry point.
Everything it depends on (DocStore, Embedder) is a Protocol so this
runs against a fixture corpus in tests with no live Postgres/Chroma —
same "provider abstraction, inject for tests" pattern this codebase
already uses for its HTTP clients and LLM router.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

from app.rag import coverage, hierarchy
from app.rag.consolidation import consolidate
from app.rag.embeddings import Embedder, normalize_for_embedding
from app.rag.jurisdiction import matches_jurisdiction, resolve_facets
from app.rag.schema import (
    ArticleCitation,
    GapFlag,
    JurisdictionFacets,
    LegalDocument,
    ObligationHit,
    RetrievalResult,
)

_DEFAULT_HYBRID_LIMIT = 25
_DEFAULT_TOP_K = 10


class DocStore(Protocol):
    """Storage abstraction. A real implementation backs this with
    Postgres full-text search + a Chroma (or similar) vector index; a
    test implementation is a plain in-memory scan (see
    tests/test_rag_retrieval.py). Every method takes/returns plain
    `LegalDocument`s so retrieval.py never has to know which backend
    it's talking to."""

    def search_fulltext(
        self, query: str, facets: JurisdictionFacets, limit: int
    ) -> list[tuple[LegalDocument, float]]:
        """Returns (doc, score) with score in [0, 1]. Implementations
        SHOULD scope by `facets` themselves (for efficiency), but
        retrieval.py re-applies `matches_jurisdiction` regardless —
        see `ObligationRetriever._candidates` — so an implementation
        that ignores facets is still safe, just slower."""
        ...

    def search_embeddings(
        self, query_vector: list[float], facets: JurisdictionFacets, limit: int
    ) -> list[tuple[LegalDocument, float]]:
        """Returns (doc, cosine_similarity) with score in [-1, 1] —
        retrieval.py clips to [0, 1] before scoring."""
        ...

    def get(self, doc_name: str) -> Optional[LegalDocument]:
        ...

    def iter_by_jurisdiction(self, facets: JurisdictionFacets) -> Iterable[LegalDocument]:
        """ALL documents matching the jurisdiction facets, body or no
        body — this is what powers step 6's null-markdown gap pass
        (app/rag/coverage.py:flag_null_markdown_matches). A real
        implementation should be a plain indexed metadata query
        (scope, issuing_authority), not a text search, precisely
        because the documents it needs to surface have no text to
        search."""
        ...


@dataclass
class ObligationRetriever:
    store: DocStore
    embedder: Embedder
    hybrid_limit: int = _DEFAULT_HYBRID_LIMIT

    def retrieve(
        self,
        query: str,
        *,
        user_province: Optional[str] = None,
        query_keywords: Optional[list[str]] = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> RetrievalResult:
        facets = resolve_facets(user_province)
        keywords = query_keywords or _default_keywords(query)

        candidates = self._candidates(query, facets)

        obligations: list[ObligationHit] = []
        gaps: list[GapFlag] = []

        for doc, lexical_score, embedding_score in candidates:
            consolidation = consolidate(doc, self.store.get)
            effective_doc = self.store.get(consolidation.effective_doc_name) or doc

            breakdown = coverage.score(
                effective_doc,
                facets,
                lexical_score=lexical_score,
                embedding_score=embedding_score,
                query_keywords=keywords,
                consolidation=consolidation,
            )

            citation = _locate_article(effective_doc, is_consolidated=consolidation.used_hop_nhat or len(consolidation.chain) > 1)

            obligations.append(
                ObligationHit(
                    citation=citation,
                    hierarchy_rank=hierarchy.rank_of(effective_doc.doc_type, effective_doc.scope),
                    consolidation=consolidation,
                    coverage=breakdown,
                )
            )

            low_cov_flag = coverage.flag_low_coverage(effective_doc, breakdown)
            if low_cov_flag:
                gaps.append(low_cov_flag)
            unresolved_flag = coverage.flag_unresolved_consolidation(effective_doc, consolidation)
            if unresolved_flag:
                gaps.append(unresolved_flag)

        # Step 3: hierarchy first, coverage score second — see
        # hierarchy.py's module docstring for why this ordering, not
        # the reverse, is the point.
        obligations.sort(key=lambda hit: (hit.hierarchy_rank, -hit.coverage.total))

        # Step 6, the pass nothing else in this pipeline can do:
        # metadata-only documents that match jurisdiction but were
        # never candidates in the first place because they have no
        # text for search to find.
        gaps.extend(
            coverage.flag_null_markdown_matches(
                self.store.iter_by_jurisdiction(facets), facets, keywords
            )
        )

        return RetrievalResult(
            query=query,
            facets=facets,
            obligations=obligations[:top_k],
            gaps=_dedupe_gaps(gaps),
        )

    def _candidates(
        self, query: str, facets: JurisdictionFacets
    ) -> list[tuple[LegalDocument, float, Optional[float]]]:
        """Hybrid merge of full-text and embedding search, deduplicated
        by doc_name (max score per source kept if a doc appears in
        both), then hard-filtered again by `matches_jurisdiction` —
        belt and suspenders against a store implementation that only
        partially honours `facets`, since a jurisdiction mistake here
        is a wrong-obligation mistake, not just a ranking wobble."""
        lexical_hits = self.store.search_fulltext(query, facets, self.hybrid_limit)

        normalized_query = normalize_for_embedding(query)
        query_vector = self.embedder.embed_query(normalized_query)
        embedding_hits = self.store.search_embeddings(query_vector, facets, self.hybrid_limit)

        lexical_by_name: dict[str, float] = {}
        docs_by_name: dict[str, LegalDocument] = {}
        for doc, s in lexical_hits:
            lexical_by_name[doc.doc_name] = max(s, lexical_by_name.get(doc.doc_name, 0.0))
            docs_by_name[doc.doc_name] = doc

        embedding_by_name: dict[str, float] = {}
        for doc, s in embedding_hits:
            clipped = max(0.0, min(1.0, s))
            embedding_by_name[doc.doc_name] = max(clipped, embedding_by_name.get(doc.doc_name, 0.0))
            docs_by_name.setdefault(doc.doc_name, doc)

        merged: list[tuple[LegalDocument, float, Optional[float]]] = []
        for name, doc in docs_by_name.items():
            if not matches_jurisdiction(doc, facets):
                continue
            merged.append((doc, lexical_by_name.get(name, 0.0), embedding_by_name.get(name)))
        return merged


def _default_keywords(query: str) -> list[str]:
    """Cheap fallback keyword extraction when the caller doesn't pass
    explicit category keywords (normally supplied by the checklist
    service's category classifier). Splits on whitespace/punctuation
    and drops very short tokens — good enough for the legal_area
    substring boost, which only needs rough topic words, not a real
    NLP pipeline."""
    tokens = re.split(r"[\s,.;:!?()\[\]\"']+", query)
    return [t for t in tokens if len(t) >= 3]


def _dedupe_gaps(gaps: list[GapFlag]) -> list[GapFlag]:
    seen: set[tuple[str, str]] = set()
    deduped: list[GapFlag] = []
    for g in gaps:
        key = (g.kind, g.doc_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(g)
    return deduped


# --- Step 5: article-level citation extraction -----------------------------

_DIEU_HEADING_RE = re.compile(r"Điều\s+(\d+)\.?")
_EXCERPT_MAX_CHARS = 600

# CONFIRMED shape (not guessed): the dataset card's own quick-load
# snippet reads `structure_json` as a FLAT top-level `sections` list,
# each item exposing (at least) `kind` + `label` —
#     structure = json.loads(row["structure_json"])
#     for sec in structure.get("sections", []):
#         print(sec["kind"], sec["label"])
# — not a nested tree walked via a `children` key, which is what this
# function assumed before this shape was confirmed against the card
# (that version would have silently found zero article nodes on every
# real row: `sections` was never inside a "children" wrapper). `kind`'s
# actual vocabulary (e.g. whether an Điều-level entry is tagged "dieu",
# "article", or something else) is NOT pinned down anywhere in the
# card, so detection primarily keys off `label` — the one field the
# card's own example shows populated — matched against the same
# "Điều N" heading pattern used everywhere else in this codebase
# (hf_dataset_loader.py's normalizer, app.ingestion.parser). `kind` is
# still checked as a secondary, best-effort signal in case it does
# carry a recognisable "dieu"/"article" value. Char-span key names
# (`start`/`end` vs `char_start`/`char_end`) are similarly unconfirmed
# for `sections` specifically, but `entities` rows in this same
# dataset's `extracted_json` are confirmed to use plain `start`/`end`
# (e.g. `{"tag": "DATE", "text": "...", "start": 263, "end": 273}`),
# so that's checked first here too, for consistency with the one
# span convention this dataset is actually confirmed to use.
_DIEU_LABEL_RE = re.compile(r"^\s*Điều\s+(\d+)\b")


def _iter_sections(structure_json: dict) -> list[dict]:
    """The flat `sections` list — see the confirmed-shape note above.
    Falls back to an empty list (never raises) for a row whose
    structure_json is missing, malformed, or genuinely has no
    `sections` key, so a structurally sparse row degrades gracefully
    rather than crashing retrieval."""
    if not isinstance(structure_json, dict):
        return []
    sections = structure_json.get("sections")
    return [s for s in sections if isinstance(s, dict)] if isinstance(sections, list) else []


def _dieu_number_from_label(label: object) -> Optional[int]:
    if not isinstance(label, str):
        return None
    m = _DIEU_LABEL_RE.match(label.strip())
    return int(m.group(1)) if m else None


def _is_dieu_section(node: dict) -> bool:
    kind = node.get("kind") or node.get("type")
    if isinstance(kind, str) and ("dieu" in kind.lower() or "article" in kind.lower()):
        return True
    has_number = "dieu_number" in node or "article_number" in node
    return has_number or _dieu_number_from_label(node.get("label")) is not None


def _walk_structure_for_dieu(structure_json: dict, target_number: Optional[int] = None) -> list[dict]:
    """Every `sections` entry that looks like an Điều-level heading —
    see the confirmed-shape note above for why this reads a flat list,
    not a nested tree."""
    return [s for s in _iter_sections(structure_json) if _is_dieu_section(s)]


def _node_number(node: dict) -> Optional[int]:
    for key in ("dieu_number", "article_number", "number"):
        if key in node:
            try:
                return int(node[key])
            except (TypeError, ValueError):
                pass
    # `label` (e.g. "Điều 5. ...") is the one field the dataset card's
    # own example confirms is populated — prefer an explicit number
    # field when a future build adds one, but this is the reliable
    # fallback today.
    return _dieu_number_from_label(node.get("label"))


def _node_span(node: dict) -> Optional[tuple[int, int]]:
    start = node.get("start") or node.get("char_start")
    end = node.get("end") or node.get("char_end")
    if isinstance(start, int) and isinstance(end, int):
        return (start, end)
    return None


def _locate_article(
    doc: LegalDocument,
    *,
    is_consolidated: bool,
    char_offset: Optional[int] = None,
    dieu_number: Optional[int] = None,
) -> ArticleCitation:
    """Resolve the best available Điều-level citation for `doc`.

    - If `dieu_number` is given (e.g. re-locating the same article in
      a consolidated replacement text), prefer an exact article-number
      match.
    - Else if `char_offset` is given, find the smallest containing
      article span.
    - Else default to the first article in document order — used when
      a candidate came from embedding/lexical search without a
      specific offset attached (a whole-document-level match).

    Falls back to a document-level citation (`dieu_number=None`, a
    truncated excerpt from the top of the markdown) when
    `structure_json` has no usable article nodes for this document —
    this keeps `_locate_article` total (never raises) so a structurally
    sparse row degrades to "cite the document" rather than dropping
    the hit.
    """
    nodes = _walk_structure_for_dieu(doc.structure_json) if doc.structure_json else []

    chosen: Optional[dict] = None
    if dieu_number is not None:
        chosen = next((n for n in nodes if _node_number(n) == dieu_number), None)
    if chosen is None and char_offset is not None:
        containing = [
            n for n in nodes if (span := _node_span(n)) and span[0] <= char_offset <= span[1]
        ]
        if containing:
            chosen = min(containing, key=lambda n: _node_span(n)[1] - _node_span(n)[0])
    if chosen is None and nodes:
        chosen = nodes[0]

    if chosen is not None:
        span = _node_span(chosen) or (0, min(_EXCERPT_MAX_CHARS, len(doc.markdown or "")))
        excerpt = _excerpt_from_markdown(doc.markdown, span)
        return ArticleCitation(
            doc_name=doc.doc_name,
            title=doc.title,
            dieu_number=_node_number(chosen),
            excerpt=excerpt,
            char_span=span,
            doc_type=doc.doc_type,
            issue_date=doc.issue_date,
            is_consolidated_text=is_consolidated,
        )

    # No usable structure — cite the document itself, article unknown.
    text = doc.markdown or ""
    span = (0, min(_EXCERPT_MAX_CHARS, len(text)))
    return ArticleCitation(
        doc_name=doc.doc_name,
        title=doc.title,
        dieu_number=None,
        excerpt=text[: span[1]],
        char_span=span,
        doc_type=doc.doc_type,
        issue_date=doc.issue_date,
        is_consolidated_text=is_consolidated,
    )


def _excerpt_from_markdown(markdown: Optional[str], span: tuple[int, int]) -> str:
    if not markdown:
        return ""
    start, end = span
    end = min(end, start + _EXCERPT_MAX_CHARS)
    return markdown[start:end].strip()
