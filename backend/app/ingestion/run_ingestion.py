"""CLI entrypoint for running the ingestion pipeline.

Two modes:

    python -m app.ingestion.run_ingestion
        Ingests ONLY the synthetic test fixture in sample_data/ — safe
        to run anywhere, proves the parse -> chunk -> tag -> embed ->
        write pipeline works end to end without needing network access
        or a real legal document.

    python -m app.ingestion.run_ingestion --crawl [--max-documents N]
        Crawls real documents from the live VBPL gateway
        (app.ingestion.crawler.VbplGatewayCrawler). CONFIRMED BROKEN as
        of this writing: the gateway is gated behind a reCAPTCHA-derived
        Bearer token a plain HTTP client can't obtain, so this fails
        with 400 Bad Request on the first request — see
        app/ingestion/crawler.py's module docstring for the full root
        cause. Left in place for when this app adds real Bearer-token
        support.

    python -m app.ingestion.run_ingestion --hf-dataset [--max-documents N]
        Ingests real, already-crawled VBPL documents by streaming the
        tmquan/vbpl-vn dataset from Hugging Face (CC-BY-4.0) — see
        app/ingestion/hf_dataset_loader.py's module docstring. Works
        around the gateway auth wall above by using a corpus someone
        else already crawled with proper auth. Requires network access
        to huggingface.co and the `datasets` package.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.ingestion.crawler import VbplGatewayCrawler
from app.ingestion.hf_dataset_loader import HfVbplDatasetLoader
from app.ingestion.metadata import SourceMeta
from app.ingestion.pipeline import ingest_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
DEFAULT_CHECKPOINT_PATH = Path(__file__).parent / "sample_data" / ".vbpl_crawl_checkpoint.json"


def ingest_sample_fixture() -> dict:
    """Ingests the synthetic test fixture — NOT real law. Useful for
    smoke-testing the full pipeline end to end without a real data
    source, and for local development against a running Chroma
    instance (e.g. via docker-compose).

    Tagged category="tax" (a REAL category from app.rag.query_builder,
    not the placeholder "test_fixture" tag unit tests use) specifically
    so the checklist generator's manual-activation button
    (POST /checklist/{user_id}/generate) has something to actually find
    and produce a visible result from on a fresh install — running this
    CLI once is what makes the "showcase" demo flow work end to end
    without needing a real crawl first. This was found to matter by
    actually running the app rather than only its test suite: tagging
    this "test_fixture" (as an earlier version of this function did)
    means EVERY real category query in checklist_service.py returns
    zero hits, so the manual-generate button silently produces an empty
    checklist with no error and no obvious explanation — exactly the
    wrong first impression for a feature whose whole point is a live
    demo. See tests/test_ingestion_pipeline.py for where the
    "test_fixture" tag is still used deliberately, for pipeline-only
    tests that don't go through the checklist generator at all.
    """
    text = (SAMPLE_DATA_DIR / "sample_test_law.txt").read_text(encoding="utf-8")
    source = SourceMeta(
        law_name="SAMPLE_TEST_LAW (synthetic fixture, NOT a real legal source — tagged 'tax' for demo purposes)",
        category="tax",
        entity_type="both",
    )
    return ingest_document(
        doc_id="sample-test-law",
        title="Sample Test Law (synthetic — demonstrates the pipeline, not real legal advice)",
        raw_text=text,
        source=source,
    )


def crawl_and_ingest_real_documents(max_documents: int | None) -> list[dict]:
    """Runs the real VBPL gateway crawler and ingests whatever it
    successfully fetches. See module docstring — CONFIRMED BROKEN
    (400 Bad Request, reCAPTCHA/Bearer-token wall) as of this writing;
    use --hf-dataset instead until this app adds real token support."""
    import requests

    crawler = VbplGatewayCrawler(
        http_client=requests.Session(), checkpoint_path=DEFAULT_CHECKPOINT_PATH
    )
    results = []
    for doc in crawler.crawl(max_documents=max_documents):
        result = ingest_document(doc.doc_id, doc.title, doc.raw_text, doc.source)
        logger.info("Ingested real document '%s': %s", doc.title, result)
        results.append(result)
    return results


def crawl_and_ingest_hf_dataset(max_documents: int | None) -> list[dict]:
    """Streams real, already-crawled VBPL documents from the
    tmquan/vbpl-vn Hugging Face dataset and ingests each one. See
    app/ingestion/hf_dataset_loader.py's module docstring for why this
    exists (works around the live gateway's auth wall)."""
    loader = HfVbplDatasetLoader()
    results = []
    for doc in loader.crawl(max_documents=max_documents):
        result = ingest_document(doc.doc_id, doc.title, doc.raw_text, doc.source)
        logger.info("Ingested real document '%s': %s", doc.title, result)
        results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--crawl",
        action="store_true",
        help="Crawl and ingest real documents from the live VBPL gateway. Currently broken — see module docstring.",
    )
    mode.add_argument(
        "--hf-dataset",
        action="store_true",
        help="Ingest real VBPL documents by streaming the tmquan/vbpl-vn dataset from Hugging Face.",
    )
    parser.add_argument(
        "--max-documents", type=int, default=10, help="Cap on how many real documents to ingest (with --crawl or --hf-dataset)."
    )
    args = parser.parse_args()

    if args.crawl:
        outcome = crawl_and_ingest_real_documents(args.max_documents)
        print(f"Crawled and ingested {len(outcome)} real document(s) from the live gateway.")
    elif args.hf_dataset:
        outcome = crawl_and_ingest_hf_dataset(args.max_documents)
        print(f"Ingested {len(outcome)} real document(s) from the tmquan/vbpl-vn dataset.")
    else:
        result = ingest_sample_fixture()
        print(result)

    # --hf-dataset streams via `datasets` (streaming=True) and stops
    # early once max_documents is hit — abandoning that iterator before
    # it's exhausted leaves the `datasets`/xet-bridge native download
    # backend's background threads running, which crashes at normal
    # interpreter shutdown with "Fatal Python error: PyGILState_Release"
    # (observed in practice; harmless in effect since it happens after
    # this point, i.e. after every write above already completed, but
    # alarming to see and returns a non-zero exit code). All real work
    # is done by the time we get here, so skip further Python
    # finalization entirely rather than let those threads race it.
    import os

    os._exit(0)
