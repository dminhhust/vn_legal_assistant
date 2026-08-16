"""FastAPI entrypoint.

Four routers, one per MVP feature:
  - profile_router    -> personal info collection
  - checklist_router   -> auto-generated checklist + manual activation
  - chat_router          -> RAG chatbot
  (the legal-source crawler is a CLI/pipeline component, not an HTTP
  feature — see app/ingestion/crawler.py + run_ingestion.py)

Before first run, create the DB schema with:
    python -m app.db.migrate
(docker-compose's backend service does this automatically — see
docker-compose.yml.)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.chat.router import router as chat_router
from app.llm.router import NoProviderAvailableError, get_router
from app.llm.schemas import Message
from app.logging_config import configure_logging
from app.middleware import RequestIDMiddleware
from app.profile.router import router as profile_router
from app.rag.router import router as checklist_router

configure_logging(level=logging.INFO)

app = FastAPI(title="VN Legal Assistant — MVP API", version="0.1.0")
app.add_middleware(RequestIDMiddleware)
app.include_router(profile_router)
app.include_router(checklist_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    router = get_router()
    return {
        "status": "ok",
        "llm_providers_available": router.available_providers(),
    }


@app.get("/llm/ping")
def llm_ping() -> dict:
    """Manual smoke-test for the Model Router — not a product feature
    endpoint. Confirms the router can reach whichever provider(s) have
    keys configured, and that the interface is wired correctly end to
    end."""
    router = get_router()
    try:
        response = router.complete(
            [Message(role="user", content="Reply with exactly the word: pong")],
            task="default",
            max_tokens=10,
        )
    except NoProviderAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "provider_used": response.provider,
        "model_used": response.model,
        "text": response.text,
    }
