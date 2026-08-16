"""Basic request tracing (Phase 9 hardening, docs/IMPLEMENTATION_PLAN.md
Phase 9: "structured logging + basic tracing"). A lightweight
request-ID middleware — not a full distributed-tracing setup
(OpenTelemetry/Jaeger etc., which this sandbox has no way to stand up
or verify anyway), but a real, testable mechanism: every request gets
a unique ID, logged on start and completion (with status/duration),
and echoed back in an `X-Request-ID` response header so a client-side
report ("my request failed") can be correlated with server logs.
"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("app.request")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
