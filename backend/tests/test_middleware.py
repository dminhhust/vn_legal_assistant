"""Integration tests for RequestIDMiddleware (middleware.py) — verifies
every response carries an X-Request-ID header, that a client-supplied
one is echoed back (for correlating a client-side report with server
logs), and that failures still get an ID logged before the error
propagates.
"""
from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import RequestIDMiddleware


@pytest.fixture()
def app():
    test_app = FastAPI()
    test_app.add_middleware(RequestIDMiddleware)

    @test_app.get("/ok")
    def ok():
        return {"status": "fine"}

    @test_app.get("/boom")
    def boom():
        raise ValueError("simulated failure")

    return test_app


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def test_response_includes_a_request_id_header(client):
    resp = client.get("/ok")
    assert "x-request-id" in {k.lower() for k in resp.headers.keys()}
    assert resp.headers["x-request-id"]  # non-empty


def test_different_requests_get_different_request_ids(client):
    resp1 = client.get("/ok")
    resp2 = client.get("/ok")
    assert resp1.headers["x-request-id"] != resp2.headers["x-request-id"]


def test_client_supplied_request_id_is_echoed_back(client):
    resp = client.get("/ok", headers={"X-Request-ID": "my-custom-id-123"})
    assert resp.headers["x-request-id"] == "my-custom-id-123"


def test_response_body_and_status_unaffected_by_middleware(client):
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert resp.json() == {"status": "fine"}


def test_failing_request_still_gets_a_request_id_logged(client, caplog):
    with caplog.at_level(logging.ERROR, logger="app.request"):
        client.get("/boom")

    assert any("request_failed" in record.message for record in caplog.records)
    assert any(getattr(record, "request_id", None) for record in caplog.records)
