"""Artifact-free unit tests for API building blocks.

These always run (locally and in CI) — no trained model required.
Covers: rate limiter, lead-tier classification, optional API-key auth.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.rate_limit import RateLimiter
from api.security import require_api_key
from api.tiers import classify

# ── RateLimiter ───────────────────────────────────────────────────────────────

def test_limiter_allows_within_limit():
    clock = [0.0]
    rl = RateLimiter(limit=3, window_seconds=60, time_func=lambda: clock[0])
    assert rl.check("client") is None
    assert rl.check("client") is None
    assert rl.check("client") is None


def test_limiter_blocks_over_limit_and_reports_wait():
    clock = [0.0]
    rl = RateLimiter(limit=2, window_seconds=60, time_func=lambda: clock[0])
    assert rl.check("client") is None
    assert rl.check("client") is None
    wait = rl.check("client")
    assert wait is not None and wait == pytest.approx(60.0)


def test_limiter_window_slides():
    clock = [0.0]
    rl = RateLimiter(limit=1, window_seconds=10, time_func=lambda: clock[0])
    assert rl.check("client") is None
    assert rl.check("client") is not None
    clock[0] = 10.1  # old hit falls out of the window
    assert rl.check("client") is None


def test_limiter_keys_are_isolated():
    rl = RateLimiter(limit=1, window_seconds=60, time_func=lambda: 0.0)
    assert rl.check("alice") is None
    assert rl.check("bob") is None
    assert rl.check("alice") is not None


# ── Lead tiers ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("prob,expected", [
    (0.95, "high"),
    (0.60, "high"),     # inclusive lower bound
    (0.59, "medium"),
    (0.30, "medium"),   # inclusive lower bound
    (0.29, "low"),
    (0.0, "low"),
])
def test_classify_tiers(prob, expected):
    assert classify(prob) == expected


# ── API-key auth ──────────────────────────────────────────────────────────────

@pytest.fixture()
def protected_client():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    def protected():
        return {"ok": True}

    return TestClient(app)


def test_auth_disabled_without_env(monkeypatch, protected_client):
    monkeypatch.delenv("API_KEY", raising=False)
    assert protected_client.get("/protected").status_code == 200


def test_auth_rejects_missing_or_wrong_key(monkeypatch, protected_client):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert protected_client.get("/protected").status_code == 401
    resp = protected_client.get("/protected", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_auth_accepts_correct_key(monkeypatch, protected_client):
    monkeypatch.setenv("API_KEY", "s3cret")
    resp = protected_client.get("/protected", headers={"X-API-Key": "s3cret"})
    assert resp.status_code == 200
