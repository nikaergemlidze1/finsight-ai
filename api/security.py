"""Optional API-key protection for cost-bearing endpoints.

Behaviour:
    - If the API_KEY environment variable is NOT set (default), the check is
      a no-op — the API stays fully open for local development and demos.
    - If API_KEY is set, protected endpoints require a matching X-API-Key
      request header and return 401 otherwise.

The Streamlit frontend sends the header automatically when an API_KEY entry
exists in its secrets.
"""
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

API_KEY_HEADER = "X-API-Key"


async def require_api_key(request: Request) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        return  # auth disabled

    provided = request.headers.get(API_KEY_HEADER, "")
    # Constant-time comparison to avoid timing side-channels
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or missing {API_KEY_HEADER} header.",
        )
