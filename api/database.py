"""Async MongoDB (Motor) client and logging helpers.

Design notes:
    - If neither MONGO_URL nor MONGODB_URI is set, every helper is a cheap
      no-op — no localhost fallback, no 3-second server-selection stalls.
    - Logging failures are logged and swallowed: persistence must never
      break a prediction response.
    - /analytics intentionally returns aggregates only. Raw research-query
      text is never exposed publicly (privacy: queries may contain personal
      or business-sensitive content).
"""
import logging
import os
import time
from datetime import datetime, timezone

import motor.motor_asyncio

from api.tiers import classify

logger = logging.getLogger("finsight.db")

MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI")
DB_NAME = "finsight_ai"

_client: motor.motor_asyncio.AsyncIOMotorClient | None = None

# Health-ping cache: (monotonic timestamp, result)
_PING_TTL_SECONDS = 30.0
_last_ping: tuple[float, bool] | None = None


def configured() -> bool:
    """True when a MongoDB connection string is available."""
    return bool(MONGO_URL)


def get_db():
    """Return the database handle, or None when Mongo is not configured."""
    global _client
    if not configured():
        return None
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URL,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )
    return _client[DB_NAME]


async def ping() -> bool:
    """Cheap connectivity check, cached for 30s so health checks stay fast."""
    global _last_ping
    if not configured():
        return False
    now = time.monotonic()
    if _last_ping is not None and now - _last_ping[0] < _PING_TTL_SECONDS:
        return _last_ping[1]
    try:
        await get_db().command("ping")
        result = True
    except Exception as e:
        logger.warning("MongoDB ping failed: %s", e)
        result = False
    _last_ping = (now, result)
    return result


async def log_prediction(input_data: dict, output_data: dict):
    db = get_db()
    if db is None:
        return
    try:
        now = datetime.now(timezone.utc)
        await db["prediction_logs"].insert_one({
            "input": input_data,
            "output": output_data,
            "timestamp": now,
            "logged_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        })
    except Exception as e:
        logger.warning("Failed to log prediction: %s", e)


async def log_research(query: str, answer: str):
    db = get_db()
    if db is None:
        return
    try:
        now = datetime.now(timezone.utc)
        await db["research_logs"].insert_one({
            "query": query,
            "answer": answer,
            "timestamp": now,
            "logged_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        })
    except Exception as e:
        logger.warning("Failed to log research query: %s", e)


async def get_analytics() -> dict:
    """Aggregate usage stats. Never returns raw query/input content."""
    db = get_db()
    if db is None:
        return {"available": False, "reason": "MongoDB not configured"}
    try:
        pred_col = db["prediction_logs"]
        res_col = db["research_logs"]

        total_predictions = await pred_col.count_documents({})
        total_questions = await res_col.count_documents({})

        avg_probability = 0.0
        tier_distribution = {"high": 0, "medium": 0, "low": 0}
        recent_activity: list[dict] = []

        if total_predictions > 0:
            cursor = pred_col.find(
                {}, {"output.probability_of_subscription": 1, "timestamp": 1}
            ).sort("timestamp", -1).limit(100)
            docs = await cursor.to_list(length=100)

            probs = [
                d["output"]["probability_of_subscription"]
                for d in docs
                if "output" in d and "probability_of_subscription" in d["output"]
            ]
            if probs:
                avg_probability = round(sum(probs) / len(probs) * 100, 1)
                for p in probs:
                    tier_distribution[classify(p)] += 1

            recent_activity = [
                {
                    "probability": round(d["output"]["probability_of_subscription"] * 100, 1),
                    "timestamp": d["timestamp"].isoformat()
                    if hasattr(d.get("timestamp"), "isoformat")
                    else str(d.get("timestamp", "")),
                }
                for d in reversed(docs[:10])
                if "output" in d and "probability_of_subscription" in d["output"]
            ]

        return {
            "available": True,
            "total_predictions": total_predictions,
            "total_questions": total_questions,
            "avg_probability": avg_probability,
            "tier_distribution": tier_distribution,
            "recent_activity": recent_activity,
        }
    except Exception as e:
        logger.warning("Analytics aggregation failed: %s", e)
        return {"available": False, "reason": "database error"}


async def get_recent_inputs(n: int = 200) -> list[dict]:
    """Return the raw input dicts of the last n predictions (for drift checks).

    Internal use only — this data is aggregated before leaving the API.
    """
    db = get_db()
    if db is None:
        return []
    try:
        cursor = db["prediction_logs"].find(
            {}, {"input": 1}
        ).sort("timestamp", -1).limit(n)
        docs = await cursor.to_list(length=n)
        return [d["input"] for d in docs if isinstance(d.get("input"), dict)]
    except Exception as e:
        logger.warning("Fetching recent inputs failed: %s", e)
        return []
