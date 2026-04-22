import os
from datetime import datetime, timezone
import motor.motor_asyncio

# The same fail-safe logic we perfected earlier
MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
DB_NAME = "finsight_ai"

_client: motor.motor_asyncio.AsyncIOMotorClient | None = None

def get_client():
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URL,
            serverSelectionTimeoutMS=5000
        )
    return _client

def get_db():
    return get_client()[DB_NAME]

# Collections for Financial Analytics
def research_logs_col():
    return get_db()["research_logs"]

def lead_scoring_logs_col():
    return get_db()["lead_scoring_logs"]

async def log_financial_query(session_id: str, query: str, answer: str, category: str):
    col = research_logs_col()
    await col.insert_one({
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "category": category, # e.g., "Retirement", "Investment"
        "timestamp": datetime.now(timezone.utc)
    })