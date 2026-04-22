import os
from datetime import datetime, timezone
import motor.motor_asyncio

MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
DB_NAME = "finsight_ai"

_client: motor.motor_asyncio.AsyncIOMotorClient | None = None

def get_db():
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    return _client[DB_NAME]

async def log_prediction(input_data: dict, output_data: dict): # MUST BE NAMED EXACTLY THIS
    try:
        col = get_db()["prediction_logs"]
        await col.insert_one({
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.now(timezone.utc)
        })
    except Exception as e:
        print(f"DB Log Fail: {e}")

async def log_research(query: str, answer: str):
    try:
        col = get_db()["research_logs"]
        await col.insert_one({"query": query, "answer": answer, "timestamp": datetime.now(timezone.utc)})
    except Exception as e:
        print(f"DB Log Fail: {e}")