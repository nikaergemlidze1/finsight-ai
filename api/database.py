import os
from datetime import datetime, timezone
import motor.motor_asyncio

MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
DB_NAME = "finsight_ai"

_client: motor.motor_asyncio.AsyncIOMotorClient | None = None

def get_db():
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URL,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )
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

async def get_analytics() -> dict:
    try:
        db = get_db()
        pred_col = db["prediction_logs"]
        res_col  = db["research_logs"]

        total_predictions = await pred_col.count_documents({})
        total_questions   = await res_col.count_documents({})

        # Average probability from last 100 predictions
        avg_probability = 0.0
        tier_distribution = {"high": 0, "medium": 0, "low": 0}
        recent_activity: list[dict] = []

        if total_predictions > 0:
            cursor = pred_col.find(
                {}, {"output.probability_of_subscription": 1, "timestamp": 1}
            ).sort("timestamp", -1).limit(100)
            docs = await cursor.to_list(length=100)

            probs = [
                d["output"]["probability_of_subscription"] * 100
                for d in docs
                if "output" in d and "probability_of_subscription" in d["output"]
            ]
            if probs:
                avg_probability = round(sum(probs) / len(probs), 1)
                for p in probs:
                    if p >= 60:
                        tier_distribution["high"] += 1
                    elif p >= 30:
                        tier_distribution["medium"] += 1
                    else:
                        tier_distribution["low"] += 1

            recent_activity = [
                {
                    "probability": round(d["output"]["probability_of_subscription"] * 100, 1),
                    "timestamp": d.get("timestamp", "").isoformat() if hasattr(d.get("timestamp", ""), "isoformat") else str(d.get("timestamp", "")),
                }
                for d in reversed(docs[:10])
                if "output" in d and "probability_of_subscription" in d["output"]
            ]

        # Recent research questions
        recent_questions: list[dict] = []
        if total_questions > 0:
            q_cursor = res_col.find(
                {}, {"query": 1, "timestamp": 1}
            ).sort("timestamp", -1).limit(10)
            q_docs = await q_cursor.to_list(length=10)
            recent_questions = [
                {
                    "query": d.get("query", ""),
                    "timestamp": d["timestamp"].isoformat() if hasattr(d.get("timestamp"), "isoformat") else str(d.get("timestamp", "")),
                }
                for d in q_docs
            ]

        return {
            "available": True,
            "total_predictions": total_predictions,
            "total_questions": total_questions,
            "avg_probability": avg_probability,
            "tier_distribution": tier_distribution,
            "recent_activity": recent_activity,
            "recent_questions": recent_questions,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}