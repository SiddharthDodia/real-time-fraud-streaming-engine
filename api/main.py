import os
import json
import time
import redis
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fraud Detection Inference API")

# Global variables
model = None
redis_client = None
kafka_producer = None



class Transaction(BaseModel):
    cc_num: str
    merch_lat: float
    merch_long: float
    amt: float
    trans_unix_time: float
    is_fraud_true: int = None
    rolling_1h_count: int = None
    rolling_1h_amt: float = None
    rolling_24h_count: int = None
    rolling_24h_amt: float = None

@app.on_event("startup")
def startup_event():
    global model, redis_client, kafka_producer
    
    # Load XGBoost model
    try:
        model = xgb.XGBClassifier()
        model.load_model("ml/models/xgb_model.json")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
    
    # Initialize Redis connection pool
    pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client = redis.Redis(connection_pool=pool)
    
    # Initialize Kafka Producer
    conf = {'bootstrap.servers': 'localhost:9092'}
    kafka_producer = Producer(conf)

@app.post("/score")
async def score_transaction(tx: Transaction):
    start_time = time.time()
    global model, redis_client, kafka_producer
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Fetch customer profile from Redis
    profile = redis_client.hgetall(tx.cc_num)
    if not profile:
        profile = {} # Cold-start customer
    
    # In a real system, we would update the rolling counts in Redis or a fast stream processor.
    # For inference simulation, we use the point-in-time values if provided, else fallback to Redis.
    features = pd.DataFrame([{
        "amt": tx.amt,
        "rolling_1h_count": tx.rolling_1h_count if tx.rolling_1h_count is not None else float(profile.get("rolling_1h_count", 0)),
        "rolling_1h_amt": tx.rolling_1h_amt if tx.rolling_1h_amt is not None else float(profile.get("rolling_1h_amt", 0.0)),
        "rolling_24h_count": tx.rolling_24h_count if tx.rolling_24h_count is not None else float(profile.get("rolling_24h_count", 0)),
        "rolling_24h_amt": tx.rolling_24h_amt if tx.rolling_24h_amt is not None else float(profile.get("rolling_24h_amt", 0.0))
    }])
    
    # Inference
    proba = model.predict_proba(features)[0][1]
    threshold = float(os.environ.get("FRAUD_DECISION_THRESHOLD", 0.0731))
    decision = "DECLINE" if proba >= threshold else "APPROVE"
    
    latency_ms = (time.time() - start_time) * 1000
    
    response_payload = {
        "cc_num": tx.cc_num,
        "decision": decision,
        "fraud_probability": float(proba),
        "amt": tx.amt,
        "is_fraud_true": tx.is_fraud_true,
        "latency_ms": latency_ms
    }
    
    # Asynchronously push to Kafka
    try:
        kafka_producer.produce('scored-decisions', key=tx.cc_num, value=json.dumps(response_payload))
        kafka_producer.poll(0) # trigger delivery callbacks
    except Exception as e:
        print(f"Error producing to Kafka: {e}")
        
    return response_payload
