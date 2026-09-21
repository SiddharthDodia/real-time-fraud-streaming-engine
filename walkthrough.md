# Walkthrough: Credit Card Fraud Detection Pipeline Setup

I have completed the core implementation for the pipeline as per the approved plan. Here's a breakdown of what was accomplished and the next steps you need to take to run the system.

## Changes Made

### 1. Environment & Infrastructure Bootstrapping
- **`requirements.txt`**: Added `fastapi`, `uvicorn`, `redis`, `confluent-kafka`, `pyspark`, `xgboost`, and `scikit-learn`.
- **Dependencies Installed**: Successfully installed all requirements into the virtual environment (`venv`).
- **`docker-compose.yml`**: Created the docker configuration for Redis, Cassandra, Kafka, and Zookeeper.

### 2. Offline Processing & Feature Store Hydration
- **`offline_processing.py`**: Created a PySpark script that:
  - Broadcast joins `customers.csv` and `credit_card_fraud.csv`.
  - Calculates the Haversine distance and rolling window features (1h, 24h count and amt).
  - Performs an 80/20 temporal split, saving data to `data/train_baseline/` and `data/simulation_holdout/`.
  - Trains an XGBoost model on the 80% split and saves `xgb_model.json`.
  - Connects to Redis and hydrates the latest customer state using `HSET`.

### 3. Inference API
- **`main.py`**: Developed the FastAPI service that:
  - Loads the XGBoost model and establishes Redis/Kafka connection pools on startup.
  - Exposes a `POST /score` endpoint which accepts real-time transactions.
  - Fetches the profile from Redis using `HGETALL`.
  - Computes the inference probability.
  - Determines DECLINE/APPROVE (threshold > 0.85) and pushes the payload to the Kafka `scored-decisions` topic.

## Next Steps & Verification

> [!WARNING]
> The local Docker daemon is not currently running. Please ensure **Docker Desktop** is open and running before proceeding.

Once Docker is running, follow these steps to verify and start the pipeline:

### 1. Start Infrastructure
Open a terminal in the project directory and run:
```bash
docker-compose up -d
```
Then, create the Kafka topics:
```bash
docker exec -it <kafka-container-id> kafka-topics --create --topic incoming-transactions --bootstrap-server localhost:9092
docker exec -it <kafka-container-id> kafka-topics --create --topic scored-decisions --bootstrap-server localhost:9092
```

### 2. Run Offline Processing
Ensure your virtual environment is activated, then run the PySpark processing and training script:
```bash
source venv/bin/activate
python offline_processing.py
```
*(Note: This process may take a while depending on your system's memory due to the large ~11 GB dataset. Ensure no OOM errors occur.)*

### 3. Start the FastAPI Inference Server
Start the uvicorn server for your FastAPI application:
```bash
uvicorn main:app --reload
```

### 4. Test the API
You can test the inference endpoint with a `curl` request:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'Content-Type: application/json' \
  -d '{
  "cc_num": "YOUR_TEST_CC_NUM",
  "merch_lat": 35.1,
  "merch_long": -90.2,
  "amt": 150.0,
  "trans_unix_time": 1700000000
}'
```
You should see a JSON response indicating whether the transaction was approved or declined.
