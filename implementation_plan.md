# Credit Card Fraud Detection Pipeline

This document outlines the proposed implementation plan to address the requested three phases: Environment & Infrastructure Bootstrapping, Offline Processing & Feature Store Hydration, and Building the Inference API.

## Proposed Changes

### Phase 1: Environment & Infrastructure Bootstrapping
- Update `requirements.txt` to include `fastapi`, `uvicorn`, `redis`, `confluent-kafka`, `pyspark`, `xgboost`, and `scikit-learn`.
- Create a `docker-compose.yml` at the project root to provision local services:
  - **Redis** for the low-latency feature cache.
  - **Cassandra** as the historical wide-column sink.
  - **Kafka** & **Zookeeper** for event streaming.
- Run `docker-compose up -d` to launch the infrastructure and create Kafka topics `incoming-transactions` and `scored-decisions`.

### Phase 2: Offline Processing & Feature Store Hydration
Create a new PySpark pipeline script `offline_processing.py`:
- Process `customers.csv` and `credit_card_fraud.csv`.
- Execute a broadcast join on `cc_num`.
- Engineer features:
  - Haversine distance between customer and merchant locations.
  - Window functions for rolling 1-hour and 24-hour transaction counts and spend amounts.
- Perform a temporal train/test split (80% train, 20% holdout) ordered by `trans_datetime`, saving to `data/train_baseline/` and `data/simulation_holdout/`.
- Train an `XGBoost` classifier on the training split, saving the artifact as `xgb_model.json`.
- Extract the most recent state for each `cc_num` in the training set and use a script to hydrate Redis (via HSET commands) with baseline profiles (location, DOB, current rolling aggregates).

### Phase 3: Building the Inference API
Create a `main.py` file to serve the FastAPI application:
- **Startup**: Load `xgb_model.json` and initialize Redis and Kafka connection pools.
- **Endpoint**: Expose a `POST /score` endpoint.
- **Logic**: 
  - Receive an incoming transaction.
  - Execute an `HGETALL` from Redis for the customer's baseline profile.
  - Calculate real-time distance and time-since-last-transaction.
  - Execute `model.predict_proba()` to generate a fraud probability score.
- **Routing**: Return DECLINE/APPROVE (e.g. threshold > 0.85) to the client and asynchronously push the full payload to the `scored-decisions` Kafka topic.

## Verification Plan

### Automated Tests
- Validate PySpark data transformations run correctly without OOM errors.
- Validate FastAPI unit tests for the `/score` endpoint functionality.

### Manual Verification
- Check Redis to verify data hydration works.
- Send test API requests using `curl` and observe the inference output and the Kafka consumer of `scored-decisions`.

## User Review Required

> [!IMPORTANT]
> The dataset size is large (~11 GB for credit card fraud). Local PySpark execution with 8GB driver memory might still be tight depending on available system RAM. Please verify that your system can accommodate these resources.
> Also, for Haversine distance and rolling aggregations in PySpark, we'll need to define some UDFs or use built-in functions cautiously to maintain performance.

Please review this implementation plan. Once approved, I will proceed with creating the configuration files and processing scripts.
