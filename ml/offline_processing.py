import os
import json
import redis
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split



def main():
    spark = SparkSession.builder \
        .appName("CreditCardFraudProcessing") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .getOrCreate()



    print("Reading data...")
    # Read datasets
    customers_df = spark.read.csv("data/archive/customers.csv", header=True, inferSchema=True, sep="|")
    transactions_df = spark.read.csv("data/archive/credit_card_fraud.csv", header=True, inferSchema=True)

    print("Joining data...")
    # Drop overlapping columns from transactions_df to avoid ambiguity
    overlap_cols = ["ssn", "first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "acct_num", "profile"]
    transactions_df = transactions_df.drop(*overlap_cols)
    # Broadcast join
    joined_df = transactions_df.join(F.broadcast(customers_df), on="cc_num", how="inner")
    
    print("Feature engineering...")
    


    # Rolling window features
    # Window ordered by time, partitioned by cc_num
    # For rolling 1 hour (3600 seconds) and 24 hours (86400 seconds)
    joined_df = joined_df.withColumn("trans_unix_time", F.col("unix_time").cast("long"))
    
    window_1h = Window.partitionBy("cc_num").orderBy("trans_unix_time").rangeBetween(-3600, 0)
    window_24h = Window.partitionBy("cc_num").orderBy("trans_unix_time").rangeBetween(-86400, 0)
    
    joined_df = joined_df.withColumn("rolling_1h_count", F.count("trans_num").over(window_1h))
    joined_df = joined_df.withColumn("rolling_1h_amt", F.sum("amt").over(window_1h))
    joined_df = joined_df.withColumn("rolling_24h_count", F.count("trans_num").over(window_24h))
    joined_df = joined_df.withColumn("rolling_24h_amt", F.sum("amt").over(window_24h))

    # Calculate percentiles for temporal split
    print("Splitting data...")
    # To strictly split by time, we can find the split timestamp
    percentiles = joined_df.approxQuantile("trans_unix_time", [0.8], 0.01)
    split_time = percentiles[0]
    
    train_df = joined_df.filter(F.col("trans_unix_time") <= split_time)
    test_df = joined_df.filter(F.col("trans_unix_time") > split_time)

    # Save splits
    train_df.write.mode("overwrite").parquet("data/train_baseline")
    test_df.write.mode("overwrite").parquet("data/simulation_holdout")

    print("Training XGBoost...")
    # Collect train data to Pandas for XGBoost (assuming it fits in memory after sampling/aggregation, but for simplicity we convert directly)
    # In production with 11GB, we'd use XGBoost Spark integration. For this local script, we'll convert to pandas.
    # To prevent OOM on local machine during toPandas, we might select only required features.
    features = ["amt", "rolling_1h_count", "rolling_1h_amt", "rolling_24h_count", "rolling_24h_amt"]
    train_pd = train_df.select(features + ["is_fraud"]).toPandas()
    
    X_train = train_pd[features]
    y_train = train_pd["is_fraud"]
    
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    
    def objective(trial):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_pred_proba)
    
    print("Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print("Best hyperparameters: ", study.best_params)
    
    print("Training final model...")
    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["use_label_encoder"] = False
    best_params["eval_metric"] = "logloss"
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    final_model.save_model("ml/models/xgb_model.json")
    print("Model saved to ml/models/xgb_model.json")

    print("Hydrating Redis...")
    # Get the most recent state for each cc_num
    window_latest = Window.partitionBy("cc_num").orderBy(F.col("trans_unix_time").desc())
    latest_state_df = train_df.withColumn("row_num", F.row_number().over(window_latest)) \
                                .filter(F.col("row_num") == 1)
    
    latest_state_pd = latest_state_df.select("cc_num", "lat", "long", "dob", "rolling_1h_count", "rolling_1h_amt", "rolling_24h_count", "rolling_24h_amt", "trans_unix_time").toPandas()
    
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    for _, row in latest_state_pd.iterrows():
        cc_num = str(row['cc_num'])
        r.hset(cc_num, mapping={
            "lat": str(row['lat']),
            "long": str(row['long']),
            "dob": str(row['dob']),
            "rolling_1h_count": str(row['rolling_1h_count']),
            "rolling_1h_amt": str(row['rolling_1h_amt']),
            "rolling_24h_count": str(row['rolling_24h_count']),
            "rolling_24h_amt": str(row['rolling_24h_amt']),
            "last_trans_unix_time": str(row['trans_unix_time'])
        })
    print("Redis hydration complete.")

if __name__ == "__main__":
    main()
