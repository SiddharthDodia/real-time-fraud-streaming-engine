import time
import json
import asyncio
import aiohttp
import pandas as pd
from pyspark.sql import SparkSession

API_URL = "http://127.0.0.1:8000/score"

async def send_transaction(session, tx):
    payload = {
        "cc_num": str(int(float(tx["cc_num"]))),
        "merch_lat": float(tx["merch_lat"]),
        "merch_long": float(tx["merch_long"]),
        "amt": float(tx["amt"]),
        "trans_unix_time": float(tx["unix_time"]),
        "is_fraud_true": int(tx["is_fraud"]),
        "rolling_1h_count": int(tx["rolling_1h_count"]),
        "rolling_1h_amt": float(tx["rolling_1h_amt"]),
        "rolling_24h_count": int(tx["rolling_24h_count"]),
        "rolling_24h_amt": float(tx["rolling_24h_amt"])
    }
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                pass # success
            else:
                text = await response.text()
                print(f"Failed to send tx: {response.status} - {text}")
    except Exception as e:
        print(f"Error sending tx: {e}")

async def blast_traffic(df):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, row in df.iterrows():
            tasks.append(asyncio.create_task(send_transaction(session, row)))
            # Add a tiny delay to simulate high but realistic throughput (e.g. 50-100 tx/sec)
            await asyncio.sleep(0.01) 
        
        # Wait for all requests to finish
        await asyncio.gather(*tasks)

def main():
    print("Loading test dataset for simulation...")
    # Use pandas directly for speed and simplicity since we just need a subset
    spark = SparkSession.builder.appName("SimulateTraffic").getOrCreate()
    df = spark.read.parquet("data/simulation_holdout")
    
    print("Sampling transactions (50% fraud to make dashboard interesting)...")
    df_pd = df.select("cc_num", "merch_lat", "merch_long", "amt", "unix_time", "is_fraud", 
                      "rolling_1h_count", "rolling_1h_amt", "rolling_24h_count", "rolling_24h_amt").limit(50000).toPandas()
    
    # Let's ensure we have a good mix of fraud in the stream so the dashboard lights up
    fraud_df = df_pd[df_pd['is_fraud'] == 1].sample(min(1000, len(df_pd[df_pd['is_fraud'] == 1])))
    legit_df = df_pd[df_pd['is_fraud'] == 0].sample(min(10000, len(df_pd[df_pd['is_fraud'] == 0])))
    
    stream_df = pd.concat([fraud_df, legit_df]).sample(frac=1).reset_index(drop=True)
    
    print(f"Starting simulation of {len(stream_df)} transactions...")
    asyncio.run(blast_traffic(stream_df))
    print("Simulation complete.")

if __name__ == "__main__":
    main()
