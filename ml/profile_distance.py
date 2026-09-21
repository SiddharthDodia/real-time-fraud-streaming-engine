import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

def main():
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("ProfileDistance") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    print("Loading data...")
    # We can use the test set or train set. We'll use the test set for quick profiling.
    df = spark.read.parquet("data/simulation_holdout")

    print("Converting to Pandas DataFrame...")
    # We only need distance_km and is_fraud
    pd_df = df.select("distance_km", "is_fraud").toPandas()
    
    print("\n--- Univariate Analysis: distance_km ---")
    print(pd_df["distance_km"].describe())
    
    print("\n--- Bivariate Analysis: distance_km by is_fraud ---")
    print(pd_df.groupby("is_fraud")["distance_km"].describe())

    # Plotting
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Univariate Plot
    sns.histplot(pd_df["distance_km"], bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Univariate Distribution of distance_km")
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Frequency")
    
    # Bivariate Plot (Boxplot)
    sns.boxplot(x="is_fraud", y="distance_km", data=pd_df, ax=axes[1])
    axes[1].set_title("Bivariate Distribution: distance_km by is_fraud")
    axes[1].set_xlabel("Is Fraud (0 = No, 1 = Yes)")
    axes[1].set_ylabel("Distance (km)")
    
    plt.tight_layout()
    plt.savefig("distance_profiling.png")
    print("Plots saved to distance_profiling.png")

if __name__ == "__main__":
    main()
