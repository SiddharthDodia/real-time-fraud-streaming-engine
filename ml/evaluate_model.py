import os
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

def main():
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("EvaluateFraudModel") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    print("Loading test data...")
    test_df = spark.read.parquet("data/simulation_holdout")

    features = ["amt", "rolling_1h_count", "rolling_1h_amt", "rolling_24h_count", "rolling_24h_amt"]
    print("Converting to Pandas DataFrame...")
    test_pd = test_df.select(features + ["is_fraud"]).toPandas()

    X_test = test_pd[features]
    y_test = test_pd["is_fraud"]
    amt_test = test_pd["amt"]

    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model("ml/models/xgb_model.json")

    print("Predicting probabilities...")
    y_scores = model.predict_proba(X_test)[:, 1]

    os.makedirs("docs/assets", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- 1. Financial Impact Curve ---
    print("Generating Financial Impact Curve...")
    ops_cost_per_review = 10.0
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        fp_cost = false_positives * ops_cost_per_review
        fn_mask = (y_pred == 0) & (y_test == 1)
        fn_cost = np.sum(amt_test[fn_mask])
        costs.append(fp_cost + fn_cost)
        
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, lw=2, color="blue", label="Total Cost (FP Review + FN Fraud Loss)")
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Optimal Threshold ({best_threshold:.3f})")
    plt.title("Financial Impact Curve (Cost-Savings by Threshold)")
    plt.xlabel("Decision Threshold (Probability)")
    plt.ylabel("Total Cost ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/assets/financial_impact.png")
    plt.close()

    # --- 2. Probability Density Plot ---
    print("Generating Probability Density Plot...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_scores[y_test == 0], fill=True, color="green", label="Legitimate (y=0)", alpha=0.5)
    sns.kdeplot(y_scores[y_test == 1], fill=True, color="red", label="Fraudulent (y=1)", alpha=0.5)
    plt.title("Probability Density Plot (Class Separation)")
    plt.xlabel("Predicted Fraud Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/assets/probability_density.png")
    plt.close()

    # --- 3. Feature Importance Bar Chart ---
    print("Generating Feature Importance Bar Chart...")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], color="skyblue")
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("docs/assets/feature_importance.png")
    plt.close()

    # --- 4. Custom Financial Confusion Matrix ---
    print("Generating Financial Confusion Matrix...")
    y_pred_opt = (y_scores >= best_threshold).astype(int)
    
    tp_savings = np.sum(amt_test[(y_pred_opt == 1) & (y_test == 1)])
    fn_loss = np.sum(amt_test[(y_pred_opt == 0) & (y_test == 1)])
    fp_cost = np.sum((y_pred_opt == 1) & (y_test == 0)) * ops_cost_per_review
    tn_cost = 0.0 # No cost for correctly approving a legit transaction
    
    matrix_vals = np.array([[tn_cost, fp_cost],
                            [fn_loss, tp_savings]])
    
    labels = np.array([[f"True Negatives\n$0.00", f"False Positives\n(Review Cost)\n${fp_cost:,.2f}"],
                       [f"False Negatives\n(Fraud Missed)\n${fn_loss:,.2f}", f"True Positives\n(Fraud Caught)\n${tp_savings:,.2f}"]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_vals, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Predicted Legit", "Predicted Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"],
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(f"Financial Confusion Matrix (Threshold = {best_threshold:.3f})")
    plt.tight_layout()
    plt.savefig("docs/assets/financial_confusion_matrix.png")
    plt.close()
    
    print("All charts generated and saved to docs/assets/")

if __name__ == "__main__":
    main()
