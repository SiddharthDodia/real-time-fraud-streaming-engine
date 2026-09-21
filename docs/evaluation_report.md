# XGBoost Model Evaluation Report

I've successfully updated and run the evaluation script to generate the PR-Curve, analyze the financial impact of different thresholds, and extract feature importances. 

## 1. Precision-Recall Curve

The PR Curve is plotted below. The model achieved a PR-AUC of **0.6606**. Look for the "elbow" where precision begins to drop sharply; this indicates the point where forcing the model to capture more fraud results in an exponentially higher number of false positives.

![Precision-Recall Curve](/Users/siddharthdodia/Documents/Projects/Credit Card Fraud Detection/pr_curve.png)

## 2. Financial Matrix Analysis

We assumed an **operational cost of $10.00 per manual review** (false positive). The false negative cost is the actual dollar amount of the fraudulent transaction that slipped through.

We evaluated 5 threshold percentiles (80th, 85th, 90th, 95th, 99th):

| Threshold | False Positives | FP Cost | False Negatives (Missed Fraud) | FN Cost (Dollar Loss) | Total Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0010** | 1,384,285 | $13,842,850.00 | 761 | $259,967.58 | $14,102,817.58 |
| **0.0015** | 1,030,350 | $10,303,500.00 | 1,067 | $280,580.92 | $10,584,080.92 |
| **0.0026** | 679,784 | $6,797,840.00 | 1,536 | $302,103.76 | $7,099,943.76 |
| **0.0061** | 332,478 | $3,324,780.00 | 2,648 | $419,962.18 | $3,744,742.18 |
| **0.0731** | 45,676 | $456,760.00 | 8,036 | $1,764,816.90 | **$2,221,576.90** |

> [!TIP]
> **Optimal Threshold**: Based on this financial matrix, setting a stricter threshold of **0.0731** (the 99th percentile of model scores) minimizes total cost to **$2.22M**. Lower thresholds flag too many legitimate transactions, ballooning the operational review costs beyond the value of the fraud they prevent.

## 3. Feature Importance

Here are the XGBoost feature importance scores (gain):

1. **`rolling_24h_count`**: 0.4645
2. **`amt`**: 0.1561
3. **`rolling_1h_count`**: 0.1542
4. **`rolling_1h_amt`**: 0.1469
5. **`rolling_24h_amt`**: 0.0783
6. **`distance_km`**: 0.0000

> [!WARNING]
> **Geospatial Feature Failure**
> The `distance_km` (Haversine distance) has an importance of exactly **0.0000**. The model found no predictive value in this feature. This could mean either:
> 1. Fraudsters in this dataset are spoofing locations effectively, or fraud happens locally.
> 2. There is a bug or misalignment in how `merch_lat`/`merch_long` and customer `lat`/`long` were joined or calculated in the PySpark pipeline. 
> 
> The temporal velocity features (rolling counts and amounts) are doing all the heavy lifting!
