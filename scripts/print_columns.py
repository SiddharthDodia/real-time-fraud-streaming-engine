import pandas as pd

def main():
    print("--- Transactions Dataset (credit_card_fraud.csv) ---")
    try:
        # Read just the first row to get columns quickly
        transactions = pd.read_csv("data/archive/credit_card_fraud.csv", nrows=0)
        print(list(transactions.columns))
    except Exception as e:
        print(f"Error reading transactions: {e}")

    print("\n--- Customers Dataset (customers.csv) ---")
    try:
        customers = pd.read_csv("data/archive/customers.csv", nrows=0, sep="|")
        print(list(customers.columns))
    except Exception as e:
        print(f"Error reading customers: {e}")

if __name__ == "__main__":
    main()
