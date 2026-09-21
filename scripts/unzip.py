import kagglehub


# Download latest version
path = kagglehub.dataset_download("karthikgangula/credit-card-fraud-mega-dataset")

print("Path to dataset files:", path)