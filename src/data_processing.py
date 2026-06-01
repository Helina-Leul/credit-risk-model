import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Load Data
# -----------------------------

df = pd.read_csv("data/raw/training.csv")

# -----------------------------
# Date Features
# -----------------------------

df["TransactionStartTime"] = pd.to_datetime(
    df["TransactionStartTime"]
)

# Aggregate Features

customer_features = (
    df.groupby("CustomerId")
    .agg(
        Total_Transaction_Amount=("Amount", "sum"),
        Average_Transaction_Amount=("Amount", "mean"),
        Transaction_Count=("Amount", "count"),
        Std_Transaction_Amount=("Amount", "std"),
    )
    .reset_index()
)

df = df.merge(customer_features, on="CustomerId", how="left")

# Time Features

df["Transaction_Hour"] = df["TransactionStartTime"].dt.hour
df["Transaction_Day"] = df["TransactionStartTime"].dt.day
df["Transaction_Month"] = df["TransactionStartTime"].dt.month
df["Transaction_Year"] = df["TransactionStartTime"].dt.year

# -----------------------------
# Select Features
# -----------------------------

numeric_features = [
    "Amount",
    "Value",
    "PricingStrategy",
    "Total_Transaction_Amount",
    "Average_Transaction_Amount",
    "Transaction_Count",
    "Std_Transaction_Amount",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Transaction_Year"
]

categorical_features = [
    "CurrencyCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId"
]

# -----------------------------
# Numeric Pipeline
# -----------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# -----------------------------
# Categorical Pipeline
# -----------------------------

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# -----------------------------
# Full Pipeline
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
    ]
)

# -----------------------------
# Fit Pipeline
# -----------------------------

X_processed = pipeline.fit_transform(df)

# Convert processed data to DataFrame
processed_df = pd.DataFrame(
    X_processed.toarray()
    if hasattr(X_processed, "toarray")
    else X_processed
)

# Save processed dataset
processed_df.to_csv(
    "data/processed/processed_data.csv",
    index=False
)

print("Processed Shape:", processed_df.shape)
print("Processed data saved successfully!")