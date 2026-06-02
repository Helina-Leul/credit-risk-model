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
# ==========================================
# TASK 4 - PROXY TARGET VARIABLE ENGINEERING
# ==========================================

from sklearn.cluster import KMeans

# -----------------------------
# Create RFM Features
# -----------------------------

snapshot_date = (
    df["TransactionStartTime"].max()
    + pd.Timedelta(days=1)
)

rfm = (
    df.groupby("CustomerId")
    .agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (
                snapshot_date - x.max()
            ).days
        ),
        Frequency=(
            "TransactionId",
            "count"
        ),
        Monetary=(
            "Amount",
            "sum"
        )
    )
    .reset_index()
)

# -----------------------------
# Scale RFM Features
# -----------------------------

rfm_features = [
    "Recency",
    "Frequency",
    "Monetary"
]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm[rfm_features]
)

# -----------------------------
# K-Means Clustering
# -----------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

# -----------------------------
# Identify High-Risk Cluster
# -----------------------------

cluster_summary = (
    rfm.groupby("Cluster")
    .agg(
        {
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        }
    )
)

print("\nCluster Summary")
print(cluster_summary)

high_risk_cluster = (
    cluster_summary.sort_values(
        by=["Frequency", "Monetary"],
        ascending=[True, True]
    ).index[0]
)

rfm["is_high_risk"] = (
    rfm["Cluster"] == high_risk_cluster
).astype(int)

# -----------------------------
# Merge Back to Main Dataset
# -----------------------------

df = df.merge(
    rfm[
        [
            "CustomerId",
            "is_high_risk"
        ]
    ],
    on="CustomerId",
    how="left"
)

# -----------------------------
# Save Final Dataset
# -----------------------------

df.to_csv(
    "data/processed/processed_data_with_target.csv",
    index=False
)

print(
    "\nProcessed dataset with target saved successfully!"
)

print(
    "\nTarget Distribution:"
)

print(
    df["is_high_risk"]
    .value_counts()
)