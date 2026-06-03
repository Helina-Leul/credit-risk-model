import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Load data
df = pd.read_csv("data/processed/processed_data_with_target.csv")

# Handle missing values
df["Std_Transaction_Amount"] = df["Std_Transaction_Amount"].fillna(0)

# Drop ID and datetime columns
drop_cols = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime"
]

df = df.drop(columns=drop_cols)

# Encode categorical variables
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("is_high_risk", axis=1)
y = df["is_high_risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),

    "RandomForest": GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None]
        },
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
}

# Train models
for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        # Get best RF model after tuning
        if model_name == "RandomForest":
            print("\nBest Parameters:", model.best_params_)
            mlflow.log_params(model.best_params_)
            model = model.best_estimator_

        predictions = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(model, model_name)

        print(f"\n{model_name}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC:", roc_auc)