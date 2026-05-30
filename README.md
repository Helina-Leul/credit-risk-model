# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end Credit Risk Probability Model for Bati Bank using alternative transaction data from an eCommerce platform. The objective is to support Buy-Now-Pay-Later (BNPL) services by estimating customer credit risk and assigning risk-based credit scores.

The solution follows Basel II principles and includes:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Proxy Target Variable Construction using RFM Analysis
* Credit Risk Modeling
* Experiment Tracking with MLflow
* Model Deployment using FastAPI
* Containerization with Docker
* CI/CD Automation with GitHub Actions

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

# Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability

Basel II requires financial institutions to establish reliable, transparent, and well-documented credit risk measurement systems. The framework emphasizes accurate estimation of the Probability of Default (PD), clear documentation of model assumptions, and continuous model monitoring.

As a result, credit scoring models must be interpretable and explainable. Banks should be able to justify lending decisions to regulators, auditors, and customers. Well-documented models improve governance, support regulatory compliance, and facilitate model validation and auditing processes.

For this project, model transparency will be an important consideration when selecting features and choosing between alternative modeling approaches.

## 2. Why a Proxy Variable is Necessary

The dataset does not contain a true loan default indicator because it consists of eCommerce transaction records rather than lending outcomes. Since supervised machine learning requires a target variable, a proxy variable must be constructed to represent customer credit risk.

This project will use customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) analysis to identify less engaged customers. Customers belonging to low-engagement segments will be treated as high-risk proxies.

However, proxy-based targets introduce several risks:

* The proxy may not accurately represent actual loan default behavior.
* Customer inactivity does not necessarily imply inability or unwillingness to repay credit.
* Incorrect proxy labels may introduce bias into the model.
* Model performance may not fully translate to real-world credit risk prediction.

Therefore, the proxy target should be treated as a modeling assumption rather than a true measure of default risk.

## 3. Interpretable Models vs High-Performance Models

A Logistic Regression model combined with Weight of Evidence (WoE) transformation provides strong interpretability. Each feature's contribution to risk can be clearly explained, making the model easier to validate and justify under Basel II requirements.

In contrast, advanced models such as Gradient Boosting, XGBoost, or LightGBM often achieve higher predictive performance by capturing complex nonlinear relationships and interactions among variables. However, these models are less transparent and can be more difficult to explain to regulators and stakeholders.

| Aspect                | Logistic Regression + WoE | Gradient Boosting |
| --------------------- | ------------------------- | ----------------- |
| Interpretability      | High                      | Low               |
| Regulatory Acceptance | High                      | Moderate          |
| Explainability        | Easy                      | Difficult         |
| Predictive Power      | Moderate                  | High              |
| Model Complexity      | Low                       | High              |
| Auditability          | Easy                      | More Challenging  |

For regulated financial applications, model selection requires balancing predictive performance with explainability, governance, and regulatory compliance.
