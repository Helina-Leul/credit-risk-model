import pandas as pd


def predict_risk(model, data):
    """
    Make prediction using trained model
    """

    df = pd.DataFrame([data])

    probability = model.predict_proba(df)[0][1]

    return probability
