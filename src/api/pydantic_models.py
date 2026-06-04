from pydantic import BaseModel


class CustomerData(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: int
    FraudResult: int
    Total_Transaction_Amount: float
    Average_Transaction_Amount: float
    Transaction_Count: int
    Std_Transaction_Amount: float
    Transaction_Hour: int
    Transaction_Day: int
    Transaction_Month: int
    Transaction_Year: int


class PredictionResponse(BaseModel):
    risk_probability: float
