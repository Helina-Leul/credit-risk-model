import pandas as pd


def test_target_column_exists():
    df = pd.read_csv("data/processed/processed_data_with_target.csv")
    assert "is_high_risk" in df.columns


def test_no_missing_target_values():
    df = pd.read_csv("data/processed/processed_data_with_target.csv")
    assert df["is_high_risk"].isnull().sum() == 0
