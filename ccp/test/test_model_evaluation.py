import pytest
from ccp.src.model_evaluation import evaluate_model
from ccp.src.model_training import train_model
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'credit_data.csv'

def test_evaluate_model():
    df = pd.read_csv(DATA_PATH)
    df.fillna(method='ffill', inplace=True)
    model, X_test, y_test = train_model(df, 'cc_cons')
    rmse, mae = evaluate_model(model, X_test, y_test)
    assert rmse > 0
    assert mae > 0
