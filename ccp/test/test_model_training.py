from pathlib import Path

import pytest
from ccp.src.model_training import train_model
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'credit_data.csv'


def test_train_model():
    df = pd.read_csv(DATA_PATH)
    df.fillna(method='ffill', inplace=True)
    model, X_test, y_test = train_model(df, 'cc_cons')
    assert model is not None
    assert not X_test.empty
    assert not y_test.empty
