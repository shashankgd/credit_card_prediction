from pathlib import Path

import pytest
from ccp.src.data_preprocessing import load_data, preprocess_data

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'credit_data.csv'

def test_load_data():
    df = load_data(DATA_PATH)
    assert not df.empty

def test_preprocess_data():
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    assert 'month' in df.columns
    assert 'day_of_week' in df.columns
