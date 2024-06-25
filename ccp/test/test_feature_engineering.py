import pytest
import pandas as pd
from ccp.src.feature_engineering import create_features, encode_categorical

def test_create_features():
    df = pd.DataFrame({
        'cc_cons_apr': [100, 200, 300],
        'cc_cons_may': [150, 250, 350],
        'cc_cons_jun': [200, 300, 400],
        'dc_cons_apr': [50, 150, 250],
        'dc_cons_may': [75, 175, 275],
        'dc_cons_jun': [100, 200, 300]
    })
    df = create_features(df)
    assert 'cc_cons_lag1' in df.columns
    assert 'cc_cons_rolling_mean_3' in df.columns
    assert 'cc_cons_quarter' in df.columns

def test_encode_categorical():
    df = pd.DataFrame({
        'account_type': ['saving', 'current'],
        'gender': ['M', 'F'],
        'region_code': [1, 2]
    })
    df = encode_categorical(df)
    expected_columns = ['account_type_saving', 'gender_M', 'region_code_2']
    for col in expected_columns:
        assert col in df.columns
