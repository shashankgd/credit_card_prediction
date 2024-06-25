import pandas as pd

from pathlib import Path

def load_data(file_path):
    return pd.read_csv(Path(file_path))

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    df['month'] = pd.to_datetime(df['id'], unit='s').dt.month
    df['day_of_week'] = pd.to_datetime(df['id'], unit='s').dt.dayofweek
    return df
