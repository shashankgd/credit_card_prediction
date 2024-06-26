import pandas as pd

def handle_missing_values(data):
    data.fillna(data.mean(), inplace=True)
    return data
