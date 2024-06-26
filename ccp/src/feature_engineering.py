import pandas as pd

def perform_feature_engineering(data):
    # Ensure necessary columns
    if 'day_of_week' not in data.columns:
        data['day_of_week'] = data.index.dayofweek
    if 'month' not in data.columns:
        data['month'] = data.index.month

    # Create lag features
    data['amount_lag_1'] = data['amount'].shift(1)
    data['amount_lag_7'] = data['amount'].shift(7)

    # Create rolling mean features
    data['amount_roll_mean_7'] = data['amount'].rolling(window=7).mean()

    # Create additional features
    data['transaction_count_last_7_days'] = data['amount'].resample('D').count().rolling(window=7).sum().shift(1)
    data['transaction_amount_mean_last_7_days'] = data['amount'].resample('D').sum().rolling(window=7).mean().shift(1)
    data['transaction_amount_std_last_7_days'] = data['amount'].resample('D').sum().rolling(window=7).std().shift(1)

    # Fill missing values
    data.fillna(0, inplace=True)

    return data
