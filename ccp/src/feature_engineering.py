import pandas as pd

def perform_feature_engineering(data_path):
    # Load data with date parsing
    data = pd.read_csv(data_path, parse_dates=['transaction_date'])

    # Debug: Print the columns to verify they are loaded correctly
    print("Columns in the DataFrame:", data.columns.tolist())

    # Feature engineering: Ensure 'day_of_week' and 'month' columns exist
    if 'day_of_week' not in data.columns:
        data['day_of_week'] = data['transaction_date'].dt.dayofweek
    if 'month' not in data.columns:
        data['month'] = data['transaction_date'].dt.month

    # Create lag features
    data.sort_values('transaction_date', inplace=True)
    data['amount_lag_1'] = data['amount'].shift(1)
    data['amount_lag_7'] = data['amount'].shift(7)

    # Create rolling mean features
    data['amount_roll_mean_7'] = data['amount'].rolling(window=7).mean()

    # Fill missing values for lag and rolling mean features
    data.fillna(0, inplace=True)

    # Create additional features
    data.set_index('transaction_date', inplace=True)
    data['transaction_count_last_7_days'] = data['amount'].resample('D').count().rolling(window=7).sum().shift(1)
    data['transaction_amount_mean_last_7_days'] = data['amount'].resample('D').sum().rolling(window=7).mean().shift(1)
    data['transaction_amount_std_last_7_days'] = data['amount'].resample('D').sum().rolling(window=7).std().shift(1)

    # Fill missing values for new features
    data.fillna(0, inplace=True)

    # Reset index
    data.reset_index(inplace=True)

    # Save the engineered data
    data.to_csv(data_path, index=False)
    print("Feature engineering complete.")
