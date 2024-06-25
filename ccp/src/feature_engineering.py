import pandas as pd

def create_features(df):
    required_columns = ['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun', 'dc_cons_apr', 'dc_cons_may', 'dc_cons_jun']

    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Or some other default value or logic

    df['cc_cons_lag1'] = df['cc_cons_apr'].shift(1)
    df['cc_cons_lag2'] = df['cc_cons_apr'].shift(2)
    df['dc_cons_lag1'] = df['dc_cons_apr'].shift(1)
    df['dc_cons_lag2'] = df['dc_cons_apr'].shift(2)

    df['cc_cons_rolling_mean_3'] = df['cc_cons_apr'].rolling(window=3).mean()
    df['cc_cons_rolling_sum_3'] = df['cc_cons_apr'].rolling(window=3).sum()
    df['dc_cons_rolling_mean_3'] = df['dc_cons_apr'].rolling(window=3).mean()
    df['dc_cons_rolling_sum_3'] = df['dc_cons_apr'].rolling(window=3).sum()

    df['cc_cons_quarter'] = df[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun']].sum(axis=1)
    df['dc_cons_quarter'] = df[['dc_cons_apr', 'dc_cons_may', 'dc_cons_jun']].sum(axis=1)

    df['credit_debit_ratio'] = df['cc_cons_apr'] / (df['dc_cons_apr'] + 1)
    return df

def encode_categorical(df):
    df = pd.get_dummies(df, columns=['account_type', 'gender', 'region_code'], drop_first=True)
    return df
