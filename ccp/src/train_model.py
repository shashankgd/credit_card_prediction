import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import datetime

def train_and_test_model(data_path, model_path, training_results_path, testing_results_path, metrics_path):
    # Load the data with date parsing
    data = pd.read_csv(data_path, parse_dates=['transaction_date'])

    # Ensure necessary feature engineering
    if 'day_of_week' not in data.columns:
        data['day_of_week'] = data['transaction_date'].dt.dayofweek
    if 'month' not in data.columns:
        data['month'] = data['transaction_date'].dt.month

    # Check data types
    print("Data types:\n", data.dtypes)

    # Define features and target
    X = data.drop(['transaction_id', 'user_id', 'transaction_date', 'amount'], axis=1)
    y = data['amount']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = lgb.LGBMRegressor()

    # Reduced hyperparameter tuning
    param_grid = {
        'num_leaves': [31, 50],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200],
        'feature_fraction': [0.8, 1.0],
        'bagging_fraction': [0.8, 1.0],
        'bagging_freq': [1, 5]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Save the model
    joblib.dump(best_model, model_path)

    # Training and testing
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Print metrics
    print(f'Training MSE: {mse_train}, R2: {r2_train}')
    print(f'Testing MSE: {mse_test}, R2: {r2_test}')

    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.plot(y_train.values, label='Actual')
    plt.plot(y_train_pred, label='Predicted')
    plt.title('Training Data: Actual vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Amount')
    plt.legend()
    plt.savefig(training_results_path)
    plt.close()

    # Plot testing results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_test_pred, label='Predicted')
    plt.title('Testing Data: Actual vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Amount')
    plt.legend()
    plt.savefig(testing_results_path)
    plt.close()

    # Plot metrics
    plt.figure(figsize=(10, 5))
    metrics_data = {'Metric': ['MSE', 'R2'], 'Training': [mse_train, r2_train], 'Testing': [mse_test, r2_test]}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Metric', inplace=True)
    metrics_df.plot(kind='bar', title='Model Performance Metrics')
    plt.ylabel('Score')
    plt.savefig(metrics_path)
    plt.close()

    return best_model, X_test, y_test

def predict_next_week(model, data_path):
    # Load the data with date parsing
    data = pd.read_csv(data_path, parse_dates=['transaction_date'])

    # Predict for the next week
    last_date = data['transaction_date'].max()
    next_week_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
    predictions = []

    for date in next_week_dates:
        week_day = date.weekday()
        month = date.month
        recent_data = data.tail(1).copy()  # Use the most recent data as a template
        recent_data['day_of_week'] = week_day
        recent_data['month'] = month
        recent_data['transaction_date'] = date
        recent_data['amount'] = 0  # Dummy amount

        # Feature engineering
        recent_data['amount_lag_1'] = data['amount'].shift(1).values[-1]
        recent_data['amount_lag_7'] = data['amount'].shift(7).values[-1]
        recent_data['amount_roll_mean_7'] = data['amount'].rolling(window=7).mean().values[-1]
        recent_data['transaction_count_last_7_days'] = data['amount'].resample('D').count().rolling(window=7).sum().shift(1).values[-1]
        recent_data['transaction_amount_mean_last_7_days'] = data['amount'].resample('D').sum().rolling(window=7).mean().shift(1).values[-1]
        recent_data['transaction_amount_std_last_7_days'] = data['amount'].resample('D').sum().rolling(window=7).std().shift(1).values[-1]
        recent_data.fillna(0, inplace=True)

        X_next = recent_data.drop(['transaction_id', 'user_id', 'transaction_date', 'amount'], axis=1)
        y_next_pred = model.predict(X_next)
        predictions.append((date, y_next_pred[0]))

    return predictions
