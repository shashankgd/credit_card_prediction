import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import datetime

def prepare_data(data_path):
    # Load the data with date parsing
    data = pd.read_csv(data_path, parse_dates=['transaction_date'])

    # Sort by date
    data.sort_values(by='transaction_date', inplace=True)

    # Set the date as the index
    data.set_index('transaction_date', inplace=True)

    # Split data
    train_data = data.iloc[:-14]  # All data except the last two weeks
    val_data = data.iloc[-14:-7]  # The second to last week
    test_data = data.iloc[-7:]    # The last week

    return train_data, val_data, test_data

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

def train_and_test_model(train_data, val_data, model_path, training_results_path, validation_results_path, metrics_path):
    # Perform feature engineering
    train_data = perform_feature_engineering(train_data)
    val_data = perform_feature_engineering(val_data)

    # Define features and target
    X_train = train_data.drop(['transaction_id', 'user_id', 'amount'], axis=1)
    y_train = train_data['amount']
    X_val = val_data.drop(['transaction_id', 'user_id', 'amount'], axis=1)
    y_val = val_data['amount']

    # Define the model
    model = lgb.LGBMRegressor()

    # Hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        'num_leaves': [31, 50, 70, 100],
        'learning_rate': [0.005, 0.003],
        'n_estimators': [500],
        'feature_fraction': [0.7, 0.8],
        'bagging_fraction': [0.7, 0.8],
        'bagging_freq': [0.5, 1]
    }

    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Best model
    best_model = random_search.best_estimator_

    # Save the model
    joblib.dump(best_model, model_path)

    # Training and validation
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    # Metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)

    # Print metrics
    print(f'Training MSE: {mse_train}, R2: {r2_train}')
    print(f'Validation MSE: {mse_val}, R2: {r2_val}')

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

    # Plot validation results
    plt.figure(figsize=(10, 5))
    plt.plot(y_val.values, label='Actual')
    plt.plot(y_val_pred, label='Predicted')
    plt.title('Validation Data: Actual vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Amount')
    plt.legend()
    plt.savefig(validation_results_path)
    plt.close()

    # Plot metrics
    plt.figure(figsize=(10, 5))
    metrics_data = {'Metric': ['MSE', 'R2'], 'Training': [mse_train, r2_train], 'Validation': [mse_val, r2_val]}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Metric', inplace=True)
    metrics_df.plot(kind='bar', title='Model Performance Metrics')
    plt.ylabel('Score')
    plt.savefig(metrics_path)
    plt.close()

    return best_model

def predict_next_week(model, test_data):
    # Perform feature engineering
    test_data = perform_feature_engineering(test_data)

    # Define features
    X_test = test_data.drop(['transaction_id', 'user_id', 'amount'], axis=1)

    # Predict
    y_test_pred = model.predict(X_test)

    # Extract actual amounts
    y_test_actual = test_data['amount'].values

    predictions = list(zip(X_test.index, y_test_pred))
    actuals = list(zip(X_test.index, y_test_actual))

    return predictions, actuals

def print_predictions(predictions, actuals):
    print("\nPredictions vs Actuals for the next week:")
    for (date, pred), (_, act) in zip(predictions, actuals):
        print(f"Date: {date.date()}, Predicted Amount: {pred:.2f}, Actual Amount: {act:.2f}")

def visualize_next_week_predictions(predictions, actuals, output_path):
    dates = [pred[0] for pred in predictions]
    predicted_values = [pred[1] for pred in predictions]
    actual_values = [act[1] for act in actuals]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, predicted_values, marker='o', linestyle='-', color='b', label='Predicted')
    plt.plot(dates, actual_values, marker='x', linestyle='-', color='r', label='Actual')
    plt.title('Predicted vs Actual Transaction Amounts for the Next Week')
    plt.xlabel('Date')
    plt.ylabel('Transaction Amount')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
