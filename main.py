import pandas as pd
import argparse
from ccp.src.data_preprocessing import load_data, preprocess_data
from ccp.src.feature_engineering import create_features, encode_categorical
from ccp.src.model_training import train_model
from ccp.src.model_evaluation import evaluate_model

def main():
    # Argument parser for configurable prediction window
    parser = argparse.ArgumentParser(description='Credit Card Consumption Prediction')
    parser.add_argument('--window', type=str, default='week', choices=['day', 'week', 'month'],
                        help='Prediction window: day, week, or month')
    args = parser.parse_args()
    window = args.window

    # Load and preprocess data
    data_path = 'ccp/data/credit_data.csv'
    df = load_data(data_path)
    df = preprocess_data(df)

    # Feature engineering
    df = create_features(df)
    df = encode_categorical(df)

    # Adjust target based on prediction window
    target_column = 'cc_cons'
    if window == 'day':
        df['target'] = df[target_column].shift(-1)
    elif window == 'week':
        df['target'] = df[target_column].shift(-7)
    elif window == 'month':
        df['target'] = df[target_column].shift(-30)

    df.dropna(subset=['target'], inplace=True)

    # Train model
    model, X_test, y_test = train_model(df, 'target')

    # Evaluate model
    rmse, mae = evaluate_model(model, X_test, y_test)

    # Print evaluation metrics
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')

if __name__ == '__main__':
    main()
