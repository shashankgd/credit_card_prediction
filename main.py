import os
from ccp.src import feature_engineering, data_imputation, data_visualization, edge_cases, statistical_testing, train_model

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'ccp', 'data', 'synthetic_consumption_data.csv')
    model_path = os.path.join(base_path, 'ccp', 'data', 'best_model.pkl')
    training_results_path = os.path.join(base_path, 'ccp', 'data', 'training_results.png')
    testing_results_path = os.path.join(base_path, 'ccp', 'data', 'testing_results.png')
    metrics_path = os.path.join(base_path, 'ccp', 'data', 'model_performance_metrics.png')
    visualization_path = os.path.join(base_path, 'ccp', 'data', 'transaction_amount_distribution.png')
    next_week_predictions_path = os.path.join(base_path, 'ccp', 'data', 'next_week_predictions.png')

    # Perform feature engineering
    feature_engineering.perform_feature_engineering(data_path)

    # Handle data imputation
    data_imputation.handle_missing_values(data_path)

    # Visualize the data
    data_visualization.visualize_data(data_path, visualization_path)

    # Handle edge cases
    edge_cases.handle_edge_cases(data_path)

    # Perform statistical testing
    statistical_testing.perform_statistical_tests(data_path)

    # Train and test the model
    best_model, X_test, y_test = train_model.train_and_test_model(data_path, model_path, training_results_path, testing_results_path, metrics_path)

    # Predict next week and get actual values
    predictions, actual = train_model.predict_next_week(best_model, data_path)

    # Visualize next week predictions
    data_visualization.visualize_next_week_predictions.visualize_next_week_predictions(predictions, actual, next_week_predictions_path)

    # Print next week predictions
    train_model.print_predictions(predictions, actual)
