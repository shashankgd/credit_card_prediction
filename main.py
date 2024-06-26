import os
from ccp.src import feature_engineering, data_imputation, data_visualization, edge_cases, statistical_testing, train_model

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'ccp', 'data', 'synthetic_consumption_data.csv')
    model_path = os.path.join(base_path, 'ccp', 'data', 'best_model.pkl')
    training_results_path = os.path.join(base_path, 'ccp', 'data', 'training_results.png')
    validation_results_path = os.path.join(base_path, 'ccp', 'data', 'validation_results.png')
    metrics_path = os.path.join(base_path, 'ccp', 'data', 'model_performance_metrics.png')
    next_week_predictions_path = os.path.join(base_path, 'ccp', 'data', 'next_week_predictions.png')

    # Prepare the data
    train_data, val_data, test_data = train_model.prepare_data(data_path)

    # Handle data imputation
    train_data = data_imputation.handle_missing_values(train_data)
    val_data = data_imputation.handle_missing_values(val_data)
    test_data = data_imputation.handle_missing_values(test_data)

    # Perform feature engineering on training, validation, and test data
    train_data = feature_engineering.perform_feature_engineering(train_data)
    val_data = feature_engineering.perform_feature_engineering(val_data)
    test_data = feature_engineering.perform_feature_engineering(test_data)

    # Handle edge cases
    train_data = edge_cases.handle_edge_cases(train_data)
    val_data = edge_cases.handle_edge_cases(val_data)
    test_data = edge_cases.handle_edge_cases(test_data)

    # Visualize the data
    data_visualization.visualize_data(train_data, os.path.join(base_path, 'ccp', 'data', 'train_data_visualization.png'))
    data_visualization.visualize_data(val_data, os.path.join(base_path, 'ccp', 'data', 'val_data_visualization.png'))
    data_visualization.visualize_data(test_data, os.path.join(base_path, 'ccp', 'data', 'test_data_visualization.png'))

    # Perform statistical testing
    statistical_testing.perform_statistical_tests(train_data)
    statistical_testing.perform_statistical_tests(val_data)
    statistical_testing.perform_statistical_tests(test_data)

    # Train and validate the model
    best_model = train_model.train_and_test_model(train_data, val_data, model_path, training_results_path, validation_results_path, metrics_path)

    # Predict next week and get actual values
    predictions, actuals = train_model.predict_next_week(best_model, test_data)

    # Visualize next week predictions
    train_model.visualize_next_week_predictions(predictions, actuals, next_week_predictions_path)

    # Print next week predictions
    train_model.print_predictions(predictions, actuals)
