import pandas as pd

def handle_missing_values(data_path):
    # Load data
    data = pd.read_csv(data_path, parse_dates=['transaction_date'])

    # Debug: Print data types before imputation
    print("Data types before imputation:\n", data.dtypes)

    # Handle missing values for numeric columns only
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Debug: Print data types after imputation
    print("Data types after imputation:\n", data.dtypes)

    # Save the data
    data.to_csv(data_path, index=False)
    print("Data imputation complete.")
