import pandas as pd

def handle_edge_cases(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Handle edge cases (example: removing outliers)
    Q1 = data['amount'].quantile(0.25)
    Q3 = data['amount'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data['amount'] >= (Q1 - 1.5 * IQR)) & (data['amount'] <= (Q3 + 1.5 * IQR))]

    # Save the data
    data.to_csv(data_path, index=False)
    print("Edge cases handled.")
