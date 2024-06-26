import pandas as pd

def handle_edge_cases(data):
    # Example: Remove transactions with negative amounts
    data = data[data['amount'] >= 0]
    return data
