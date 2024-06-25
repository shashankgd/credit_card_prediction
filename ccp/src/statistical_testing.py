import pandas as pd
from scipy.stats import normaltest

def perform_statistical_tests(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Statistical test (example: normality test)
    stat, p = normaltest(data['amount'])
    print(f'Normality test: stat={stat}, p={p}')
    if p > 0.05:
        print("Data follows a normal distribution")
    else:
        print("Data does not follow a normal distribution")
