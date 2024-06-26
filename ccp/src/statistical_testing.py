import pandas as pd
import scipy.stats as stats

def perform_statistical_tests(data):
    # Check if there are enough samples for the normality test
    if len(data) < 8:
        print("Not enough samples for normality test")
        return

    # Perform a normality test on the 'amount' column
    k2, p = stats.normaltest(data['amount'])
    print(f"Normality test result: Statistic={k2}, p-value={p}")
