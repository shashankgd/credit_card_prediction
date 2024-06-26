import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(output_path):
    num_users = 1000
    num_transactions = 40000
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    categories = ['Groceries', 'Entertainment', 'Utilities', 'Restaurants', 'Travel', 'Shopping']
    card_types = ['Credit', 'Debit']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    device_types = ['Mobile', 'Web', 'POS']

    data = []
    for _ in range(num_transactions):
        transaction_id = random.randint(1000000, 9999999)
        user_id = random.randint(1, num_users)
        transaction_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        amount = round(random.uniform(5.0, 500.0), 2)
        merchant_category = random.choice(categories)
        card_type = random.choice(card_types)
        location = random.choice(locations)
        device_type = random.choice(device_types)

        data.append([transaction_id, user_id, transaction_date, amount, merchant_category, card_type, location, device_type])

    df = pd.DataFrame(data, columns=['transaction_id', 'user_id', 'transaction_date', 'amount', 'merchant_category', 'card_type', 'location', 'device_type'])
    print(output_path)
    df.to_csv(output_path, index=False)
    print("Synthetic data generation complete.")


if __name__ == '__main__':
    generate_data('synthetic_consumption_data.csv')