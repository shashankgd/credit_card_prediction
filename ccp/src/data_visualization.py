import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(data_path, visualization_path):
    # Load data
    data = pd.read_csv(data_path, parse_dates=['transaction_date'])

    # Visualization (example)
    plt.figure(figsize=(10, 5))
    data['amount'].hist(bins=50)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.savefig(visualization_path)
    # plt.show()
    print("Data visualization complete.")
    # plt.clo```````````````se()

def visualize_next_week_predictions(predictions, output_path):
    dates = [pred[0] for pred in predictions]
    values = [pred[1] for pred in predictions]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, values, marker='o', linestyle='-', color='b')
    plt.title('Predicted Transaction Amounts for the Next Week')
    plt.xlabel('Date')
    plt.ylabel('Predicted Amount')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()