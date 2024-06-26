import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(data, output_path):
    plt.figure(figsize=(10, 5))
    data['amount'].hist(bins=50)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()
