def data_summary(df):
    print("Data Summary:")
    print(df.describe())

def missing_values(df):
    print("Missing Values:")
    print(df.isnull().sum())

