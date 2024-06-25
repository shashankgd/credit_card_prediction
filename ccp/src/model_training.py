from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd

def train_model(df, target_column):
    # Convert object types to categorical and then to numerical codes
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.Categorical(df[col]).codes

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=3, alpha=10, n_estimators=50)
    model.fit(X_train, y_train)

    return model, X_test, y_test
