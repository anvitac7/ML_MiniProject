import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class WalmartDataset:
    def __init__(self, file_path):
        self.csv_path = file_path
        self.scaler = StandardScaler()
        self.df = None 

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
        self.df = df
        return df

    def create_multivariate_sequences(self, data, seq_len, pred_len):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i : i + seq_len, :])
            # y is only the sales forecast (index 0)
            y.append(data[i + seq_len : i + seq_len + pred_len, 0]) 
        return np.array(X), np.array(y)

    def prepare_data(self, store_id, seq_len, pred_len, split_ratio=0.8):
        df = self.load_data() if self.df is None else self.df
        store_df = df[df["Store"] == store_id].copy().reset_index(drop=True)

        # Features used for AIML prediction
        features = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'Unemployment']
        data_values = store_df[features].values
        
        scaled_data = self.scaler.fit_transform(data_values)
        X, y = self.create_multivariate_sequences(scaled_data, seq_len, pred_len)

        split_idx = int(len(X) * split_ratio)
        return {
            "X_train": X[:split_idx], "y_train": y[:split_idx],
            "X_val": X[split_idx:], "y_val": y[split_idx:],
            "scaler": self.scaler, "feature_names": features,
            "raw_sales": data_values[:, 0]
        }