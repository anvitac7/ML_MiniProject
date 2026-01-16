import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class WalmartDataset:
    def __init__(self, file_path):
        self.csv_path = file_path
        self.scaler = StandardScaler()
        self.df = None 

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
        self.df = df
        return df

    def filter_store(self, df: pd.DataFrame, store_id: int) -> pd.DataFrame:
        store_df = df[df["Store"] == store_id].copy()
        if store_df.empty:
            raise ValueError(f"No data found for store {store_id}")
        return store_df.reset_index(drop=True)

    def create_sequences(self, data_array: np.ndarray, seq_len: int, pred_len: int):
        """
        X will contain all 5 features (multivariate).
        y will contain only 'Weekly_Sales' (the target).
        """
        X, y = [], []
        for i in range(len(data_array) - seq_len - pred_len + 1):
            # Grab all 5 columns for the input window
            X.append(data_array[i : i + seq_len, :])
            # Grab only the 1st column (Weekly_Sales) for the target
            y.append(data_array[i + seq_len : i + seq_len + pred_len, 0])

        return np.array(X), np.array(y)

    def prepare_data(self, store_id: int, seq_len: int, pred_len: int, split_ratio: float = 0.8):
        df = self.load_data() if self.df is None else self.df
        store_df = self.filter_store(df, store_id)

        # 1. Define Multivariate Features
        features = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'Unemployment']
        data_values = store_df[features].values 
        
        # 2. Scale all 5 features together
        scaled_data = self.scaler.fit_transform(data_values)

        # 3. Create windows (X is now 3D: Batch, Seq, Features)
        X, y = self.create_sequences(scaled_data, seq_len, pred_len)

        # 4. Split
        split_idx = int(len(X) * split_ratio)

        return {
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_val": X[split_idx:],
            "y_val": y[split_idx:],
            "scaler": self.scaler,
            "feature_names": features,
            "raw_sales": store_df["Weekly_Sales"].values
        }