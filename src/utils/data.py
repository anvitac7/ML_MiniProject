import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class WalmartDataset:
    """
    Handles loading, preprocessing, and sequence generation
    for Walmart sales forecasting.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Load and sort Walmart sales data"""
        df = pd.read_csv(self.csv_path)
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
        return df

    def filter_store(self, df: pd.DataFrame, store_id: int) -> pd.DataFrame:
        """Filter data for a single store"""
        store_df = df[df["Store"] == store_id].copy()
        store_df.reset_index(drop=True, inplace=True)

        if store_df.empty:
            raise ValueError(f"No data found for store {store_id}")

        return store_df

    def create_sequences(
        self,
        sales: np.ndarray,
        seq_len: int,
        pred_len: int,
    ):
        """Create rolling input-output sequences"""
        X, y = [], []

        if len(sales) < seq_len + pred_len:
            raise ValueError(
                "Not enough data to create sequences "
                f"(required {seq_len + pred_len}, got {len(sales)})"
            )

        for i in range(len(sales) - seq_len - pred_len + 1):
            X.append(sales[i : i + seq_len])
            y.append(sales[i + seq_len : i + seq_len + pred_len])

        return np.array(X), np.array(y)

    def prepare_data(
        self,
        store_id: int,
        seq_len: int,
        pred_len: int,
        split_ratio: float = 0.8,
    ):
        """
        Full pipeline:
        - load data
        - filter store
        - scale sales
        - create train/validation splits
        """
        df = self.load_data()
        store_df = self.filter_store(df, store_id)

        sales = store_df["Weekly_Sales"].values.reshape(-1, 1)
        sales_scaled = self.scaler.fit_transform(sales).flatten()

        X, y = self.create_sequences(sales_scaled, seq_len, pred_len)

        split_idx = int(len(X) * split_ratio)

        return {
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_val": X[split_idx:],
            "y_val": y[split_idx:],
            "scaler": self.scaler,
            "raw_sales": sales.flatten(),
        }
