import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class WalmartDataset:
    """
    Handles loading, preprocessing, and sequence generation
    for Walmart sales forecasting.
    """

    def __init__(self, file_path):
        self.csv_path = file_path
        self.scaler = StandardScaler()
        # Initializing df as None so it can be loaded on demand
        self.df = None 

    def load_data(self) -> pd.DataFrame:
        """Load and sort Walmart sales data with correct date parsing."""
        df = pd.read_csv(self.csv_path)
        # Using dayfirst=True to handle the '19-02-2010' format error
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
        self.df = df
        return df

    def filter_store(self, df: pd.DataFrame, store_id: int) -> pd.DataFrame:
        """Filter data for a single store and handle empty results."""
        store_df = df[df["Store"] == store_id].copy()
        if store_df.empty:
            raise ValueError(f"No data found for store {store_id}")
        return store_df.reset_index(drop=True)

    def create_sequences(self, sales: np.ndarray, seq_len: int, pred_len: int):
        """
        Create rolling input-output sequences.
        Example: If seq_len=52 and pred_len=4, it uses 52 weeks to predict the next 4.
        """
        X, y = [], []
        if len(sales) < seq_len + pred_len:
            raise ValueError(
                f"Not enough data to create sequences (required {seq_len + pred_len}, got {len(sales)})"
            )

        for i in range(len(sales) - seq_len - pred_len + 1):
            X.append(sales[i : i + seq_len])
            y.append(sales[i + seq_len : i + seq_len + pred_len])

        return np.array(X), np.array(y)

    def prepare_data(self, store_id: int, seq_len: int, pred_len: int, split_ratio: float = 0.8):
        """
        Main pipeline: Load -> Filter -> Scale -> Sequence -> Split.
        """
        # 1. Load data if not already loaded
        df = self.load_data() if self.df is None else self.df
        
        # 2. Filter for specific store
        store_df = self.filter_store(df, store_id)

        # 3. Scale sales data
        sales = store_df["Weekly_Sales"].values.reshape(-1, 1)
        sales_scaled = self.scaler.fit_transform(sales).flatten()

        # 4. Create time-series windows
        X, y = self.create_sequences(sales_scaled, seq_len, pred_len)

        # 5. Temporal split (Train/Val)
        split_idx = int(len(X) * split_ratio)

        return {
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_val": X[split_idx:],
            "y_val": y[split_idx:],
            "scaler": self.scaler,
            "raw_sales": sales.flatten(),
        }