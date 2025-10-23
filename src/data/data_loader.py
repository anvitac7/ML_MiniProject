import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

def generate_sample_data(n_samples=2000):
    """Generate sample Walmart-like sales data"""
    np.random.seed(42)
    
    # Create realistic sales data with trend and seasonality
    t = np.arange(n_samples)
    trend = 0.1 * t
    seasonal = 100 * np.sin(2 * np.pi * t / 52)  # Weekly seasonality
    noise = np.random.normal(0, 50, n_samples)
    
    sales = 1000 + trend + seasonal + noise
    return sales.astype(np.float32)

def load_walmart_data(config):
    """Load Walmart data - replace with actual data loading"""
    try:
        # Try to load real data
        if os.path.exists(config.data_path):
            df = pd.read_csv(config.data_path)
            # Assuming sales data is in a column called 'sales'
            data = df['sales'].values.astype(np.float32)
            print(f"Loaded real data with {len(data)} samples")
        else:
            # Generate sample data
            data = generate_sample_data()
            print(f"Generated sample data with {len(data)} samples")
            
    except Exception as e:
        print(f"Error loading data: {e}. Using sample data.")
        data = generate_sample_data()
    
    return data.reshape(-1, 1)  # Reshape to [n_samples, 1]

def prepare_data(data, config):
    """Prepare data loaders"""
    # Split data
    n = len(data)
    train_size = int(n * config.train_ratio)
    val_size = int(n * config.val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, config.seq_len, config.pred_len)
    val_dataset = TimeSeriesDataset(val_data, config.seq_len, config.pred_len)
    test_dataset = TimeSeriesDataset(test_data, config.seq_len, config.pred_len)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader