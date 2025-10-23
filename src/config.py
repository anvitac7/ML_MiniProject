class Config:
    def __init__(self):
        # Data configuration
        self.data_path = "data/raw/walmart_sales.csv"  # Update with your actual file path
        
        # Time series configuration
        self.seq_len = 52      # Lookback window (1 year of weekly data)
        self.pred_len = 4      # Prediction length (4 weeks ahead)
        
        # PatchTST configuration
        self.patch_len = 8
        self.stride = 4
        self.d_model = 64
        self.n_heads = 4
        self.n_layers = 2
        self.d_ff = 128
        self.dropout = 0.1
        
        # Training configuration
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.patience = 10
        
        # Data split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        
        # Device
        self.device = "cuda"  # Will be set to CPU if CUDA not available