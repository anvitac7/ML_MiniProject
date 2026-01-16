import torch
import torch.nn as nn

# REMOVE: from src.models.patchtst import PatchTST <--- Delete this line!

class PatchTST(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int = 1):
        super().__init__()
        
        # Linear projection of input features
        self.input_proj = nn.Linear(num_features, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=8, 
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        x = self.input_proj(x) 
        x = self.encoder(x) 
        
        x = x.transpose(1, 2)   # (batch, 64, seq_len)
        x = self.temporal_proj(x) # (batch, 64, pred_len)
        x = x.transpose(1, 2)   # (batch, pred_len, 64)
        
        x = self.output_proj(x) 
        return x.squeeze(-1)    # (batch, pred_len)