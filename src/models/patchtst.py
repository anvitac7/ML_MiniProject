import torch
import torch.nn as nn

class SimplePatchTST(nn.Module):
    def __init__(self, config, n_channels):
        super().__init__()
        self.config = config
        self.n_channels = n_channels
        
        # Calculate patches
        self.num_patches = (config.seq_len - config.patch_len) // config.stride + 1
        
        print(f"Model Configuration:")
        print(f"  - Sequence length: {config.seq_len}")
        print(f"  - Prediction length: {config.pred_len}")
        print(f"  - Number of patches: {self.num_patches}")
        print(f"  - Number of channels: {n_channels}")
        
        # Patch embedding
        self.patch_embed = nn.Linear(config.patch_len * n_channels, config.d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, config.d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(config.d_model * self.num_patches, config.pred_len * n_channels)
        
    def forward(self, x):
        batch_size, seq_len, n_channels = x.shape
        
        # Create patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :]
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len, n_channels]
        patches_flat = patches.reshape(batch_size, self.num_patches, -1)
        
        # Embed patches
        embedded = self.patch_embed(patches_flat)
        embedded = embedded + self.pos_embed
        
        # Transformer
        encoded = self.transformer(embedded)
        encoded_flat = encoded.reshape(batch_size, -1)
        
        # Output
        output = self.output_layer(encoded_flat)
        output = output.reshape(batch_size, self.config.pred_len, self.n_channels)
        
        return output