import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()

        self.input_proj = nn.Linear(1, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = self.input_proj(x)
        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = self.temporal_proj(x)
        x = x.transpose(1, 2)

        return self.output_proj(x)
