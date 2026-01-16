import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """
    Standard LSTM Baseline to compare against the PatchTST Transformer.
    Input: (Batch, Sequence_Len, Features)
    Output: (Batch, Prediction_Len)
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=4):
        super().__init__()
        # input_dim=5 handles: Sales, Holiday, Temp, Fuel, Unemployment
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Linear layer maps the final hidden state to the 4-week prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # out shape: (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x)
        
        # We take the output of the very last time step (out[:, -1, :])
        # and pass it through the fully connected layer.
        return self.fc(out[:, -1, :])