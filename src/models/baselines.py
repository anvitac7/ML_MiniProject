import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=96):
        super(LSTMBaseline, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

class LinearBaseline(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearBaseline, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x_flat = x.reshape(x.size(0), -1)
        return self.linear(x_flat)

class CNNBaseline(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNBaseline, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)