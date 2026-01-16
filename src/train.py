import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.patchtst import PatchTST
from src.utils.data import WalmartDataset

def train_multivariate():
    dataset = WalmartDataset("data/Walmart.csv")
    data = dataset.prepare_data(store_id=1, seq_len=52, pred_len=4) #
    
    train_x = torch.tensor(data["X_train"]).float()
    train_y = torch.tensor(data["y_train"]).float()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)

    model = PatchTST(seq_len=52, pred_len=4, num_features=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} Complete")

    torch.save(model.state_dict(), "models/patchtst_multivariate.pt")
    print("✅ Multivariate weights saved.")

if __name__ == "__main__":
    train_multivariate()