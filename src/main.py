import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import numpy as np

from utils.data import WalmartDataset
from models.patchtst import PatchTST
from utils.plots import plot_training_loss, plot_forecast_vs_actual

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            total_loss += criterion(preds, y).item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/walmart.csv")
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=52)
    parser.add_argument("--pred_len", type=int, default=4)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WalmartDataset(args.data_path)
    data = dataset.prepare_data(args.store, args.seq_len, args.pred_len)

    X_train = torch.tensor(data["X_train"]).unsqueeze(-1).float()
    y_train = torch.tensor(data["y_train"]).unsqueeze(-1).float()
    X_val = torch.tensor(data["X_val"]).unsqueeze(-1).float()
    y_val = torch.tensor(data["y_val"]).unsqueeze(-1).float()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=32, shuffle=False
    )

    model = PatchTST(args.seq_len, args.pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    plot_training_loss(
        train_losses, val_losses, "results/training_loss.png"
    )

    last_seq = X_val[-1].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        forecast_scaled = model(last_seq).cpu().numpy().flatten()

    forecast = data["scaler"].inverse_transform(
        forecast_scaled.reshape(-1, 1)
    ).flatten()

    actual = data["raw_sales"][-args.seq_len :]

    plot_forecast_vs_actual(actual, forecast, "results/forecast_vs_actual.png")

    print("Training completed. Results saved to results/ folder.")


if __name__ == "__main__":
    main()
