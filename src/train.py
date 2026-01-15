import torch
import wandb
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Ensure y is (batch, pred_len)
        if y.dim() == 3:
            y = y.squeeze(-1)
            
        optimizer.zero_grad()
        preds = model(x) # Now returns (batch, pred_len)
        
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y.squeeze(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")