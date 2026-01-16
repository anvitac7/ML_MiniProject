import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from src.models.patchtst import PatchTST
from src.models.lstm import SimpleLSTM
from src.utils.data import WalmartDataset

def train_and_compare(store_id=1):
    dataset = WalmartDataset("data/walmart.csv")
    data = dataset.prepare_data(store_id, seq_len=52, pred_len=4)
    
    # Prepare Loaders
    train_loader = DataLoader(TensorDataset(torch.tensor(data["X_train"]).float(), 
                                            torch.tensor(data["y_train"]).float()), 
                              batch_size=16, shuffle=True)
    
    val_x = torch.tensor(data["X_val"]).float()
    val_y = torch.tensor(data["y_val"]).float()

    # Initialize both models
    models = {
        "PatchTST (Transformer)": PatchTST(52, 4, num_features=5),
        "SimpleLSTM (Baseline)": SimpleLSTM(input_dim=5, output_dim=4)
    }
    
    performance_table = []

    for name, model in models.items():
        print(f"🚀 Training {name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Short training loop for demonstration
        for epoch in range(30):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = model(val_x)
            mse = criterion(preds, val_y).item()
            mae = torch.mean(torch.abs(preds - val_y)).item()
            
        performance_table.append({"Model": name, "MSE": round(mse, 5), "MAE": round(mae, 5)})
        
        # Save weights for the dashboard
        filename = f"models/{name.split(' ')[0].lower()}_weights.pt"
        torch.save(model.state_dict(), filename)

    # 📊 Output the Comparative Table
    df = pd.DataFrame(performance_table)
    print("\n--- Comparative Metrics ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    train_and_compare()