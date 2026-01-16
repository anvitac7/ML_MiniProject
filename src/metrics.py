import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.patchtst import PatchTST
from src.models.lstm import SimpleLSTM
from src.utils.data import WalmartDataset

def run_benchmarks():
    dataset = WalmartDataset("data/walmart.csv")
    # Test across multiple stores to show robustness
    stores_to_test = [1, 15, 21, 33]
    all_metrics = []

    for store in stores_to_test:
        data = dataset.prepare_data(store, 52, 4)
        x_test = torch.tensor(data["X_val"]).float()
        y_true = data["y_val"] # Actual sales (scaled)

        # Load PatchTST
        p_model = PatchTST(52, 4, num_features=5)
        # (Load your weights here if you have them)
        
        # Load LSTM
        l_model = SimpleLSTM(input_dim=5, output_dim=4)

        with torch.no_grad():
            p_preds = p_model(x_test).numpy()
            l_preds = l_model(x_test).numpy()

        # Calculate Scores
        p_mae = mean_absolute_error(y_true, p_preds)
        l_mae = mean_absolute_error(y_true, l_preds)

        all_metrics.append({
            "Store": store,
            "PatchTST MAE": round(p_mae, 4),
            "LSTM MAE": round(l_mae, 4),
            "Improvement": f"{((l_mae - p_mae) / l_mae * 100):.1f}%"
        })

    # Save to Markdown for your README
    df = pd.DataFrame(all_metrics)
    df.to_markdown("RESULTS.md", index=False)
    print("✅ Performance Summary generated in RESULTS.md")
    print(df)

if __name__ == "__main__":
    run_benchmarks()