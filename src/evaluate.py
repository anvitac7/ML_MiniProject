import torch
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.patchtst import PatchTST
from src.utils.data import WalmartDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_robustness(store_ids=[1, 4, 10, 20]):
    dataset = WalmartDataset("data/Walmart.csv")
    results = []

    # Load the trained multivariate model weights
    model = PatchTST(seq_len=52, pred_len=4, num_features=5)
    model.load_state_dict(torch.load("models/patchtst_multivariate.pt", map_location='cpu'))
    model.eval()

    for store_id in store_ids:
        # Prepare test data for each store
        data = dataset.prepare_data(store_id, seq_len=52, pred_len=4)
        X_test = torch.tensor(data["X_val"]).float()
        y_test_raw = data["y_val"]

        with torch.no_grad():
            preds_scaled = model(X_test).numpy()
        
        # Rescale predictions to original dollar values
        # (Assuming index 0 is Weekly_Sales in your multivariate scaler)
        mae = mean_absolute_error(y_test_raw, preds_scaled)
        mse = mean_squared_error(y_test_raw, preds_scaled)
        
        results.append({
            "Store": store_id,
            "MAE_scaled": round(mae, 4),
            "MSE_scaled": round(mse, 4)
        })

    # Display as a table for your README
    df_results = pd.DataFrame(results)
    print("\n--- Model Performance Comparison ---")
    print(df_results.to_markdown(index=False))
    return df_results

if __name__ == "__main__":
    evaluate_robustness()