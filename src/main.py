import argparse
import torch
import torch.nn as nn
from src.utils.data import WalmartDataset
from src.models.patchtst import PatchTST
from src.models.lstm import SimpleLSTM
from src.train import train_one_epoch, validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="patchtst", choices=["patchtst", "lstm"])
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WalmartDataset("data/walmart.csv")
    data = dataset.prepare_data(args.store, 52, 4)

    # Initialize selected model
    if args.model == "patchtst":
        model = PatchTST(52, 4).to(device)
    else:
        model = SimpleLSTM(output_dim=4).to(device)

    # Standard training boilerplate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training logic... (similar to your previous train.py)
    print(f"Started training {args.model} for Store {args.store}")
    # Save after training
    torch.save(model.state_dict(), f"models/{args.model}_store_{args.store}.pt")

if __name__ == "__main__":
    main()