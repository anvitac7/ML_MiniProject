import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_loss(train_losses, val_losses, save_path="results/training_loss.png"):
    Path("results").mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_forecast_vs_actual(actual, forecast, save_path="results/forecast_vs_actual.png"):
    Path("results").mkdir(exist_ok=True)

    history_len = len(actual)
    forecast_len = len(forecast)

    plt.figure(figsize=(10, 5))
    plt.plot(range(history_len), actual, label="Historical Sales")
    plt.plot(
        range(history_len, history_len + forecast_len),
        forecast,
        label="Forecast",
        marker="o",
    )

    plt.xlabel("Weeks")
    plt.ylabel("Sales")
    plt.title("Sales Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
