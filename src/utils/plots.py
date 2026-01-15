import matplotlib.pyplot as plt

def plot_training_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Model Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_forecast_vs_actual(actual, forecast, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual Sales")
    plt.plot(range(len(actual), len(actual) + len(forecast)), forecast, label="Predicted")
    plt.title("Walmart Sales Forecast")
    plt.legend()
    plt.savefig(save_path)
    plt.close()