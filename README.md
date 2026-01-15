# Walmart Sales Forecasting with PatchTST & LSTM

An end-to-end Time-Series Forecasting pipeline comparing State-of-the-Art (SOTA) Transformers against traditional RNNs. This project features a professional ML engineering workflow, automated experiment tracking, and an interactive business intelligence dashboard.

---

## 🧠 Project Summary

- **Advanced Modeling:** Implemented PatchTST, a transformer architecture that uses sub-series patching to capture local semantic patterns more effectively than point-wise models.
- **Comparative Analysis:** Includes a Simple LSTM baseline to benchmark the performance gains of transformer-based forecasting.
- **Interactive Interface:** Developed a Streamlit dashboard for real-time forecasting, allowing users to simulate "What-If" scenarios and explore seasonal trends.
- **MLOps Integration:** Integrated with Weights & Biases (W&B) for experiment tracking and hyperparameter logging.

---

## 🔧 Technical Highlights

- **Custom Data Pipeline:** Robust date parsing (DD-MM-YYYY) and sliding-window sequence generation (52-week history to predict 4-week future).
- **Seasonal Decomposition:** Automated extraction of Trend and Seasonality using statsmodels to provide deeper business insights.
- **Scalable Structure:** Decoupled model logic (src/models) from data utilities (src/utils) to follow industry clean-code standards.
- **Error Diagnostics:** Built-in residual analysis to identify model bias and prediction skewness.

---

## 🗂️ Project Structure

```text
ML_MINIPROJECT/
├── app/
│   └── app.py                # Interactive Streamlit dashboard
├── data/
│   └── walmart.csv           # Walmart weekly sales dataset
├── results/
│   ├── forecast_vs_actual.png # Visual validation of predictions
│   └── training_loss.png      # Training/Validation convergence curves
├── src/
│   ├── models/
│   │   ├── lstm.py           # Baseline LSTM implementation
│   │   └── patchtst.py       # SOTA PatchTST Transformer model
│   ├── utils/
│   │   ├── data.py           # Data loaders & scaling logic (fixed date parsing)
│   │   └── plots.py          # Visualization utilities
│   ├── main.py               # CLI Entry point for training models
│   └── train.py              # Modular training & W&B logging logic
├── requirements.txt          # Reproducible environment
└── README.md
```

---

## 🚀 How to Run
1. Setup Environment
```bash
git clone https://github.com/anvitac7/ML_MiniProject.git
cd ML_MiniProject
pip install -r requirements.txt
```

2. Train Models (CLI):
You can train either model for a specific store via the terminal:
```bash
# Train the Transformer
python -m src.main --model patchtst --store 1 --epochs 50

# Train the LSTM baseline
python -m src.main --model lstm --store 1 --epochs 50
```

3. Launch Interactive Dashboard
```bash
streamlit run app/app.py
```

---

## 📊 Results

- PatchTST Performance: Demonstrates superior capture of holiday-driven sales spikes (seasonality) compared to standard RNNs.
- Interpretability: Dashboard metrics show Accuracy % and MSE/MAE side-by-side for executive-level decision making.

---

## 🙌 Author
Anvita Choudhary







