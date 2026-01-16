# Walmart Sales Forecasting with PatchTST & LSTM

An end-to-end Time-Series Forecasting pipeline comparing State-of-the-Art (SOTA) Transformers against traditional RNNs. This project features a professional ML engineering workflow, automated experiment tracking, and an interactive business intelligence dashboard.

---

## 🧠 Project Summary

- **Multivariate Forecasting:** Unlike univariate models, this pipeline integrates 5 key economic features: Weekly Sales, Holiday Flags, Temperature, Fuel Price, and Unemployment to predict future demand.
- **Advanced Modeling:** Implemented a custom PatchTST (Patch Time Series Transformer). By grouping time steps into sub-series "patches," the model captures local semantic patterns more effectively than point-wise transformers.
- **Comparative Analysis:** Includes a Simple LSTM baseline to benchmark and quantify the performance gains of transformer-based architectures.
- **Interactive Interface:** A Streamlit dashboard for real-time forecasting, allowing managers to explore seasonal trends and model accuracy across 45 different stores.

---

## 📊 Performance & Comparative Results

To ensure transparency and address model robustness, the models were benchmarked across multiple stores using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
```text
|   Store |   PatchTST MAE |   LSTM MAE | Improvement   |
|--------:|---------------:|-----------:|:--------------|
|       1 |         0.4743 |     0.4431 | -7.1%         |
|      15 |         0.445  |     0.346  | -28.6%        |
|      21 |         0.447  |     0.5152 | 13.2%         |
|      33 |         0.6325 |     0.6584 | 3.9%          |
```
Metrics calculated on a 4-week forecast horizon using a 52-week historical look-back window.

---

## 🏗️ Architectural Deep-Dive

PatchTST Implementation
The core innovation in this project is the Patching Layer. Instead of treating each week as an isolated token, the model:
- Patches: Groups the input into overlapping 16-week windows.
- Projects: Maps these patches into a 128-dimensional latent space.
- Attends: Uses Multi-Head Self-Attention to find correlations between different times of the year (e.g., how "Temperature" in Week 10 impacts "Sales" in Week 40).

---

## 🔧 Technical Highlights

- **Multivariate Scaling:** Implemented a StandardScaler pipeline that handles multiple features with different units (e.g., Temperature vs. Millions of dollars in Sales) to ensure model convergence.
- **3D Tensor Engineering:** Designed a data loader that transforms raw CSV data into (Batch, Sequence, Features) tensors, compatible with high-performance Transformer encoders.
- **Seasonal Decomposition:** Integrated statsmodels to extract underlying trends from noisy retail data, providing an "Explainable AI" layer for store managers.ness.

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

2. Generate Performance Metrics To reproduce the comparative table and verify model robustness:
```bash
python -m src.evaluate
```

3. Launch Interactive Dashboard
```bash
streamlit run app/app.py
```

---

## 🙌 Author
Anvita Choudhary ML Engineering Project | Time-Series Focus








