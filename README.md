# 🛒 Walmart Sales Forecasting with PatchTST & LSTM
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

An end-to-end Time-Series Forecasting pipeline comparing State-of-the-Art (SOTA) Transformers against traditional RNNs. This project features a professional ML engineering workflow, automated experiment tracking, and an interactive business intelligence dashboard.

> **TL;DR:** Comparing PatchTST vs LSTM for Walmart sales forecasting across 45 stores. LSTM wins on small datasets (<100 weeks), PatchTST excels with more data. Includes interactive dashboard with trend decomposition.

---

## 🚀 Project Overview

The goal of this project is to predict weekly sales across different Walmart departments. We specifically evaluate whether PatchTST (Patch Time Series Transformer) outperforms a baseline LSTM in a retail context.

Key Features:
- Patching Mechanism: Implements PatchTST to handle long-term dependencies by grouping time steps into patches.
- Comparative Analysis: Side-by-side performance metrics for 4 representative stores.
- Interactive UI: A Streamlit app that allows users to select stores and visualize forecasts.

---

## 🖥️ Interactive Dashboard
![Dashboard Overview](results/dashboard_main.png)
*Business Intelligence interface with real-time model comparison and performance metrics*

---

## 📊 Dashboard Capabilities
### Performance Monitoring
- Live accuracy tracking: PatchTST (93.8%) vs LSTM (95.9%)
- Smart recommendation engine highlighting top-performing model per store
- Store-specific analysis across 45 Walmart locations

### Sales Forecasting
- 4-week ahead predictions with dual-model comparison
- Visual separation of historical data and future forecasts
- Confidence tracking through accuracy metrics

![Pattern Analysis](results/dashboard_pattern.png)
*Automated trend and seasonality decomposition for explainable forecasting*

### Pattern Discovery
- Trend Analysis: Identifies long-term growth or decline patterns
- Seasonality Detection: Reveals recurring weekly/monthly cycles
- Business Insights: Translates complex ML outputs into actionable retail intelligence

---

## 📊 Performance & Comparative Results

In the initial results, PatchTST underperformed the LSTM on 2 out of 4 stores (Store [1] and Store [15]).
Why did PatchTST underperform?
- Data Volume: PatchTST is a high-capacity model that thrives on large-scale data. For individual stores with limited history, the LSTM’s simpler architecture acted as a regularizer, preventing the overfitting seen in the Transformer model.
- Hyperparameter Sensitivity: Current results suggest the PatchTST requires more granular tuning of patch length and stride for smaller datasets.

```text
|   Store |   PatchTST MAE |   LSTM MAE | Improvement   |
|--------:|---------------:|-----------:|:--------------|
|       1 |         0.4743 |     0.4431 | -7.1%         |
|      15 |         0.445  |     0.346  | -28.6%        |
|      21 |         0.447  |     0.5152 | 13.2%         |
|      33 |         0.6325 |     0.6584 | 3.9%          |
```

---

## 📦 Dataset Details
- **Source:** Walmart Store Sales Forecasting (Kaggle)
- **Time Range:** Feb 2010 - Oct 2012
- **Granularity:** Weekly aggregated sales
- **Features:** 5 (Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, Unemployment)
- **Stores:** 45 locations across different regions

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
- **Seasonal Decomposition:** Integrated statsmodels to extract underlying trends from noisy retail data, providing an "Explainable AI" layer for store managers.

---

## 🗂️ Project Structure

```text
ML_MINIPROJECT/
├── app/
│   └── app.py                    # Interactive Streamlit dashboard
├── data/
│   └── walmart.csv               # Walmart weekly sales dataset
├── results/
│   ├── dashboard_main.png        # Dashboard overview screenshot
│   ├── dashboard_patterns.png    # Pattern discovery screenshot
│   ├── forecast_vs_actual.png    # Visual validation of predictions
│   └── training_loss.png         # Training/Validation convergence curves
├── src/
│   ├── models/
│   │   ├── lstm.py               # Baseline LSTM implementation
│   │   └── patchtst.py           # SOTA PatchTST Transformer model
│   ├── utils/
│   │   ├── data.py               # Data loaders & scaling logic
│   │   └── plots.py              # Visualization utilities
│   ├── evaluate.py               # Multi-store benchmarking script
│   ├── main.py                   # CLI entry point for training
│   └── train.py                  # Modular training & W&B logging
├── requirements.txt              # Reproducible environment
├── README.md                     # Project documentation
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

## ⚠️ Known Limitations
- Models trained on individual stores (not global)
- No handling of extreme outliers (e.g., Black Friday spikes)
- Requires minimum 52 weeks of historical data
- Streamlit app runs locally only (not production-ready)

---

## 🧪 Scalability & Future Scope
- Global Modeling: Currently, models are trained per-store. Moving to a Global Model (training on all 45 stores simultaneously) would allow PatchTST to leverage cross-series information.
- Statistical Significance: Future iterations will include Diebold-Mariano tests to confirm if PatchTST improvements are statistically significant.
- Deployment: The app is designed to be containerized using Docker and can be deployed via AWS App Runner or Streamlit Cloud.

---
## 🙌 Author
Anvita Choudhary _ML Engineering Project | Time-Series Focus_
