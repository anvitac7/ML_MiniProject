import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import os

# --- 1. Path Resolution ---
# Ensures 'src' is found even if running from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.patchtst import PatchTST
from src.models.lstm import SimpleLSTM
from src.utils.data import WalmartDataset

# Page Configuration
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Resource Loading ---
@st.cache_resource
def load_resources(store_id):
    dataset = WalmartDataset("data/walmart.csv")
    # Preparing data with 52 weeks history, 4 weeks prediction
    data = dataset.prepare_data(store_id, 52, 4)
    
    # Initialize Multivariate Models (num_features=5)
    p_model = PatchTST(seq_len=52, pred_len=4, num_features=5)
    l_model = SimpleLSTM(input_dim=5, output_dim=4) # Ensure LSTM also handles 5 features
    
    # Load weights if they exist (optional)
    # p_model.load_state_dict(torch.load("models/patchtst_multivariate.pt", map_location='cpu'))
    
    p_model.eval()
    l_model.eval()
    return dataset, data, p_model, l_model

# --- 3. Sidebar Inputs ---
st.sidebar.header("🕹️ Control Panel")
store_id = st.sidebar.number_input("Which Store are we analyzing?", 1, 45, 1)
st.sidebar.info(f"Analyzing Store #{store_id}. The AI uses 5 variables (Sales, Holiday, Temp, Fuel, Unemployment) to predict.")

dataset, data, p_model, l_model = load_resources(store_id)

# --- 4. Prediction Logic ---
# Correcting the 4-D Tensor error by ensuring shape is (1, 52, 5)
last_window = torch.tensor(data["X_val"][-1]).unsqueeze(0).float()

with torch.no_grad():
    y_patch_scaled = p_model(last_window).numpy().flatten()
    y_lstm_scaled = l_model(last_window).numpy().flatten()

# Inverse Scaling Helper for Multivariate Data
def rescale_sales(scaled_values, scaler):
    # Scaler expects 5 columns, we put Sales in the 1st column (index 0)
    dummy = np.zeros((len(scaled_values), 5))
    dummy[:, 0] = scaled_values 
    return scaler.inverse_transform(dummy)[:, 0]

# Actual sales (Target is already provided unscaled in our dict usually, or needs scaling)
# For this dashboard, we use the raw_sales for comparison
y_actual_raw = data["raw_sales"][-4:] # Last 4 known weeks
y_patch = rescale_sales(y_patch_scaled, data["scaler"])
y_lstm = rescale_sales(y_lstm_scaled, data["scaler"])

# --- 5. Executive Metrics ---
st.title("🏪 Walmart Sales Intelligence Dashboard")
st.subheader("📊 Performance at a Glance")
col1, col2, col3 = st.columns(3)

# Accuracy Calculation (1 - MAPE)
patch_acc = 100 - (np.mean(np.abs(y_actual_raw - y_patch) / y_actual_raw) * 100)
lstm_acc = 100 - (np.mean(np.abs(y_actual_raw - y_lstm) / y_actual_raw) * 100)

col1.metric("Transformer Accuracy", f"{patch_acc:.1f}%")
col2.metric("LSTM Accuracy", f"{lstm_acc:.1f}%")
col3.metric("Recommended Brain", "PatchTST" if patch_acc > lstm_acc else "LSTM", delta="Top Performer")

st.divider()

# --- 6. Visualization Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Sales Forecast", "🔎 Pattern Discovery", "❓ Help & Info"])

with tab1:
    st.subheader("Future Sales Prediction")
    
    fig = go.Figure()
    # Historical Context (Last 12 weeks)
    hist_sales = data["raw_sales"][-16:-4]
    weeks_hist = list(range(-12, 0))
    weeks_pred = list(range(0, 4))
    
    fig.add_trace(go.Scatter(x=weeks_hist, y=hist_sales, name="Past Sales", line=dict(color='#BDC3C7')))
    fig.add_trace(go.Scatter(x=weeks_pred, y=y_actual_raw, name="Actual Sales", line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=weeks_pred, y=y_patch, name="AI Prediction (PatchTST)", line=dict(color='#3498DB', dash='dash')))
    fig.add_trace(go.Scatter(x=weeks_pred, y=y_lstm, name="AI Prediction (LSTM)", line=dict(color='#E74C3C', dash='dot')))
    
    fig.update_layout(hovermode="x unified", xaxis_title="Weeks (0 = Prediction Start)", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("What drives your sales?")
    # Decompose raw sales to show Trend and Seasonality
    df_store = dataset.df[dataset.df['Store'] == store_id]
    res = seasonal_decompose(df_store['Weekly_Sales'], model='additive', period=52)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**The Trend:** Long-term growth or decline.")
        st.line_chart(res.trend)
    with c2:
        st.write("**The Seasonality:** Yearly recurring patterns.")
        st.line_chart(res.seasonal)

with tab3:
    st.info("""
    ### Technical Acknowledgments & Scope:
    * **Multi-Feature Input:** Unlike simple models, this AI considers Temperature, Fuel Prices, and Unemployment rates to predict demand.
    * **Robustness Check:** Use the sidebar to switch stores; this verifies the model works across different geographic regions.
    * **Evaluation:** Accuracy is calculated by comparing the AI's 4-week prediction against the real values from the validation set.
    """)