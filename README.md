# 🏪 Walmart Sales Forecasting with PatchTST

This project implements an AI-powered sales forecasting system for Walmart stores using **PatchTST (Patch Time Series Transformer)** — a modern transformer-based deep learning model designed for time series forecasting.  
It predicts weekly sales across multiple Walmart stores with **94.3% accuracy**, helping improve inventory, staffing, and business strategy.

<img width="3570" height="1166" alt="complete_forecasting_results" src="https://github.com/user-attachments/assets/0ae6d9cd-3fe3-49a9-87e4-eedaafc265c4" />

---

## 🎯 Key Features
- 🔮 **Accurate Predictions** – 12-week sales forecasts with 94.3% accuracy  
- 🏪 **Multi-Store Analysis** – Compare results for 13 different stores  
- 📈 **Business Insights** – Discover growth trends and seasonal patterns  
- 🤖 **Advanced AI** – Transformer-based PatchTST architecture  
- 📊 **Comprehensive Visualization** – Interactive charts and detailed reports  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- Basic knowledge of machine learning

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/anvitac7/ML_Miniproject.git
cd walmart-forecasting

# 2. Create a virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
Required Packages
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
🏃‍♂️ How to Run
Step 1: Prepare Your Data
Place your Walmart.csv file in the data/ directory.
The dataset should contain:
Store numbers (1–13)
Dates (weekly)
Weekly sales figures
Holiday flags
Economic indicators (temperature, fuel price, CPI, unemployment)

Step 2: Run the Forecasting Pipeline
python src/main.py
Step 3: View Results
Check the results/ directory for:
complete_forecasting_results.png – Forecast charts
comprehensive_forecasting_analysis.png – Detailed analysis
Training progress logs

```

## 📈 Sample Output
### Time Series Analysis for Store 1
- Time period: 2010-02-05 to 2012-10-26
- Total weeks: 143
- Average weekly sales: $1,555,264.40

### Forecasting Results
- Week 1:  $1,655,571.88
- Week 2:  $1,656,648.62
  ...
- Week 12: $1,693,464.12
  
- Average forecast: $1,652,784.38

## 📊 PERFORMANCE METRICS:
- Forecast vs Historical: +6.27%
- Accuracy: 94.3%
- Volatility Reduction: 72%

## 🧠 How It Works
#### 1. Data Processing
- Loads and cleans Walmart sales data
- Creates sequences of 52 weeks to predict the next 4 weeks
- Normalizes data for better training

#### 2. PatchTST Model
- Advanced transformer architecture
- self.encoder = nn.TransformerEncoder(
  d_model=64, nhead=8, 
  num_layers=3, batch_first=True)
- Patching: Breaks time series into meaningful segments
- Channel Independence: Handles each store separately
- Multi-Head Attention: Captures complex temporal patterns

#### 3. Training Process
- 100 epochs
- 80–20 train-validation split
- Mean Squared Error (MSE) loss function
- Adam optimizer with learning rate 0.001

#### 4. Forecasting
- Generates 12-week ahead predictions
- Provides confidence intervals
- Compares performance across multiple stores

## 💡 Business Applications
### 🏪 Store Managers
- Inventory Planning: Prepare for expected sales increases
- Staff Scheduling: Optimize workforce based on demand
- Promotion Planning: Time marketing campaigns with peak weeks

### 📊 Corporate Strategy
- Performance Comparison: Identify high-performing stores
- Risk Management: Understand sales volatility
- Growth Planning: Forecast future revenue trends

### 🎓 Academic Significance
- This project demonstrates:
- Real-world application of the PatchTST architecture
- Use of supervised learning for time-series forecasting
- Tangible business impact of AI in retail
- A scalable solution applicable across multiple locations

## 📊 Results Interpretation
### ✅ Green Flags
- Training loss decreases steadily
- Validation loss stable → no overfitting
- Forecasts close to historical averages
- Low volatility → confident predictions

### 🚩 Red Flags
- Validation loss increasing → possible overfitting
- High forecast volatility → uncertain predictions
- Large deviations from historical patterns → check data quality
