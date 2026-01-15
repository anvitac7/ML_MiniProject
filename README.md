# Walmart Sales Forecasting using PatchTST

Predicting future weekly sales for Walmart stores using a transformer-based time series forecasting model.

This project implements a machine learning pipeline using the PatchTST architecture to forecast weekly sales across multiple Walmart stores. The goal is to analyze historical sales data and generate accurate future sales predictions.

---

## 🧠 Project Overview

- Forecasts weekly sales using time-series data
- Uses PatchTST (Transformer for Time Series)
- Compares actual vs predicted sales
- Designed as a mini project for academic and learning purposes

---

## 📦 Features

- Data preprocessing and normalization
- Transformer-based forecasting model
- Multi-store sales prediction
- Visualization of results
- Clean and modular code structure

---

## 🛠️ Requirements

- Python 3.8+
- Required libraries listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🗂️ Project Structure
ML_MiniProject/
│
├── data/                   # Dataset files
├── src/                    # Source code
│   ├── preprocessing.py    # Data cleaning and preparation
│   ├── model.py            # PatchTST model
│   ├── train.py            # Model training
│   └── evaluate.py         # Model evaluation
│
├── requirements.txt
└── README.md


---

## 🚀 How to Run
Clone the repository:
```bash
git clone https://github.com/anvitac7/ML_MiniProject.git
cd ML_MiniProject
```
Add the dataset to the data/ folder.
The dataset should include:
- Store ID
- Date
- Weekly sales values

Train the model:
```bash
python src/train.py
```

Evaluate results:
```bash
python src/evaluate.py
```
---

## 📊 Output

- Predicted vs actual sales plots
- Model evaluation metrics
- Store-level forecasting results

---

## 🔍 Model Used

- PatchTST (Patch Time Series Transformer)
- A transformer architecture designed for long-term time-series forecasting by learning temporal dependencies through attention mechanisms.

---

## 📝 Notes

- This project is intended for educational use.
- You can extend it by adding more features or trying different forecasting models.

---

## 🙌 Author
Anvita Choudhary


