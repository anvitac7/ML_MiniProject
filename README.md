# Walmart Sales Forecasting using PatchTST

A machine learning project focused on forecasting weekly Walmart store sales using a transformer-based time-series model (PatchTST). The project demonstrates end-to-end model development, from data preprocessing to evaluation and visualization.

---

## 🧠 Project Summary

- Built a time-series forecasting pipeline for predicting weekly retail sales
- Implemented PatchTST, a transformer architecture designed for long-term forecasting
- Trained and evaluated models across multiple Walmart stores
- Visualized predicted vs actual sales to assess model performance
- Designed with modular, reusable code suitable for real-world ML workflows

---

## 🔧 Technical Highlights

- Time-series data preprocessing and normalization
- Sequence generation for supervised learning
- Transformer-based forecasting using PatchTST
- Model training, evaluation, and result visualization
- Clean project structure following ML engineering best practices

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, PyTorch / TensorFlow (as applicable), Matplotlib  
- **Domain:** Time-Series Forecasting, Deep Learning  

---

## 🗂️ Project Structure

```text
ML_MINIPROJECT/
├── app/
│   └── app.py                 # Entry point for running the application
│
├── data/
│   └── walmart.csv            # Walmart weekly sales dataset
│
├── results/
│   ├── forecast_vs_actual.png # Sales forecast vs actual visualization
│   └── training_loss.png      # Model training loss curve
│
├── src/
│   ├── models/
│   │   ├── lstm.py             # LSTM baseline model
│   │   └── patchtst.py         # PatchTST transformer model
│   │
│   ├── utils/
│   │   ├── data.py             # Data loading and preprocessing utilities
│   │   └── plots.py            # Visualization utilities
│   │
│   ├── main.py                 # End-to-end pipeline execution
│   └── train.py                # Model training logic
│
├── .gitignore
├── requirements.txt
└── README.md

```

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

## 📊 Results

- Generated weekly sales forecasts for multiple stores
- Compared predicted and actual sales trends using visual plots
- Evaluated forecasting accuracy using standard regression metrics

---

## 📌 Key Learnings

- Practical implementation of transformer models for time-series data
- Handling real-world retail sales datasets
- Structuring ML projects for scalability and readability
- Interpreting forecasting results for business insights

---

## 🙌 Author
Anvita Choudhary





