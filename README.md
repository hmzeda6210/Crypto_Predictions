# Bitcoin Price Prediction Projects

## Project 1: LSTM Price Prediction
**Notebook:** `LSTM_BTC_Final_Model.ipynb`

### Description
Uses Long Short-Term Memory (LSTM) neural network to predict future Bitcoin prices based on historical data.

### Optimal Configuration
- **Data**: 6 months of 1-hour intervals
- **Preprocessing**: 
  - Scaling
  - Reshaping 
  - Time-window creation
- **Task**: Forecast next price point
- **Visualizations**: Prediction vs actual price accuracy

### Purpose
Model temporal patterns in Bitcoin price trends using deep learning for short-term forecasting.

---

## Project 2: Price Movement Classification  
**Notebook:** `BTC_Classification.ipynb`

### Description
Classifies whether the next Bitcoin price move will be up/down using machine learning.

### Features
- **Binary classification** (Up/Down)
- **Models**: Random Forest
- **Feature Engineering**:
  - Price returns
  - Moving averages
  - Technical indicators (RSI)
- **Evaluation**:
  - Confusion matrix
  - Accuracy metrics

### Purpose
Classify short-term Bitcoin price direction using statistical features and supervised learning.

---

## Usage
1. Clone repository:
```bash
git clone https://github.com/hmzeda6210/Crypto_Predictions.git
