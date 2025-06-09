# Stock Market Analysis and Forecasting using Deep Learning

A comprehensive stock market prediction project utilizing LSTM and GRU neural networks to forecast stock prices for major technology companies.

## 📊 Project Overview

This project performs stock market analysis and forecasting using deep learning techniques. The analysis focuses on predicting future stock prices based on historical data patterns, implementing both LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural networks.

## 🎯 Objectives

- Analyze historical stock price data for major tech companies
- Implement time series forecasting using deep learning models
- Compare performance between LSTM and GRU architectures
- Provide accurate stock price predictions with performance metrics

## 📈 Companies Analyzed

- **Amazon (AMZN)** - 2006-2018
- **IBM** - 2006-2018


## 🛠️ Technologies Used

### Programming Language
- Python 3.x

### Deep Learning Frameworks
- **TensorFlow/Keras** - For LSTM implementation
- **PyTorch** - For GRU implementation

### Data Analysis & Visualization
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive visualizations

### Machine Learning
- **Scikit-learn** - Data preprocessing and metrics
- **Statsmodels** - Time series decomposition

## 📁 Project Structure

```
├── asingh101_Project_LSTM_Stock_market.ipynb    # LSTM implementation
├── GRU_Stock_market.ipynb                       # GRU implementation
├── data/
│   ├── AMZN_2006-01-01_to_2018-01-01.csv
│   └── IBM_2006-01-01_to_2018-01-01.csv
└── README.md
```

## 🔍 Analysis Components

### 1. Exploratory Data Analysis (EDA)
- **Statistical Summary**: Mean, standard deviation, min/max values
- **Missing Value Analysis**: Data quality assessment
- **Correlation Analysis**: Relationship between Open, High, Low, Close prices
- **Trend Analysis**: Long-term price movements
- **Seasonality Detection**: Periodic patterns in stock prices

### 2. Data Visualization
- **Time Series Plots**: Stock price movements over time
- **Comparative Analysis**: Performance comparison between companies
- **Expanding Window Analysis**: Moving averages and standard deviations
- **Seasonal Decomposition**: Trend, seasonal, and residual components

### 3. Deep Learning Models

#### LSTM Model Architecture
- **Input Layer**: Time series sequences
- **LSTM Layers**: 3 layers with 50 units each
- **Dropout**: 0.2 for regularization
- **Dense Layer**: Final prediction layer
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error

#### GRU Model Architecture
- **Input Dimension**: 1 (Close price)
- **Hidden Dimension**: 32
- **Number of Layers**: 2
- **Output Dimension**: 1
- **Optimizer**: Adam (lr=0.01)
- **Loss Function**: MSE

## 📊 Model Performance

### Amazon Stock Prediction

| Model | Train RMSE | Test RMSE | Accuracy | Training Time |
|-------|------------|-----------|----------|---------------|
| LSTM  | 0.023      | 0.070     | 77.68%   | ~30s          |
| GRU   | 0.03       | 0.04      | 85.86%   | 10.82s        |

### IBM Stock Prediction

| Model | Train RMSE | Test RMSE | Accuracy | Training Time |
|-------|------------|-----------|----------|---------------|
| LSTM  | 0.054      | 0.051     | 56.40%   | ~40s          |
| GRU   | 3.30       | 2.86      | 98.12%   | 10.91s        |


## 📋 Data Preprocessing

1. **Data Loading**: CSV files with Date as index
2. **Normalization**: MinMaxScaler with range (-1, 1)
3. **Sequence Creation**: 20-day lookback windows
4. **Train-Test Split**: 80-20 split
5. **Data Shape**: 
   - Training: (2399, 19, 1)
   - Testing: (600, 19, 1)

## 🎯 Key Findings

### Market Insights
- **2009 Economic Impact**: All companies experienced significant losses during the 2008-2009 financial crisis
- **Amazon Growth**: Exponential growth pattern starting from 2012
- **IBM Performance**: More volatile with multiple peaks and valleys
- **Seasonal Patterns**: Strong seasonality detected in all stocks

### Model Comparison
- **GRU Advantages**: Faster training, better accuracy for Amazon predictions
- **LSTM Benefits**: More stable training, better for complex patterns
- **Training Efficiency**: GRU models train approximately 3x faster than LSTM

## 📊 Visualization Features

- **Interactive Plots**: Plotly-based interactive time series
- **Confusion Matrices**: Binary classification performance
- **Training Loss Curves**: Model convergence visualization
- **Prediction vs Actual**: Side-by-side comparison plots

## 🔮 Future Enhancements

- [ ] Add more companies and sectors
- [ ] Implement ensemble methods
- [ ] Include external factors (news sentiment, economic indicators)
- [ ] Real-time prediction capability
- [ ] Web application deployment
- [ ] Advanced feature engineering



## ⚠️ Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Past performance does not guarantee future results.
