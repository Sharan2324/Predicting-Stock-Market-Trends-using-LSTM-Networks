# Predicting-Stock-Market-Trends-using-LSTM-Networks
#📌 Overview

This project implements a Bidirectional Long Short-Term Memory (BiLSTM) model to predict stock prices in the Indian stock market (NSE). The model is trained using historical stock data and aims to forecast future stock prices based on past trends. The project provides data preprocessing, model training, evaluation, and visualization to analyze stock price movements.

#📊 Dataset

Stock Data Source: Yahoo Finance (yfinance API) 
Stock Used: Reliance Industries Limited (NSE: RELIANCE) 
Data Features: Open, High, Low, Close, Volume 
Training Data: 80% of historical stock data 
Testing Data: 20% of historical stock data 

#🔥 Technologies Used

Python  
TensorFlow/Keras (Deep Learning Model)  
pandas & numpy (Data Preprocessing)  
matplotlib & seaborn (Data Visualization)  
yfinance (Fetching Stock Data)  
mcal (Market Calendar for NSE Trading Days)  

#🚀 Project Workflow

Data Collection: Fetch historical stock data from Yahoo Finance.
Data Preprocessing: Normalize data, create sequences, and handle missing values.
Model Architecture: Implement a Bidirectional LSTM model with dropout layers for regularization.
Training the Model: Train using past stock data, optimize hyperparameters, and monitor performance.
Evaluation: Measure accuracy using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
Prediction: Forecast the next trading day’s closing price.
Visualization: Compare actual vs. predicted prices.

#📈 Results

Prediction Accuracy:
Mean Squared Error (MSE): 1401.4399
Root Mean Squared Error (RMSE): 37.4358
Mean Absolute Percentage Error (MAPE): 0.03%
Predicted Closing Price for Next Trading Day: {predicted_next_price:.2f} INR

#📌 Observations:

✔ The model effectively captures stock price trends but struggles with high-volatility periods.
✔ Future enhancements include news sentiment analysis, additional technical indicators, and hybrid deep learning architectures.

#🔍 Future Improvements

✅ Feature Engineering: Add RSI, moving averages, and Bollinger Bands.
✅ News Sentiment Analysis: Integrate financial news to enhance predictions.
✅ Hyperparameter Optimization: Fine-tune model parameters for improved accuracy.
✅ Exploring Transformer Models: Compare LSTMs with modern deep learning approaches.
