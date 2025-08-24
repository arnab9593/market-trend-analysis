import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math

# download data 
def download_data(tickers, start, end):
    data_dict = {ticker: yf.download(ticker, start, end) for ticker in tickers}
    return data_dict

# plot historical closing prices 
def plot_closing_prices(data_dict):
    n = len(data_dict)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5*rows))

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, (ticker, df) in zip(axes, data_dict.items()):
        df['Close'].plot(ax=ax, title=f"Closing Price of {ticker}")
        ax.set_ylabel("Close")

    for ax in axes[len(data_dict):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# add moving averages
def add_moving_averages(data_dict, ma_days=[10,20,50]):
    for ma in ma_days:
        for ticker, df in data_dict.items():
            df[f"MA_{ma}"] = df['Close'].rolling(ma).mean()
    return data_dict

# prepare training data
def prepare_data(ticker, start="2012-01-01"):
    df = yf.download(ticker, start=start, end=datetime.now())
    data = df[['Close']]
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.95))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return df, dataset, training_data_len, scaler, x_train, y_train

# build lstm 
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# plot results
def predict_and_plot(df, dataset, training_data_len, scaler, model, ticker):
    test_data = scaler.transform(dataset[training_data_len-60:])
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test).reshape(-1, 60, 1)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = df[:training_data_len]
    valid = df[training_data_len:].copy()
    valid['Predictions'] = predictions

    plt.figure(figsize=(16,6))
    plt.title(f'{ticker} Prediction')
    plt.plot(train['Close'], label='Train')
    plt.plot(valid['Close'], label='Validation')
    plt.plot(valid['Predictions'], label='Predictions')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    tickers_input = input("Enter stock tickers separated by commas")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    data_dict = download_data(tickers, start, end)

    plot_closing_prices(data_dict)

    data_dict = add_moving_averages(data_dict)

    for ticker in tickers:
        print(f"\nRunning prediction model for {ticker}...")
        df, dataset, training_data_len, scaler, x_train, y_train = prepare_data(ticker)

        model = build_lstm((x_train.shape[1], 1))
        model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0)

        predict_and_plot(df, dataset, training_data_len, scaler, model, ticker)
