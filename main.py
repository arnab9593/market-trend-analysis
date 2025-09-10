import pandas as pd
import math
import ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas_datareader as pdr
import plotly.express as px
from datetime import datetime
from pandas_datareader.data import DataReader
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras.optimizers import Adam


sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

ticker_input = input("Enter tickers by comma separated: ")
ticker_list = [t.strip().upper() for t in ticker_input.split(',')]

name_input = input("Enter company names by comma separated: ")
org_names = [n.strip() for n in name_input.split(',')]
print(ticker_list, org_names)

end_date = datetime.now()
start_date = datetime(end_date.year-1, end_date.month, end_date.day)

company_data = []

for ticker, org in zip(ticker_list, org_names):
    df = yf.download(ticker, start=start_date, end=end_date)

    df["Company"] = org
    df["Ticker"] = ticker

    company_data.append(df)

df_all = pd.concat(company_data, axis=0)

df_all.to_csv("all_input_companies_df.csv", index=True)

for ticker, org, data in zip(ticker_list, org_names, company_data):
    filename = f"{ticker}_{org}.csv"
    data.to_csv(filename, index=True)

print(df_all.head())

for ticker, data in zip(ticker_list, company_data):
    print(ticker)
    print(data.describe())
    print(data.info())
    print("\n")

def make_grid(n, cols=2):
    cols = 1 if n == 1 else cols
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20 * cols, 15 * rows), squeeze=False)
    return fig, axes, rows, cols

# historical view of closing price
fig, axes, rows, cols = make_grid(len(company_data), cols=2)
for i, (ticker, data) in enumerate(zip(ticker_list, company_data)):
    ax = axes[i // cols][i % cols]
    data["Close"].plot(ax=ax)
    ax.set_title(f"Closing Price of {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")

# hide any unused axes
for j in range(i + 1, rows * cols):
    fig.delaxes(axes[j // cols][j % cols])

fig.tight_layout()
plt.show()

# daily return graph
fig, axes, rows, cols = make_grid(len(company_data), cols=2)
for i, (ticker, data) in enumerate(zip(ticker_list, company_data)):
    ax = axes[i // cols][i % cols]
    daily_ret = data["Close"].pct_change()
    ax.plot(daily_ret.index, daily_ret, alpha=0.7)
    ax.set_title(f"Daily Returns Over Time: {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
fig.tight_layout()
plt.show()

# closing price
df = yf.download(ticker_list, start=start_date, end=datetime.now())
# print(df)
plt.figure(figsize=(20,10))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# prediction
start_date = "2020-01-01"
end_date = datetime.now()
look_back = 90
forecast_horizon = 7

df_multi = yf.download(ticker_list, start=start_date, end=end_date)

all_results = []

for ticker in ticker_list:
    print(f"=== Processing {ticker} ===")

    # ----- isolate, ticker dataframe -----
    df = df_multi.loc[: ,(slice(None), ticker)]
    df.columns = df.columns.droplevel(1)
    df = df.dropna()

    # ----- features -----
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X_raw = df[features].values
    y_raw = df[['Close']].values

    # ----- scaling -----
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    # ----- sequences -----
    X, y = [], []
    for i in range(look_back, len(df)):
        X.append(X_scaled[i - look_back:i :])
        y.append(y_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    # ----- train split -----
    split = int(len(X) * 0.8)
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

    if len(X_valid) == 0:
        print(f"Skipping {ticker}, not enough data.")
        continue

    # ----- model -----
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stop],
        verbose=0
    )

    # ----- prediction -----
    y_pred_scaled = model.predict(X_valid, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_y.inverse_transform(y_valid.reshape(-1,1)).ravel()

    valid_index = df.index[look_back + split:]
    train_close = df['Close'].iloc[:look_back + split]

    valid_df = pd.DataFrame(
        {f'{ticker}_Actual': y_true, f'{ticker}_Predicted': y_pred},
        index=valid_index
    )
    all_results.append(valid_df)

    # ----- plot actual vs prediction -----
    plt.figure(figsize=(14,6))
    plt.plot(train_close, label='Train')
    plt.plot(valid_df[f'{ticker}_Actual'], label='Actual', color='green')
    plt.plot(valid_df[f'{ticker}_Predicted'], label='Prediction', color='red')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date'); plt.ylabel('Price USD')
    plt.legend(); plt.show()

    if all_results:
      final_results = pd.concat(all_results, axis=1)
      print("\n--- Final Results (Last 10 Days) ---")
      pd.set_option("display.max_columns", None)
      print(final_results.tail(10))
    else:
      print("\nNo results to display.")