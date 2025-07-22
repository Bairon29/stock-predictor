import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# 1. Download AAPL & VIX
ticker = 'AAPL'
df = yf.download(ticker, '2020-01-01', '2024-01-01')
vix = yf.download('^VIX', '2020-01-01', '2024-01-01')['Close']

# 2. Compute Indicators
df['RSI'] = RSIIndicator(df['Close'].squeeze()).rsi()
macd = MACD(df['Close'].squeeze())
df['MACD'] = macd.macd()
df['Signal'] = macd.macd_signal()
df['MA_20'] = df['Close'].rolling(20).mean()
df['Volume'] = df['Volume']
df['VIX'] = vix.reindex(df.index)

# 3. Compute Put/Call ratio (by volume)
opt = yf.Ticker(ticker).options
option_chain = yf.Ticker(ticker).option_chain(opt[0])  # get first expiry options chain

put_volume = option_chain.puts['volume'].sum()
call_volume = option_chain.calls['volume'].sum()

pcr = put_volume / call_volume if call_volume != 0 else np.nan
df['PCR'] = pcr  # This is a single scalar; you’ll want a time series for modeling


# 4. Drop NaNs
df.dropna(inplace=True)

# 5. Scale features
features = ['Close','RSI','MACD','Signal','MA_20','Volume','VIX','PCR']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
X, y = [], []
seq_len = 60
for i in range(len(scaled) - seq_len):
    X.append(scaled[i:i+seq_len])
    y.append(scaled[i+seq_len, 0])  # predicting close price

X, y = np.array(X), np.array(y)
split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = X_train.reshape(len(X_train), seq_len, len(features))
X_test = X_test.reshape(len(X_test), seq_len, len(features))

# 6. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, len(features))),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile('adam', 'mse')
cb = [EarlyStopping(patience=5, restore_best_weights=True),
      ModelCheckpoint('best_model.h5', save_best_only=True)]
model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=cb)

# 7. Predict & Undo Scaling
y_pred = model.predict(X_test)
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1),np.zeros((len(y_test),len(features)-1)))) )[:,0]
y_pred_actual = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), len(features)-1)))))[:,0]

# 8. Metrics & chart
print("MAE:", mean_absolute_error(y_test_actual,y_pred_actual))
print("R2:", r2_score(y_test_actual, y_pred_actual))
plt.plot(y_test_actual,label='Actual'); plt.plot(y_pred_actual,label='Predicted'); plt.legend(); plt.show()

# 9. Print raw table with Open, Actual, Predicted, RSI, MACD, VIX, PCR
offset = seq_len + split
table = df[['Open','Close','RSI','MACD','VIX','PCR']].iloc[offset:].copy()
table['Predicted'] = y_pred_actual
print(table.tail(10)[['Open','Close','Predicted','RSI','MACD','VIX','PCR']])
