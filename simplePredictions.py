import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from utilities.data_helpers import load_data
from modal_data_processors.data_preprocessing import (
    add_technical_indicators,
    drop_na_rows,
    scale_features,
    create_sequences,
    split_train_test_data,
    reshape_for_lstm,
)
from constants.stock_constants import STOCK_FEATURES
from modals.LSTM.lstm_modal import (
    build_lstm_model,
    compile_model,
    prepare_callback,
    train_model,
    predict,
    inverse_transform_zeros_reshaped,
)

# 1. Download AAPL & VIX
ticker = "AAPL"
# df = yf.download(ticker, '2020-01-01', '2024-01-01')
df = load_data(ticker, "/Volumes/Bairon/ModalTrain/Data")
vix = load_data("^VIX", "/Volumes/Bairon/ModalTrain/Data")["Close"]

# 2. Compute Indicators
# df['RSI'] = RSIIndicator(df['Close'].squeeze()).rsi()
# macd = MACD(df['Close'].squeeze())
# df['MACD'] = macd.macd()
# df['Signal'] = macd.macd_signal()
# df['MA_20'] = df['Close'].rolling(20).mean()
# df['Volume'] = df['Volume']
# df['VIX'] = vix.reindex(df.index)
df = add_technical_indicators(df, vix)

# 3. Drop NaNs
# df.dropna(inplace=True)
df = drop_na_rows(df)

# 4. Scale features
# features = ['Close','RSI','MACD','Signal','MA_20','Volume','VIX']
# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(df[STOCK_FEATURES])
scaled, scaler = scale_features(df, STOCK_FEATURES)

# 5. Create sequences
X, y_price, y_dir = [], [], []
seq_len = 60

# for i in range(len(scaled) - seq_len - 1):
#     X.append(scaled[i:i+seq_len])
#     # Predict price (Close)
#     y_price.append(scaled[i+seq_len, 0])
#     # Predict direction: 1 if next close > current close else 0
#     y_dir.append(1 if scaled[i+seq_len,0] > scaled[i+seq_len-1,0] else 0)

# X, y_price, y_dir = np.array(X), np.array(y_price), np.array(y_dir)

X, y_price, y_dir = create_sequences(scaled, seq_len)


# Split train/test
# split = int(0.8*len(X))
# X_train, X_test = X[:split], X[split:]
# y_price_train, y_price_test = y_price[:split], y_price[split:]
# y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]
split, X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = (
    split_train_test_data(X, y_price, y_dir, train_size=0.8)
)


# Reshape for LSTM
# X_train = X_train.reshape(len(X_train), seq_len, len(STOCK_FEATURES))
# X_test = X_test.reshape(len(X_test), seq_len, len(STOCK_FEATURES))
X_train, X_test = reshape_for_lstm(X_train, X_test, seq_len, len(STOCK_FEATURES))
neurons = 64
dropout_rate = 0.2
output_units = 1
# 5a. Build LSTM model for price
# model_price = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(seq_len, len(STOCK_FEATURES))),
#     Dropout(0.2),
#     LSTM(64),
#     Dropout(0.2),
#     Dense(1)
# ])
model_price = build_lstm_model(
    neurons=neurons,
    return_sequences=True,
    input_shape=(seq_len, len(STOCK_FEATURES)),
    dropout_rate=dropout_rate,
    output_units=output_units,
)


# model_price.compile("adam", "mse")
model_price = compile_model(model_price, loss="mse", optimizer="adam", metrics=["mae"])

# cb = [
#     EarlyStopping(patience=5, restore_best_weights=True),
#     ModelCheckpoint("best_model_price.h5", save_best_only=True),
# ]
cb = prepare_callback(monitor="val_loss", patience=5, filepath="best_model_price.h5")

# model_price.fit(
#     X_train, y_price_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=cb
# )
train_model(
    model=model_price,
    X_train=X_train,
    y_train=y_price_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=cb,
)

# 5b. Build LSTM model for direction
# model_dir = Sequential(
#     [
#         LSTM(64, return_sequences=True, input_shape=(seq_len, len(STOCK_FEATURES))),
#         Dropout(0.2),
#         LSTM(64),
#         Dropout(0.2),
#         Dense(1, activation="sigmoid"),
#     ]
# )
model_dir = build_lstm_model(
    neurons=neurons,
    return_sequences=True,
    input_shape=(seq_len, len(STOCK_FEATURES)),
    dropout_rate=dropout_rate,
    output_units=output_units,
    activation="sigmoid"
)

# model_dir.compile("adam", "binary_crossentropy", metrics=["accuracy", "Precision", "Recall"])
model_dir = compile_model(model_dir, loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "Precision", "Recall"])

# cb_dir = [
#     EarlyStopping(patience=5, restore_best_weights=True),
#     ModelCheckpoint("best_model_dir.h5", save_best_only=True),
# ]
cb_dir = prepare_callback(monitor="val_loss", patience=5, filepath="best_model_dir.h5")

# model_dir.fit(
#     X_train,
#     y_dir_train,
#     validation_split=0.1,
#     epochs=50,
#     batch_size=32,
#     callbacks=cb_dir,
# )
train_model(
    model=model_dir,
    X_train=X_train,
    y_train=y_dir_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=cb_dir,
)

# 6. Predict & Undo Scaling
# y_pred_price = model_price.predict(X_test)
y_pred_price = predict(model_price, X_test)

# y_pred_price_actual = scaler.inverse_transform(
#     np.hstack((y_pred_price, np.zeros((len(y_pred_price), len(STOCK_FEATURES) - 1))))
# )[:, 0]

# y_test_price_actual = scaler.inverse_transform(
#     np.hstack(
#         (
#             y_price_test.reshape(-1, 1),
#             np.zeros((len(y_price_test), len(STOCK_FEATURES) - 1)),
#         )
#     )
# )[:, 0]

y_pred_price_actual, y_test_price_actual = inverse_transform_zeros_reshaped(
    scaler, STOCK_FEATURES, y_pred_price, y_price_test
)

# y_pred_dir = model_dir.predict(X_test).flatten()
y_pred_dir = predict(model_dir, X_test).flatten()

y_pred_dir_class = (y_pred_dir > 0.5).astype(int)

# 7. Metrics
print("Price MAE:", mean_absolute_error(y_test_price_actual, y_pred_price_actual))
print("Price R2:", r2_score(y_test_price_actual, y_pred_price_actual))
print("Direction Accuracy:", accuracy_score(y_dir_test, y_pred_dir_class))

# 8a. Price prediction reporting
offset = seq_len + split
table_price = (
    df[["Open", "High", "Low", "Close", "RSI", "MACD", "VIX"]].iloc[offset:].copy()
)

# Align lengths
min_len = min(len(table_price), len(y_pred_price_actual))
table_price = table_price.iloc[-min_len:]
y_pred_price_actual = y_pred_price_actual[-min_len:]

table_price["PredictedPrice"] = y_pred_price_actual
print("\n--- Price Prediction Report ---")
print(
    table_price.tail(10)[
        ["Open", "High", "Low", "Close", "PredictedPrice", "RSI", "MACD", "VIX"]
    ]
)

# 8b. Direction prediction reporting
table_dir = df[["Close"]].iloc[offset:].copy()

# Align lengths
min_len = min(len(table_dir), len(y_pred_dir_class))
table_dir = table_dir.iloc[-min_len:]
y_pred_dir_class = y_pred_dir_class[-min_len:]

table_dir["PredictedDirection"] = y_pred_dir_class
print("\n--- Direction Prediction Report ---")
print(table_dir.tail(10))

# 9. Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test_price_actual, label="Actual Price")
plt.plot(y_pred_price_actual, label="Predicted Price")
plt.legend()
plt.title("AAPL Price Prediction (Test Set)")
plt.show()
