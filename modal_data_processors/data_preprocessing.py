import yfinance as yf
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(symbol):
    df = yf.download(symbol, '2020-01-01', '2024-01-01')
    return df
    
def add_technical_indicators(stockDataFrame, vix):
    # 2. Compute Indicators
    stockDataFrame['RSI'] = RSIIndicator(stockDataFrame['Close'].squeeze()).rsi()
    macd = MACD(stockDataFrame['Close'].squeeze())
    stockDataFrame['MACD'] = macd.macd()
    stockDataFrame['Signal'] = macd.macd_signal()
    stockDataFrame['MA_20'] = stockDataFrame['Close'].rolling(20).mean()
    stockDataFrame['Volume'] = stockDataFrame['Volume']
    stockDataFrame['VIX'] = vix.reindex(stockDataFrame.index)
    
    return stockDataFrame

def drop_na_rows(df):
    df.dropna(inplace=True)
    return df
    
def scale_features(df, features):
    # 4. Scale features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler
    
def create_sequences(scaled_data, seq_len):
    sample_inputs, y_price, y_dir = [], [], []

    for i in range(len(scaled_data) - seq_len - 1):
        sample_inputs.append(scaled_data[i:i+seq_len])
        # Predict price (Close)
        y_price.append(scaled_data[i+seq_len, 0])
        # Predict direction: 1 if next close > current close else 0
        y_dir.append(1 if scaled_data[i+seq_len,0] > scaled_data[i+seq_len-1,0] else 0)

    sample_inputs, y_price, y_dir = np.array(sample_inputs), np.array(y_price), np.array(y_dir)
    return sample_inputs, y_price, y_dir

def split_train_test_data(sample_date_set, y_price, y_dir, train_size=0.8):
    # Split train/test
    split = int(train_size*len(sample_date_set))
    train_set, test_set = sample_date_set[:split], sample_date_set[split:]
    y_price_train, y_price_test = y_price[:split], y_price[split:]
    y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]
    return split, train_set, test_set, y_price_train, y_price_test, y_dir_train, y_dir_test

def reshape_for_lstm(train_set, test_set, seq_len, num_features):
    # Reshape for LSTM
    new_train_set = train_set.reshape(len(train_set), seq_len, num_features)
    new_test_set = test_set.reshape(len(test_set), seq_len, num_features)
    return new_train_set, new_test_set