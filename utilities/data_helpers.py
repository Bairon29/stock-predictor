import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# === Setup Storage ===
# def setup_storage(folder_name="stock_data", gdrive_path=None):
#     """
#     Sets up Google Drive storage path locally.
#     Example gdrive_path: '/Users/yourusername/Library/CloudStorage/GoogleDrive-your@email/My Drive'
#     """
#     if gdrive_path is None:
#         raise ValueError("⚠️ Please provide your local Google Drive path")

#     base_path = os.path.join(gdrive_path, folder_name)
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)

#     return base_path


# === Fetch Data ===
def fetch_data(symbol, start="2016-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    print(f"📥 Fetching {symbol} data from {start} to {end}")
    return yf.download(symbol, start=start, end=end)


# === Load Data ===
# def load_data(symbol, folder):
#     filepath = os.path.join(folder, f"{symbol}.csv")
#     if os.path.exists(filepath):
#         print(f"📂 Loading existing data for {symbol} from {filepath}")
#         file = pd.read_csv(filepath, index_col="Date", parse_dates=True)
#         return file
#     return None
def load_data(symbol, folder):
    filepath = os.path.join(folder, f"{symbol}.csv")
    if os.path.exists(filepath):
        print(f"📂 Loading existing data for {symbol} from {filepath}")
        # Read with two header rows and set "Date" as index
        file = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)
        return file
    return None


# === Merge Data ===
def merge_data(old_df, new_df):
    if old_df is None:
        return new_df
    combined = pd.concat([old_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined


# === Save Data ===
def save_data(df, symbol, folder):
    filepath = os.path.join(folder, f"{symbol}.csv")
    df.to_csv(filepath, index=True)
    print(f"✅ Saved {symbol} data to {filepath}")


# === Update Stock ===
def update_stock(symbol, start="2016-01-01", end=None, folder_path="stock_data"):
    old_data = load_data(symbol, folder_path)
    new_data = fetch_data(symbol, start, end)
    combined = merge_data(old_data, new_data)
    save_data(combined, symbol, folder_path)
    return combined


# === Example Usage ===
if __name__ == "__main__":
    # 👇 CHANGE this to your actual Google Drive local path
    EXTERNAL_DRIVE_PATH = "/Volumes/Bairon/ModalTrain/Data"
    # storage_folder = "Data"
    symbols = ["AAPL", "AMD", "MSFT", "^VIX"]

    for symbol in symbols:
        df = update_stock(symbol, start="2019-11-01", end="2024-01-01",
                          folder_path=EXTERNAL_DRIVE_PATH)
        print(df.tail())
        # stock = load_data(symbol, EXTERNAL_DRIVE_PATH)
        # print(stock.tail())
