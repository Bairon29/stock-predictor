import os
import pandas as pd
import yfinance as yf
from time import sleep
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


# Write a function to save a stock to a CSV file
# For a single stock, save the symbol, company name,
# start date (This date represents the start date passed as parameter to the yf.download function)
# end date (This date represents the end date passed as parameter to the yf.download function)
# The stock info should be saved in the save CVS as the rest of the stock data
# The function should append the stock info to the CSV file if it already exists
# sort the CSV file by symbol in ascending order


def save_stock_info(
    symbols, file_name, folder_path, company_names, start_date, end_date
):
    """
    Saves stock info for multiple symbols into stock_info.csv

    Args:
        symbols (list[str]): List of stock symbols
        company_names (list[str]): List of company names corresponding to symbols
        start_date (str): Start date (same for all)
        end_date (str): End date (same for all)
        folder_path (str): Folder to save stock_info.csv
    """
    info_filepath = os.path.join(folder_path, f"{file_name}.csv")

    # Build DataFrame for all stocks
    new_info = pd.DataFrame(
        {
            "Symbol": symbols,
            "Company Name": company_names,
            "Start Date": [start_date] * len(symbols),
            "End Date": [end_date] * len(symbols),
        }
    ).set_index("Symbol")

    # If file exists, merge with existing info
    if os.path.exists(info_filepath):
        existing_info = load_data("stock_info", folder_path)
        combined_info = merge_data(existing_info, new_info)
    else:
        combined_info = new_info

    # Save merged DataFrame
    save_data(combined_info, "stock_info", folder_path)
    print(f"✅ Saved stock info for {len(symbols)} symbols to {info_filepath}")


# def save_stock_info(symbol, company_name, start_date, end_date, folder_path="stock_data"):
#     info_filepath = os.path.join(folder_path, "stock_info.csv")
#     new_info = pd.DataFrame({
#         "Symbol": [symbol],
#         "Company Name": [company_name],
#         "Start Date": [start_date],
#         "End Date": [end_date]
#     }).set_index("Symbol")

#     if os.path.exists(info_filepath):
#         existing_info = pd.read_csv(info_filepath)
#         combined_info = merge_data(existing_info, new_info)
#     else:
#         combined_info = new_info

#     # combined_info.to_csv(info_filepath, index=False)
#     save_data(combined_info, "stock_info", folder_path)
#     print(f"✅ Saved stock info for {symbol} to {info_filepath}")


# === Example Usage ===
if __name__ == "__main__":
    # 👇 CHANGE this to your actual Google Drive local path
    EXTERNAL_DRIVE_PATH = "/Volumes/Bairon/ModalTrain/Data"
    EXTERNAL_DRIVE_PATH_STOCKS_INFO = "/Volumes/Bairon/ModalTrain/LastFetchedStocksInfo"
    # storage_folder = "Data"
    high_volume_stocks = [
        "GDX",  # VanEck Gold Miners ETF
        "QQQ",  # Invesco QQQ Trust,
        "DIA",  # SPDR Dow Jones Industrial Average ETF Trust
        "MSFT",  # Microsoft,
        "SPY",  # SPDR S&P 500 ETF Trust
        "GE",  # General Electric
        "EEM",  # iShares MSCI Emerging Markets ETF
        "BAC",  # Bank of America
        "AMD",  # Advanced Micro Devices
        "XLF",  # Financial Select Sector SPDR
        "TSLA",  # Tesla
        "PLTR",  # Palantir Technologies
        "AAPL",  # Apple
        "AMZN",  # Amazon
        "F",  # Ford Motor
        "NIO",  # NIO Inc.
        "NVDA",  # Nvidia
        "INTC",  # Intel
        "HPE",  # Hewlett Packard Enterprise
        "SOXL",  # Direxion Daily Semiconductor Bear ETF
        "TQQQ",  # ProShares UltraPro QQQ
    ]

    for symbol in high_volume_stocks:
        df = update_stock(
            symbol,
            start="1980-01-01",
            end="2001-02-31",
            folder_path=EXTERNAL_DRIVE_PATH,
        )
        sleep(6)  # To respect rate limits
        print(df.tail())
    # stock = load_data(symbol, EXTERNAL_DRIVE_PATH)
    # print(stock.tail())

    ## Save list of stocks info
    # symbols, file_name, folder_path, company_names, start_date, end_date
    # save_stock_info(
    #     symbols=high_volume_stocks,
    #     file_name="stocks_info",
    #     folder_path=EXTERNAL_DRIVE_PATH_STOCKS_INFO,
    #     company_names=[
    #         "VanEck Gold Miners ETF",
    #         "Invesco QQQ Trust",
    #         "SPDR Dow Jones Industrial Average ETF Trust",
    #         "Microsoft",
    #         "SPDR S&P 500 ETF Trust",
    #         "General Electric",
    #         "iShares MSCI Emerging Markets ETF",
    #         "Bank of America",
    #         "Advanced Micro Devices",
    #         "Financial Select Sector SPDR",
    #         "Tesla",
    #         "Palantir Technologies",
    #         "Apple",
    #         "Amazon",
    #         "Ford Motor",
    #         "NIO Inc.",
    #         "Nvidia",
    #         "Intel",
    #         "Hewlett Packard Enterprise",
    #         "Direxion Daily Semiconductor Bear ETF",
    #         "ProShares UltraPro QQQ",
    #     ],
    #     start_date="2000-01-01",
    #     end_date="2025-01-01",
    # )
