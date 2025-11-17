import os
import pandas as pd
import yfinance as yf
from time import sleep
from datetime import datetime, timedelta

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


def fetch_intraday(symbols, start_date, end_date, sleep_time=6, output_dir="data"):
    """
    Fetch 1-min intraday data from Yahoo Finance in 7-day batches,
    moving backwards from start_date to end_date. Saves each symbol
    as a CSV file named <SYMBOL>.csv.

    Args:
        symbols (list): List of stock tickers.
        start_date (str): Starting date (closest to present), format 'YYYY-MM-DD'.
        end_date (str): Ending date (further in the past), format 'YYYY-MM-DD'.
        sleep_time (int): Seconds to sleep between API calls.
        output_dir (str): Directory to save CSV files.

    Returns:
        dict: {symbol: DataFrame of intraday OHLCV}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        all_data = []

        current_end = start_dt
        while current_end > end_dt:
            current_start = max(end_dt, current_end - timedelta(days=7))

            print(f"  {symbol}: {current_start.date()} → {current_end.date()}")

            try:
                df = yf.download(
                    symbol,
                    interval="1m",
                    start=current_start.strftime("%Y-%m-%d"),
                    end=current_end.strftime("%Y-%m-%d"),
                    progress=False,
                )

                if not df.empty:
                    all_data.append(df)

                    # Save incremental data to CSV
                    file_path = os.path.join(output_dir, f"{symbol}.csv")
                    if os.path.exists(file_path):
                        df.to_csv(file_path, mode="a", header=False)
                    else:
                        df.to_csv(file_path, mode="w", header=True)

            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")

            # Move the window back 7 days
            current_end = current_start
            sleep(sleep_time)

        if all_data:
            results[symbol] = pd.concat(all_data).sort_index()
        else:
            results[symbol] = pd.DataFrame()

    return results


# === Load Data ===
# def load_data(symbol, folder):
#     filepath = os.path.join(folder, f"{symbol}.csv")
#     if os.path.exists(filepath):
#         print(f"📂 Loading existing data for {symbol} from {filepath}")
#         file = pd.read_csv(filepath, index_col="Date", parse_dates=True)
#         return file
#     return None
def load_data(
    symbol, folder, has_multi_level_headers=False, index_col="Symbol", skiprows=0
):
    filepath = os.path.join(folder, f"{symbol}.csv")
    if os.path.exists(filepath):
        print(f"📂 Loading existing data for {symbol} from {filepath}")
        # Read with two header rows and set "Date" as index
        file = (
            pd.read_csv(
                filepath,
                header=1,
                index_col=index_col,
                parse_dates=True,
                skiprows=skiprows,
            )
            if has_multi_level_headers
            else pd.read_csv(
                filepath, index_col=index_col, parse_dates=True, skiprows=skiprows
            )
        )
        return file
    return None


# === Remove whitespaces from columns names ===
def strip_whitespace_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    return df


# === Reset Data frame index ===
def reset_dataframe_index(df, index_name="Date"):
    return df.reset_index()


# === Set Data Frame Index ===
def set_dataframe_index(df, index_col="Date"):
    if df.index.name is None:
        df.index.name = index_col
    elif df.index.name != index_col:
        df.index.name = index_col
    return df


# === Merge Data ===
def merge_data(old_df, new_df, keep="last"):
    if old_df is None:
        return new_df
    combined = pd.concat([old_df, new_df])
    combined = combined[~combined.index.duplicated(keep=keep)]
    # combined = combined.drop_duplicates(subset=[ "Description", "daily_start", "daily_end", "1min_start", "1min_end"], keep=keep)
    combined.sort_index(inplace=True)
    return combined


# === Save Data ===
def save_data(df, symbol, folder, index=True):
    filepath = os.path.join(folder, f"{symbol}.csv")
    df.to_csv(filepath, index=index)
    print(f"✅ Saved {symbol} data to {filepath}")


# === Update Stock ===
def update_stock(symbol, start="2016-01-01", end=None, folder_path="stock_data"):
    old_data = load_data(symbol, folder_path)
    new_data = fetch_data(symbol, start, end)
    combined = merge_data(old_data, new_data)
    save_data(combined, symbol, folder_path)
    return combined


## Create a function to save a dictionary of stockes info into a single cvs file
# The function should take as parameters:
# stocks info as a dictionary where the key is the stock symbol and the value is a dictionary with the following keys
# "company_name", "start_date", "end_date"
# file_name: name of the file to save the stock info
# folder_path: path to the folder where the file will be saved
# The function should save the stock info into a CSV file with the following columns:
# "Symbol", "Company Name", "Start Date", "End Date"
# The function should append the stock info to the CSV file if it already exists
# sort the CSV file by symbol in ascending order
def save_stocks_info_dict(stocks_info, file_name, folder_path):
    info_filepath = os.path.join(folder_path, f"{file_name}.csv")

    # Build DataFrame from stocks_info dictionary
    new_info = pd.DataFrame.from_dict(stocks_info, orient="index").set_index("Symbol")
    # If file exists, merge with existing info
    if os.path.exists(info_filepath):
        existing_info = load_data(file_name, folder_path)
        print("\n\n\n\n============existing_info===========================")
        print(existing_info)
        print("=======================================\n\n\n\n")
        combined_info = merge_data(existing_info, new_info)
    else:
        combined_info = new_info

    print("\n\n\n\n============combined_info===========================")
    print(combined_info)
    print("=======================================\n\n\n\n")
    # Save merged DataFrame
    save_data(combined_info, file_name, folder_path)
    print(f"✅ Saved stock info for {len(stocks_info)} symbols to {info_filepath}")


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
    # high_volume_stocks = [
    #     "GDX",  # VanEck Gold Miners ETF
    #     "QQQ",  # Invesco QQQ Trust,
    #     "DIA",  # SPDR Dow Jones Industrial Average ETF Trust
    #     "MSFT",  # Microsoft,
    #     "SPY",  # SPDR S&P 500 ETF Trust
    #     "GE",  # General Electric
    #     "EEM",  # iShares MSCI Emerging Markets ETF
    #     "BAC",  # Bank of America
    #     "AMD",  # Advanced Micro Devices
    #     "XLF",  # Financial Select Sector SPDR
    #     "TSLA",  # Tesla
    #     "PLTR",  # Palantir Technologies
    #     "AAPL",  # Apple
    #     "AMZN",  # Amazon
    #     "F",  # Ford Motor
    #     "NIO",  # NIO Inc.
    #     "NVDA",  # Nvidia
    #     "INTC",  # Intel
    #     "HPE",  # Hewlett Packard Enterprise
    #     "SOXL",  # Direxion Daily Semiconductor Bear ETF
    #     "TQQQ",  # ProShares UltraPro QQQ
    # ]

    # for symbol in high_volume_stocks:
    #     df = update_stock(
    #         symbol,
    #         start="1980-01-01",
    #         end="2001-02-31",
    #         folder_path=EXTERNAL_DRIVE_PATH,
    #     )
    #     sleep(6)  # To respect rate limits
    #     print(df.tail())
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

    # Example usage of fetch_intraday
    # symbols = ["AAPL", "MSFT"]
    # data = fetch_intraday(
    #     symbols,
    #     start_date="2025-08-31",  # closest to present
    #     end_date="2025-07-01",  # furthest back
    #     sleep_time=6,
    #     output_dir="intraday_data",
    # )

    # print("\nAAPL sample data:")
    # print(data["AAPL"].head())

    # === Example Usage save_stocks_info_dict ===
    stocks_info = {
        "AAPL": {
            "Symbol": "AAPL",
            "Company Name": "Apple Inc.",
            "Start Date": "2020-01-01",
            "End Date": "2025-01-01",
        },
        #  "ARM": {
        #     "Symbol": "ARM",
        #     "Company Name": "Arm Holdings plc",
        #     "Start Date": "2020-01-01",
        #     "End Date": "2025-01-01",
        # }
    }
    save_stocks_info_dict(
        stocks_info=stocks_info,
        file_name="stocks_info",
        folder_path=EXTERNAL_DRIVE_PATH_STOCKS_INFO,
    )
