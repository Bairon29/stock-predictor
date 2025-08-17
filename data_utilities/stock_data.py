import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# -------------------------
# Utility Functions
# -------------------------

def load_existing_data(symbol: str, folder: str = "data") -> pd.DataFrame:
    """Load existing CSV data for a symbol if it exists, else return empty DataFrame."""
    filepath = os.path.join(folder, f"{symbol}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col="Date", parse_dates=True)
    return pd.DataFrame()


def fetch_new_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch new stock data from yfinance for a given symbol and date range."""
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        print(f"⚠️ No data returned for {symbol} between {start} and {end}")
    return df


def merge_data(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge old and new data, remove duplicates, and keep sorted by date."""
    if existing.empty:
        combined = new
    else:
        combined = pd.concat([existing, new])
    # Remove duplicate dates if overlap
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


def save_data(symbol: str, df: pd.DataFrame, folder: str = "data") -> None:
    """Save DataFrame to CSV in the data folder."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{symbol}.csv")
    df.to_csv(filepath, index=True)
    print(f"✅ Saved {symbol} data to {filepath}")


def update_symbol_data(symbol: str, start: str, end: str, folder: str = "data"):
    """Load existing, fetch new, merge, and save final dataset for a symbol."""
    print(f"📥 Updating {symbol} data...")
    existing = load_existing_data(symbol, folder)
    new_data = fetch_new_data(symbol, start, end)
    combined = merge_data(existing, new_data)
    save_data(symbol, combined, folder)


def update_multiple_symbols(symbols: list, start: str, end: str, folder: str = "data"):
    """Update multiple stock symbols in one go."""
    for symbol in symbols:
        update_symbol_data(symbol, start, end, folder)


# -------------------------
# Main Script
# -------------------------

if __name__ == "__main__":
    # Example: download AAPL and AMD history
    symbols = ["AAPL", "AMD"]
    start_date = "2016-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    update_multiple_symbols(symbols, start=start_date, end=end_date)
