
from utilities.data_helpers import (
    save_stocks_info_dict,
)
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
